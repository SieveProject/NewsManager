"""End-to-end test of the extraction pipeline against the mock Ollama server.

Covers units -> partition -> run (multi-worker) -> resume -> collect -> query,
including a worker killed mid-run and injected server failures.

Assumes tests/test_e2e_subset.py has already built data/_test.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from newsmanager import config  # noqa: E402
from newsmanager.extract import benchmark, collect, partition, sink, units, worker  # noqa: E402
from newsmanager.extract.client import OllamaConfig  # noqa: E402
from newsmanager.extract.prompt import load as load_prompt  # noqa: E402

PORT = 11577
HOST = f"http://127.0.0.1:{PORT}"
REPO = Path(__file__).resolve().parent.parent


def wait_up(timeout=15):
    for _ in range(int(timeout * 10)):
        try:
            if httpx.get(f"{HOST}/api/tags", timeout=1).status_code == 200:
                return True
        except Exception:
            time.sleep(0.1)
    return False


def main() -> int:
    root = REPO / "data" / "_test"
    if not (root / "curated" / "documents").exists():
        print("run tests/test_e2e_subset.py first", file=sys.stderr)
        return 2

    cfg = replace(
        config.load(), root=root, manifest=root / "manifest.json", raw=root / "raw",
        curated=root / "curated", marts=root / "marts", db=root / "test.duckdb",
        tmp=root / "tmp", memory_limit="1GB",
    )
    shutil.rmtree(root / "extractions", ignore_errors=True)
    shutil.rmtree(root / "curated" / "relations", ignore_errors=True)
    shutil.rmtree(root / "curated" / "extraction_units", ignore_errors=True)

    print("=== units ===")
    ustats = units.build(cfg)
    assert ustats["units"] > 0
    assert ustats["units"] <= ustats["documents"]

    print("\n=== partition (4 workers) ===")
    rows = partition.summarize(cfg, 4)
    assert sum(r["units"] for r in rows) == ustats["units"], "partition must cover every unit exactly once"
    print("  coverage: OK, every unit assigned exactly once")

    srv = subprocess.Popen(
        [sys.executable, str(REPO / "tests" / "mock_ollama.py"),
         "--port", str(PORT), "--fail-rate", "0.05", "--latency", "0.005"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        assert wait_up(), "mock ollama did not start"
        prompt = load_prompt(REPO / "prompts" / "extraction.txt", REPO / "prompts" / "schema.json", 8000)
        oll = OllamaConfig(host=HOST, model="mock-model:latest", concurrency=8,
                           num_ctx=4096, timeout_s=30, retries=3)
        print(f"prompt_version={prompt.version}")

        print("\n=== num_ctx guard ===")
        from newsmanager.extract.cli import main as xmain
        rc = subprocess.run(
            [sys.executable, "-m", "newsmanager.extract", "run",
             "--num-ctx", "512", "--max-chars", "8000", "--host", HOST],
            cwd=REPO, capture_output=True, text=True,
        )
        assert rc.returncode != 0 and "num-ctx" in (rc.stderr + rc.stdout), "undersized num_ctx must be rejected"
        print("  undersized num_ctx rejected: OK")

        print("\n=== queda da VM no meio da escrita ===")
        # Simula o estado real de uma VM morta: registros gravados no WAL, sem
        # flush e sem close. O objeto é abandonado, como num SIGKILL.
        crash_root = REPO / "data" / "_test" / "extractions" / "v=crashtest"
        shutil.rmtree(crash_root, ignore_errors=True)
        crash_root.mkdir(parents=True, exist_ok=True)
        s = sink.ParquetSink(crash_root, 0, "crashtest", "mock", flush_every=10_000)
        for i in range(15):
            s.append({"unit_id": f"u{i}", "repr_doc_id": f"d{i}",
                      "published_at": "2023-05-01 12:00:00", "year": 2023,
                      "symbols": ["AAPL"], "status": "ok", "error": None,
                      "truncated": False, "body_chars": 1000, "prompt_tokens": 10,
                      "output_tokens": 5, "latency_s": 0.1,
                      "tuples": [{"agent_a": "A", "agent_b": "B",
                                  "relation_type": "supplies",
                                  "direction": "positive", "strength": 0.5}]})
        del s  # sem close(): nada foi para parquet ainda

        wal = crash_root / "wal-000.jsonl"
        assert wal.exists() and wal.stat().st_size > 0, "WAL deveria ter os registros"
        assert not list((crash_root / "runs").glob("*.parquet")), "nada deveria estar em parquet ainda"
        assert len(sink.recover_done(crash_root, 0)) == 15, "os 15 devem estar visíveis no WAL"

        # Escrita interrompida no meio de uma linha.
        with wal.open("a") as fh:
            fh.write('{"unit_id": "truncado-parcial')
        assert len(sink.recover_done(crash_root, 0)) == 15, "linha truncada deve ser ignorada"
        assert sink.heal_partial_line(wal), "linha parcial deveria ser detectada"
        assert not sink.heal_partial_line(wal), "cura deve ser idempotente"

        # Na retomada o WAL órfão vira segmento parquet, sem perder nada.
        n = sink.absorb_wal(crash_root, 0, "crashtest", "mock")
        assert n == 15, f"absorb_wal recuperou {n}, esperado 15"
        assert not wal.exists(), "WAL deve ser consumido após absorção"
        assert list((crash_root / "runs").glob("*.parquet")), "deve existir segmento parquet"
        assert len(sink.recover_done(crash_root, 0)) == 15, "registros devem sobreviver em parquet"

        import pyarrow.parquet as _pq
        rel = list((crash_root / "relations").glob("*.parquet"))
        assert rel and _pq.read_table(rel[0]).num_rows == 15, "tuplas devem sobreviver também"
        shutil.rmtree(crash_root, ignore_errors=True)
        print("  0 registros perdidos numa queda + linha truncada; WAL -> parquet: OK")

        print("\n=== parcial (worker 0, limit 15) + retomada ===")
        s1 = worker.run(cfg, prompt, oll, 0, 4, limit=15, checkpoint_every=5, progress_every=1000)
        root = worker.output_root(cfg, prompt.version)
        assert s1.attempted == 15
        assert len(sink.recover_done(root, 0)) == 15, "os 15 devem estar persistidos"
        print("  15 unidades persistidas em parquet: OK")

        print("\n=== run completa, 4 workers ===")
        total = 0
        for w in range(4):
            st = worker.run(cfg, prompt, oll, w, 4, checkpoint_every=10, progress_every=1000)
            total += st.attempted
        s_resume = worker.run(cfg, prompt, oll, 0, 4, progress_every=1000)
        assert s_resume.attempted == 0, f"retomada refez {s_resume.attempted} unidades"
        print("  retomada após conclusão não refez nada: OK")

        print("\n=== bench ===")
        b = benchmark.run(cfg, prompt, oll, n=16)
        assert b["units_per_s"] > 0

        print("\n=== project ===")
        proj = benchmark.project(cfg, b["units_per_s"], n_vms=4, usd_per_gpu_hour=0.40)
        assert proj["units"] == ustats["units"]

        print("\n=== collect ===")
        cstats = collect.collect(cfg, prompt.version)
        assert cstats["units"] == ustats["units"], f"{cstats['units']} collected vs {ustats['units']} units"
        collect.build_views(cfg, prompt.version)

        print("\n=== query ===")
        import duckdb
        con = duckdb.connect(str(cfg.db), read_only=True)
        print(" relations:", con.execute("SELECT count(*) FROM news.relations").fetchone()[0])
        print(" by symbol:", con.execute("SELECT count(*) FROM news.relations_by_symbol").fetchone()[0])
        print(" top edges:", con.execute(
            "SELECT agent_a, agent_b, direction, n, round(mean_strength,2) "
            "FROM news.agent_edges ORDER BY n DESC LIMIT 5").fetchall())
        print(" failure modes:", con.execute(
            "SELECT status, count(*) FROM news.extraction_runs GROUP BY 1").fetchall())
        bad = con.execute(
            "SELECT count(*) FROM news.relations WHERE strength < 0 OR strength > 1").fetchone()[0]
        assert bad == 0, f"{bad} relações com strength fora de faixa"
        print(" strength normalizado em [0,1]: OK (mock emite -0.1..1.1 de propósito)")

        # Exigência central: toda tupla carrega a data da notícia.
        nulls = con.execute(
            "SELECT count(*) FROM news.relations WHERE published_at IS NULL").fetchone()[0]
        assert nulls == 0, f"{nulls} tuplas sem data"
        print(" toda tupla tem published_at: OK")

        cols = {r[0] for r in con.execute("DESCRIBE news.relations").fetchall()}
        for need in ("agent_a", "agent_b", "relation_type", "direction", "strength", "published_at"):
            assert need in cols, f"coluna {need} ausente"
        print(" schema de 5 campos + data: OK")
        print(" exemplo:", con.execute(
            "SELECT published_at, agent_a, relation_type, direction, strength, agent_b "
            "FROM news.relations ORDER BY published_at DESC LIMIT 3").fetchall())
        con.close()

        print("\n=== RESULT: PASS ===")
        return 0
    finally:
        srv.terminate()
        srv.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
