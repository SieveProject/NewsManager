"""Etapa link de ponta a ponta, sem rede e sem GPU.

Relações sintéticas + listas de tickers locais -> link-prep -> link-run contra o
Ollama simulado (que escolhe o primeiro candidato) -> retomada -> link-build.

    python tests/test_link.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import duckdb
import httpx
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from newsmanager import config  # noqa: E402
from newsmanager.extract import link  # noqa: E402
from newsmanager.extract.client import OllamaConfig  # noqa: E402

PORT = 11578
HOST = f"http://127.0.0.1:{PORT}"
REPO = Path(__file__).resolve().parent.parent
RV = "testlink0001"


def main() -> int:
    root = REPO / "data" / "_test_link"
    shutil.rmtree(root, ignore_errors=True)
    cfg = replace(config.load(), root=root, curated=root / "curated", db=root / "test.duckdb")

    # Normalização
    assert link.norm("The Apple Inc.") == link.norm("Apple") == link.norm("apple, inc") == "apple"
    assert link.norm("U.S. Federal Reserve") == "us federal reserve"
    assert link.norm("Apple's") == "apple"
    assert link.norm("Inc.") == "inc"                       # nunca esvazia
    uni = {"AAPL", "VTEB", "BRK-A"}
    assert link.explicit_ticker("Vanguard Tax-Exempt Bond (VTEB)", uni) == "VTEB"
    assert link.explicit_ticker("Apple (NASDAQ: AAPL)", uni) == "AAPL"
    assert link.explicit_ticker("Berkshire (BRK.A)", uni) == "BRK-A"
    assert link.explicit_ticker("crude oil (WTI)", uni) is None   # fora do universo
    assert link.clean_security_name("Alcoa Corporation Common Stock ") == "Alcoa Corporation"
    assert link.slug("the Federal Reserve") == "FEDERAL_RESERVE"

    # Referências locais
    ref = root / "reference"
    ref.mkdir(parents=True)
    (ref / "price_tickers.txt").write_text("AAPL\nMSFT\nOCN\nXOM\n")
    (ref / "nasdaqlisted.txt").write_text(
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
        "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
        "MSFT|Microsoft Corporation - Common Stock|Q|N|N|100|N|N\n"
        "File Creation Time: 1009202600:00||||||\n")
    (ref / "otherlisted.txt").write_text(
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
        "XOM|Exxon Mobil Corporation Common Stock|N|XOM|N|100|N|XOM\n"
        "File Creation Time: 1009202600:00||||||\n")
    paths = {"price_tickers.txt": ref / "price_tickers.txt",
             "nasdaqlisted.txt": ref / "nasdaqlisted.txt", "otherlisted.txt": ref / "otherlisted.txt"}

    # Relações sintéticas no formato de relations_flagged
    rows = []
    def rel(i, a, b, syms, noise=None):
        rows.append({"unit_id": f"u{i}", "repr_doc_id": f"d{i}", "published_at": None, "year": 2020,
                     "symbols": syms, "agent_a": a, "agent_b": b, "relation_type": "affects",
                     "direction": "positive", "strength": 0.8, "noise": noise})
    for i in range(6):
        rel(i, "Apple Inc.", "Federal Reserve", ["AAPL"])
    rel(10, "Apple", "Microsoft Corp.", ["AAPL", "MSFT"])
    rel(11, "Ocwen Financial (OCN)", "regulators", ["OCN"])
    rel(12, "Ocwen Financial Corp.", "Exxon", ["OCN", "XOM"])
    rel(13, "Ocwen Financial", "the Fed", ["OCN"])
    rel(14, "investors", "Apple", ["AAPL"], noise="generic")
    rel(15, "European Central Bank", "euro", [])             # sem símbolos: sem candidato
    rd = cfg.curated / "relations" / f"v={RV}"
    rd.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), rd / "relations_flagged.parquet")

    print("=== link-prep ===")
    st = link.prepare(cfg, RV, reference=paths)
    c = duckdb.connect()
    d = link.entities_dir(cfg, RV)
    cand = {k: (cs, ex) for k, cs, ex in c.execute(
        f"SELECT key, candidates, explicit FROM read_parquet('{d}/candidates.parquet')").fetchall()}
    assert "investors" not in cand, "nome só de tupla ruidosa não vira entidade"
    assert cand["apple"][0][0]["ticker"] == "AAPL", cand["apple"]
    assert cand["microsoft"][0][0]["evidence"] == "same company name", cand["microsoft"]
    assert cand["ocwen financial"][1] == "OCN"
    ocn = cand["ocwen financial"][0][0]
    assert "Ocwen" in ocn["name"], ocn                     # deslistado ganha nome das notícias
    assert st["universe"] == 4 and st["keys"] == 9, st
    assert cand["european central bank"][0] == []

    print("=== link-run (mock) ===")
    mock = subprocess.Popen([sys.executable, str(REPO / "tests" / "mock_ollama.py"), "--port", str(PORT)],
                            stdout=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                if httpx.get(f"{HOST}/api/tags", timeout=1).status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)
        oll = OllamaConfig(host=HOST, model="mock-model:latest", num_ctx=3072, num_predict=128,
                           concurrency=4, num_thread=1)
        tp = REPO / "prompts" / "link.txt"
        r1 = link.run(cfg, RV, oll, tp, limit=3, segment=2)
        assert r1["ok"] == 3, r1
        r2 = link.run(cfg, RV, oll, tp, segment=2)           # retomada: só o resto
        assert r2["ok"] == st["keys"] - 3 and r2["version"] == r1["version"], r2
        r3 = link.run(cfg, RV, oll, tp)
        assert r3["ok"] == 0, r3
    finally:
        mock.terminate()

    print("=== link-build ===")
    b = link.build(cfg, RV)
    assert b["pending"] == 0, b
    em = dict(c.execute(f"SELECT key, ticker FROM read_parquet('{d}/entity_map.parquet')").fetchall())
    assert em["apple"] == "AAPL" and em["ocwen financial"] == "OCN", em
    assert em["european central bank"] is None, em         # sem candidato -> NONE
    ids = dict(c.execute(f"SELECT key, entity_id FROM read_parquet('{d}/entity_map.parquet')").fetchall())
    assert ids["european central bank"] == "EUROPEAN_CENTRAL_BANK", ids
    lk = c.execute(f"""SELECT count(*), count(*) FILTER (WHERE ticker_a = 'AAPL')
                       FROM read_parquet('{d}/relations_linked.parquet')""").fetchone()
    assert lk == (len(rows), 7), lk                         # nenhuma linha perdida ou duplicada
    db = duckdb.connect(str(cfg.db))
    assert db.execute("SELECT count(*) FROM news.relations_linked").fetchone()[0] == len(rows)
    db.close()

    shutil.rmtree(root, ignore_errors=True)
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
