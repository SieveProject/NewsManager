"""Resumo legível da run de extração, para acompanhar sem SSH.

    .venv/bin/python deploy/status.py > logs/status.txt

Chamado pelo deploy/backup.sh a cada passada, que sobe o resultado para a nuvem
como status.txt -- dá para abrir pelo celular com o Mac desligado. Só lê:
segmentos Parquet, WAL, `ollama ps`, tmux e o log do backup.

A versão acompanhada é o diretório data/extractions/v=* modificado por último,
ou seja, o da run em andamento.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from newsmanager.extract.units import units_glob  # noqa: E402
from newsmanager import config  # noqa: E402


def _sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001 - a status line must never crash the backup
        return f"(erro: {e})"


def main() -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    print(f"NewsManager -- status da extração   {now:%Y-%m-%d %H:%M} UTC")
    print("=" * 60)

    versions = sorted((ROOT / "data" / "extractions").glob("v=*"), key=lambda p: p.stat().st_mtime)
    if not versions:
        print("nenhuma extração encontrada")
        return
    v = versions[-1]
    runs = v / "runs"
    wal = sum(1 for _ in open(v / "wal-000.jsonl", encoding="utf-8", errors="replace")) \
        if (v / "wal-000.jsonl").exists() else 0

    con = duckdb.connect()
    total = con.execute(f"SELECT count(*) FROM read_parquet('{units_glob(config.load())}')").fetchone()[0]
    if any(runs.glob("*.parquet")):
        r = f"read_parquet('{runs}/*.parquet')"
        done, failed, model = con.execute(
            f"SELECT count(*), count(*) FILTER (WHERE status <> 'ok'), any_value(model) FROM {r}").fetchone()
        last = con.execute(f"SELECT max(extracted_at) FROM {r}").fetchone()[0]
        # Rate over the last hour of segments -- or over the whole run while it
        # is younger than an hour, else a 5-minute-old run reads as 0.1/s.
        recent, first = con.execute(
            f"SELECT count(*), min(extracted_at) FROM {r} WHERE extracted_at > ?",
            [last - timedelta(hours=1)]).fetchone()
        window_s = max((last - first).total_seconds(), 60) if first else 3600
        n_rel = con.execute(f"SELECT count(*) FROM read_parquet('{v}/relations/*.parquet')").fetchone()[0] \
            if any((v / "relations").glob("*.parquet")) else 0
    else:
        done = failed = recent = n_rel = 0
        model, last, window_s = "?", None, 3600

    done_all = done + wal
    pct = 100 * done_all / total if total else 0
    rate = recent / window_s
    left = max(total - done_all, 0)
    eta_h = left / rate / 3600 if rate else float("inf")

    print(f"versão        {v.name}   modelo {model}")
    print(f"progresso     {done_all:,} / {total:,} unidades ({pct:.1f}%)")
    print(f"falhas        {failed:,} ({100 * failed / done if done else 0:.2f}%)")
    print(f"tuplas        {n_rel:,}")
    print(f"ritmo         {rate:.2f}/s ({rate * 3600:,.0f}/h, última hora gravada)")
    if rate:
        end = now + timedelta(hours=eta_h)
        print(f"previsão      ~{eta_h:.0f} h  (término ~{end:%d/%m %H:%M} UTC, ~US$ {eta_h * 0.456:.0f} a mais)")
    if last:
        age = (now - last).total_seconds() / 60
        flag = "" if age < 30 else "   <-- ATENÇÃO: nada gravado há mais de 30 min"
        print(f"último dado   {last:%H:%M} UTC (há {age:.0f} min){flag}")

    print()
    ps = _sh("ollama ps | tail -n +2")
    gpu = "100% GPU" in ps
    print(f"GPU           {'ok, 100% GPU' if gpu else 'ATENÇÃO: ' + (ps or 'nenhum modelo carregado')}")
    tmux = _sh("tmux ls 2>/dev/null | cut -d: -f1 | tr '\\n' ' '")
    print(f"tmux          {tmux or 'nenhuma sessão'}"
          f"{'' if 'run' in tmux.split() else '   <-- ATENÇÃO: sessão run ausente'}")
    util = _sh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
    print(f"nvidia-smi    {util}")
    restarts = _sh(f"grep -c reiniciando {ROOT}/logs/worker-0.log 2>/dev/null; true") or "0"
    print(f"reinícios     {restarts}")
    backup = _sh(f"grep -E ' OK | FALHA' {ROOT}/logs/backup.log | tail -1")
    print(f"backup        {backup or '(sem registro ainda)'}")
    run_end = _sh(f"grep -a 'EXIT=' {ROOT}/logs/run.log | tail -1")
    if run_end:
        print(f"\nRUN TERMINOU: {run_end}")


if __name__ == "__main__":
    main()
