"""Etapa prices sem rede: um zip sintético com cada anomalia medida no FNSPID.

    python tests/test_prices.py
"""

from __future__ import annotations

import io
import shutil
import sys
import zipfile
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from newsmanager import config, prices  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def weekdays(start: date, end: date) -> list[date]:
    d, out = start, []
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def csv(rows: list[tuple], vol_first: bool) -> bytes:
    """rows: (date, close, adj_close, volume). Duas ordens de coluna, como no FNSPID."""
    b = io.StringIO()
    if vol_first:
        b.write("date,volume,open,high,low,close,adj close\n")
        for d, c, a, v in reversed(rows):           # o FNSPID vem do mais novo ao mais antigo
            b.write(f"{d},{v},{c},{c},{c},{c},{a}\n")
    else:
        b.write("date,open,high,low,close,adj close,volume\n")
        for d, c, a, v in reversed(rows):
            b.write(f"{d},{c},{c},{c},{c},{a},{v}\n")
    return b.getvalue().encode()


def main() -> int:
    root = REPO / "data" / "_test_prices"
    shutil.rmtree(root, ignore_errors=True)
    cfg = replace(config.load(), root=root, raw=root / "raw", curated=root / "curated",
                  db=root / "test.duckdb", tmp=root / "tmp")
    cfg.ensure_dirs()
    prices.BATCH_END_MIN_TICKERS = 3

    days = [d for d in weekdays(date(2020, 6, 1), date(2020, 7, 31)) if d != date(2020, 7, 3)]
    files: dict[str, bytes] = {}
    # 4 tickers "longos" atravessam a emenda; o lote novo vem em escala 1/4
    for i, t in enumerate(("AAA", "BBB", "CCC", "DDD")):
        rows, px = [], 100.0
        for d in days:
            px *= 1.001
            adj = px / 4 if d >= date(2020, 7, 6) else px
            rows.append((d, adj, adj, 1000))
        if t == "BBB":       # pico de cotação que se reverte
            k = days.index(date(2020, 6, 10))
            d, c, a, v = rows[k]
            rows[k] = (d, c * 20, a * 20, v)
        if t == "CCC":       # buraco: some um pregão
            rows = [r for r in rows if r[0] != date(2020, 6, 17)]
        if t == "DDD":       # preço inválido
            rows.append((date(2020, 6, 6), -1.0, -1.0, 0))   # sábado e negativo
        files[t] = csv(rows, vol_first=(i % 2 == 0))
    # 3 tickers acabam juntos em 2020-07-02: fim de coleta
    for t in ("OLD1", "OLD2", "OLD3"):
        files[t] = csv([(d, 50.0 + j, 50.0 + j, 10) for j, d in enumerate(days) if d <= date(2020, 7, 2)], True)
    # ticker parado: volume 0 e preço repetido
    files["STL"] = csv([(d, 10.0, 10.0, 0 if d > date(2020, 7, 20) else 5) for d in days], False)

    zp = prices.zip_path(cfg)
    zp.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("full_history/", b"")
        z.writestr("__MACOSX/._full_history", b"lixo")
        for t, b in files.items():
            z.writestr(f"full_history/{t}.csv", b)

    prices.build(cfg)
    d = cfg.curated / "prices"
    con = duckdb.connect()
    con.execute(f"CREATE VIEW p AS SELECT * FROM read_parquet('{d / 'daily.parquet'}')")
    con.execute(f"CREATE VIEW q AS SELECT * FROM read_parquet('{d / 'quality.parquet'}')")
    con.execute(f"CREATE VIEW c AS SELECT * FROM read_parquet('{d / 'calendar.parquet'}')")

    flag = lambda t, dt: con.execute("SELECT flag FROM p WHERE ticker = ? AND date = ?", [t, dt]).fetchone()[0]  # noqa: E731
    ret = lambda t, dt: con.execute("SELECT ret FROM p WHERE ticker = ? AND date = ?", [t, dt]).fetchone()[0]  # noqa: E731

    # emendas: retorno do dia inválido (todas as SPLICE_DATES); vizinhos corretos (0,1%)
    for t in ("AAA", "BBB", "CCC", "DDD"):
        assert flag(t, date(2020, 7, 6)) == "splice", t
        assert flag(t, date(2020, 7, 2)) == "splice", t
        assert abs(ret(t, date(2020, 7, 7)) - 0.001) < 1e-9, t
        assert abs(ret(t, date(2020, 7, 1)) - 0.001) < 1e-9, t
    # pico e reversão
    assert flag("BBB", date(2020, 6, 10)) == "spike"
    assert flag("BBB", date(2020, 6, 11)) == "spike_revert"
    assert ret("BBB", date(2020, 6, 9)) is not None
    # buraco: o retorno do dia seguinte cobre 2 pregões -> inválido
    assert flag("CCC", date(2020, 6, 18)) == "gap"
    # linha inválida descartada; colunas lidas pelo cabeçalho nas duas ordens
    assert con.execute("SELECT count(*) FROM p WHERE close <= 0 OR dayofweek(date) IN (0, 6)").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM p WHERE volume NOT IN (1000, 10, 5, 0)").fetchone()[0] == 0
    # parado
    assert flag("STL", date(2020, 7, 22)) == "stale"
    # calendário: 2020-07-03 (feriado) não é pregão
    assert not con.execute("SELECT count(*) FROM c WHERE date = DATE '2020-07-03'").fetchone()[0]
    # fim de coleta x série longa
    tr = dict(con.execute("SELECT ticker, truncated FROM q").fetchall())
    assert tr["OLD1"] and tr["OLD2"] and tr["OLD3"] and not tr["AAA"], tr
    # primeiro dia de cada série
    assert con.execute("SELECT count(*) FROM p WHERE flag = 'first'").fetchone()[0] == len(files)
    # views
    db = duckdb.connect(str(cfg.db), read_only=True)
    assert db.execute("SELECT count(*) FROM news.returns WHERE ticker = 'AAA'").fetchone()[0] == len(days) - 3  # first + 2 emendas

    shutil.rmtree(root, ignore_errors=True)
    print("[test_prices] ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
