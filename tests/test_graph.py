"""Grafo dinâmico: regras de nó, classes de relação e snapshot sem look-ahead.

Sem rede, sem GPU e sem o corpus: eventos sintéticos num DuckDB em memória.

    python tests/test_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from newsmanager import graph  # noqa: E402
from newsmanager.extract.graph import compatible, rel_class  # noqa: E402


def main() -> int:
    # Canônico do modelo só vale se tiver algo em comum com o nome
    for key, can in [("fed", "Federal Reserve"), ("us federal reserve", "Federal Reserve"),
                     ("ecb", "European Central Bank"), ("eia", "U.S. Energy Information Administration"),
                     ("oil prices", "crude oil"), ("brent crude", "crude oil"), ("apple inc", "Apple Inc.")]:
        assert compatible(key, can), (key, can)
    for key, can in [("mexico central bank", "Federal Reserve"), ("nigerian banks", "Federal Reserve"),
                     ("moderna", "Merck & Company, Inc."), ("energy shares", "crude oil"),
                     ("dfs", "Federal Reserve")]:
        assert not compatible(key, can), (key, can)

    for t, c in [("raises price target on", "analyst"), ("raises rates on", "monetary"),
                 ("buys shares of", "ownership"), ("acquires", "ma"), ("competes with", "competition"),
                 ("sues", "legal_regulatory"), ("supplies", "supply_chain"), ("affects", "impact"),
                 ("imposes tariffs on", "trade_policy"), ("criticizes", "stance"), ("xyzzy", "other")]:
        assert rel_class(t) == c, (t, rel_class(t), c)

    con = duckdb.connect()
    con.execute("CREATE SCHEMA news")
    con.execute("""CREATE TABLE news.graph_nodes AS SELECT * FROM (VALUES
        ('FEDERAL_RESERVE', 'Federal Reserve', 'central_bank', NULL, false),
        ('JPM', 'JPMorgan', 'company', 'JPM', true),
        ('BAC', 'Bank of America', 'company', 'BAC', true)) t(node_id, label, kind, ticker, has_price)""")
    con.execute("""CREATE TABLE news.graph_events AS SELECT dt, doc_id, src, dst, rel_class, sign, strength::DOUBLE AS strength, w::DOUBLE AS w FROM (VALUES
        (DATE '2020-03-01', 'd1', 'FEDERAL_RESERVE', 'JPM', 'monetary', -1, 0.8, -0.8),
        (DATE '2020-03-10', 'd2', 'FEDERAL_RESERVE', 'JPM', 'monetary', 1, 0.5, 0.5),
        (DATE '2020-03-10', 'd2', 'JPM', 'BAC', 'competition', -1, 0.4, -0.4),
        (DATE '2020-03-16', 'd3', 'FEDERAL_RESERVE', 'JPM', 'monetary', -1, 1.0, -1.0)
        ) t(dt, doc_id, src, dst, rel_class, sign, strength, w)""")

    # asof exclusivo: a notícia do próprio dia 16 não entra
    e = {(r[0], r[1]): r for r in graph.snapshot(con, "2020-03-16", window_days=30).fetchall()}
    fed = e[("FEDERAL_RESERVE", "JPM")]
    assert fed[2] == 2 and abs(fed[6] - (-0.3)) < 1e-9, fed
    # janela: só o que tem até 7 dias antes de 16/03
    e = graph.snapshot(con, "2020-03-16", window_days=7).fetchall()
    assert {(r[0], r[1]) for r in e} == {("FEDERAL_RESERVE", "JPM"), ("JPM", "BAC")}
    # decaimento: meia-vida de 6 dias, evento de 6 dias atrás pesa 1/2
    (r,) = graph.snapshot(con, "2020-03-16", window_days=7, half_life_days=6, tickers_only=False,
                          classes=("monetary",)).fetchall()
    assert abs(r[8] - 0.25) < 1e-9, r
    # só ticker dos dois lados
    e = graph.snapshot(con, "2020-03-17", window_days=30, tickers_only=True).fetchall()
    assert [(r[0], r[1]) for r in e] == [("JPM", "BAC")]
    # não direcionado
    e = graph.snapshot(con, "2020-03-17", window_days=30, directed=False).fetchall()
    assert {(r[0], r[1]) for r in e} == {("FEDERAL_RESERVE", "JPM"), ("BAC", "JPM")}
    assert len(list(graph.snapshots(con, "2020-03-01", "2020-03-29", freq="1 week"))) == 5

    print("[test_graph] ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
