"""Grafo dinâmico: o estado das relações numa data, a partir de news.graph_events.

Nada é materializado por data: um snapshot agrega os eventos de uma janela
(e, opcionalmente, com decaimento exponencial) num instante. Construa os eventos
antes com `python -m newsmanager.extract graph-build`.

    from newsmanager import config, query, graph
    con = query.connect(config.load())
    edges = graph.snapshot(con, "2020-03-16", window_days=90, half_life_days=30)
    for asof, e in graph.snapshots(con, "2020-01-01", "2020-12-31", freq="1 week"):
        ...

Sem look-ahead: `published_at` não tem hora (é sempre meia-noite), então uma
notícia do dia d pode ter saído depois do fechamento. O snapshot em `asof` usa
só `dt < asof` -- serve para prever o retorno do próprio dia `asof`.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterator

import duckdb

DateLike = str | date | datetime


def _where(asof: DateLike, window_days: int | None, classes, kinds, tickers_only: bool,
           min_strength: float) -> tuple[str, list]:
    w, p = ["e.dt < CAST(? AS DATE)"], [str(asof)]
    if window_days:
        w.append("e.dt >= CAST(? AS DATE) - CAST(? AS INTEGER)")
        p += [str(asof), int(window_days)]
    if classes:
        w.append(f"e.rel_class IN ({', '.join('?' for _ in classes)})")
        p += list(classes)
    if min_strength:
        w.append("e.strength >= ?")
        p.append(float(min_strength))
    if kinds:
        ks = ", ".join("?" for _ in kinds)
        w.append(f"na.kind IN ({ks}) AND nb.kind IN ({ks})")
        p += list(kinds) * 2
    if tickers_only:
        w.append("na.ticker IS NOT NULL AND nb.ticker IS NOT NULL")
    return " AND ".join(w), p


def snapshot_sql(asof: DateLike, window_days: int | None = 90, half_life_days: float | None = None,
                 by_class: bool = False, classes: tuple[str, ...] = (), kinds: tuple[str, ...] = (),
                 tickers_only: bool = False, min_events: int = 1, min_strength: float = 0.0,
                 directed: bool = True) -> tuple[str, list]:
    """SQL (e parâmetros) das arestas do grafo em `asof`.

    Uma linha por (src, dst[, rel_class]):
      n_events, n_docs, n_pos, n_neg, w_sum (soma de sinal x força), w_mean,
      w_decay (w ponderado por 0.5 ** (idade_dias / half_life_days); = w_sum se
      half_life_days for None), first_dt, last_dt.
    `directed=False` soma src->dst e dst->src numa aresta (src < dst).
    """
    where, params = _where(asof, window_days, classes, kinds, tickers_only, min_strength)
    decay = (f"power(0.5, (CAST(? AS DATE) - e.dt) / {float(half_life_days)})"
             if half_life_days else "1.0")
    dparams = [str(asof)] if half_life_days else []
    a, b = ("e.src", "e.dst") if directed else ("least(e.src, e.dst)", "greatest(e.src, e.dst)")
    cls = ", e.rel_class" if by_class else ""
    sql = f"""
        SELECT {a} AS src, {b} AS dst{cls},
               count(*) AS n_events, count(DISTINCT e.doc_id) AS n_docs,
               count(*) FILTER (WHERE e.sign > 0) AS n_pos,
               count(*) FILTER (WHERE e.sign < 0) AS n_neg,
               sum(e.w) AS w_sum, avg(e.w) AS w_mean,
               sum(e.w * {decay}) AS w_decay,
               min(e.dt) AS first_dt, max(e.dt) AS last_dt
        FROM news.graph_events e
        JOIN news.graph_nodes na ON na.node_id = e.src
        JOIN news.graph_nodes nb ON nb.node_id = e.dst
        WHERE {where}
        GROUP BY ALL
        HAVING count(*) >= {int(min_events)}
    """
    # a expressão de decaimento aparece antes do WHERE no texto do SQL
    return sql, dparams + params


def snapshot(con: duckdb.DuckDBPyConnection, asof: DateLike, **kw) -> duckdb.DuckDBPyRelation:
    """Arestas do grafo em `asof` como relação DuckDB (.fetchall(), .arrow(), .df())."""
    sql, params = snapshot_sql(asof, **kw)
    return con.sql(sql, params=params) if hasattr(con, "sql") else con.execute(sql, params)


def snapshots(con: duckdb.DuckDBPyConnection, start: DateLike, end: DateLike,
              freq: str = "1 week", **kw) -> Iterator[tuple[date, duckdb.DuckDBPyRelation]]:
    """(asof, arestas) para cada data de `start` a `end` a cada `freq` (intervalo DuckDB)."""
    dates = con.execute(
        "SELECT CAST(d AS DATE) FROM generate_series(CAST(? AS TIMESTAMP), CAST(? AS TIMESTAMP), "
        "CAST(? AS INTERVAL)) t(d)", [str(start), str(end), freq]).fetchall()
    for (d,) in dates:
        yield d, snapshot(con, d, **kw)


def to_networkx(con: duckdb.DuckDBPyConnection, edges: duckdb.DuckDBPyRelation, directed: bool = True):
    """Converte um snapshot num networkx (Multi)DiGraph com os atributos dos nós.

    networkx não é dependência do projeto: `pip install networkx` para usar.
    Com `by_class=True` o grafo é multigrafo (uma aresta por classe).
    """
    import networkx as nx

    cols = edges.columns
    multi = "rel_class" in cols
    g = (nx.MultiDiGraph() if directed else nx.MultiGraph()) if multi else (nx.DiGraph() if directed else nx.Graph())
    for row in edges.fetchall():
        r = dict(zip(cols, row))
        s, d = r.pop("src"), r.pop("dst")
        if multi:
            g.add_edge(s, d, key=r["rel_class"], **r)
        else:
            g.add_edge(s, d, **r)
    if g.number_of_nodes():
        ids = list(g.nodes)
        rows = con.execute(
            "SELECT node_id, label, kind, ticker, has_price FROM news.graph_nodes "
            "WHERE node_id IN (SELECT unnest(?))", [ids]).fetchall()
        for nid, label, kind, ticker, hp in rows:
            g.nodes[nid].update(label=label, kind=kind, ticker=ticker, has_price=hp)
    return g
