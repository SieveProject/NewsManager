"""Universo de ativos por data de rebalanceamento -- etapa `universe`.

Em cada data t (primeiro pregão do mês), as n ações com mais eventos limpos no
grafo de notícias nos `news_days` anteriores a t, entre as que passam nos
filtros de preço na janela in-sample [t - insample_days pregões, t):

  - retorno válido em >= `min_ret_frac` dos pregões da janela;
  - nenhum pico de cotação (`spike`) na janela -- pega tickers reutilizados
    como o ACB 2006-2016, que intercala dois papéis;
  - preço >= `min_price` no último pregão antes de t e mediana de volume
    financeiro >= `min_dollar_volume`;
  - só ações (`stocks_only`): nó `company` que não é ETF na listagem;
  - série não truncada por FIM DE COLETA antes do fim do período de
    carteira (`hold_days` pregões a partir de t). Isso usa uma informação do
    futuro, mas sobre um artefato do arquivo (o lote antigo do FNSPID acaba em
    2020-07-02), não sobre retornos: sem o filtro, a carteira montada em
    2020-07-01 seguraria BAC, NFLX e FB sem preço nenhum depois do dia 2.
    Deslistagens reais (fins de série espalhados) NÃO são filtradas: o
    backtest tem de tratá-las.

Tudo o mais usa só dados < t. O nó do grafo é o ticker (classes de ação já
fundidas pelo graph-build: GOOGL entra como GOOG).
"""

from __future__ import annotations

import time
from datetime import date

import duckdb
import pyarrow as pa

from .config import Config, ConfigError

DEFAULTS = dict(n=100, insample_days=252, news_days=365, min_ret_frac=0.95,
                min_price=5.0, min_dollar_volume=5e6, hold_days=21, stocks_only=True)


def universe_dir(cfg: Config):
    return cfg.curated / "universe"


def select_sql(asof: str, n: int, insample_days: int, news_days: int, min_ret_frac: float,
               min_price: float, min_dollar_volume: float, hold_days: int, stocks_only: bool) -> str:
    # ETF (XLK, SPY...) é uma cesta das próprias ações: fora de um estudo de
    # covariância entre ações. Nó `company` e não-ETF na listagem da Nasdaq.
    stocks = ("AND g.kind = 'company' AND NOT coalesce(tk.is_etf, false)" if stocks_only else "")
    return f"""
    WITH cal AS (SELECT date, cal_i FROM news.trading_calendar),
         t0 AS (SELECT min(cal_i) AS i FROM cal WHERE date >= DATE '{asof}'),
         win AS (SELECT min(date) AS w0, max(date) AS w1 FROM cal, t0
                 WHERE cal_i >= t0.i - {insample_days} AND cal_i < t0.i),
         hold AS (SELECT max(date) AS h1 FROM cal, t0 WHERE cal_i >= t0.i AND cal_i < t0.i + {hold_days}),
         px AS (SELECT p.ticker,
                       count(p.ret) AS n_ret,
                       count(*) FILTER (WHERE p.flag IN ('spike', 'spike_revert')) AS n_spike,
                       arg_max(p.close, p.date) AS last_close,
                       max(p.date) AS last_date,
                       median(p.dollar_volume) AS median_dollar_volume
                FROM news.prices p, win WHERE p.date BETWEEN win.w0 AND win.w1 GROUP BY 1),
         ok AS (SELECT px.* FROM px
                LEFT JOIN news.price_quality q ON q.ticker = px.ticker
                CROSS JOIN win CROSS JOIN hold
                WHERE px.n_ret >= {min_ret_frac} * {insample_days}
                  AND px.n_spike = 0
                  AND px.last_date = win.w1
                  AND px.last_close >= {min_price}
                  AND px.median_dollar_volume >= {min_dollar_volume}
                  AND NOT (coalesce(q.truncated, false) AND q.last_date < hold.h1)),
         ev AS (SELECT node, count(*) AS n_events, count(DISTINCT doc_id) AS n_docs
                FROM (SELECT src AS node, doc_id FROM news.graph_events
                      WHERE dt >= DATE '{asof}' - {int(news_days)} AND dt < DATE '{asof}'
                      UNION ALL
                      SELECT dst, doc_id FROM news.graph_events
                      WHERE dt >= DATE '{asof}' - {int(news_days)} AND dt < DATE '{asof}')
                GROUP BY 1)
    SELECT DATE '{asof}' AS rebalance_date, ok.ticker, ev.n_events, ev.n_docs, ok.n_ret, ok.last_close,
           ok.median_dollar_volume,
           row_number() OVER (ORDER BY ev.n_events DESC, ok.median_dollar_volume DESC) AS rank
    FROM ok
    JOIN news.graph_nodes g ON g.node_id = ok.ticker AND g.ticker IS NOT NULL
    LEFT JOIN news.tickers tk ON tk.ticker = ok.ticker
    JOIN ev ON ev.node = ok.ticker
    WHERE TRUE {stocks}
    QUALIFY rank <= {int(n)}
    ORDER BY rank"""


def select(con: duckdb.DuckDBPyConnection, asof: str | date, **kw) -> list[tuple]:
    """O universo em `asof` (lista de linhas: asof, ticker, n_events, ..., rank)."""
    return con.execute(select_sql(str(asof), **{**DEFAULTS, **kw})).fetchall()


def rebalance_dates(con: duckdb.DuckDBPyConnection, start: str, end: str) -> list[date]:
    """Primeiro pregão de cada mês entre start e end."""
    return [d for (d,) in con.execute(
        """SELECT min(date) FROM news.trading_calendar WHERE date BETWEEN ? AND ?
           GROUP BY date_trunc('month', date) ORDER BY 1""", [start, end]).fetchall()]


def build(cfg: Config, start: str = "2011-01-01", end: str = "2023-12-31", views: bool = True, **kw) -> dict:
    """Universo de cada rebalanceamento mensal -> data/curated/universe/n=<n>.parquet."""
    if not cfg.db.exists():
        raise ConfigError(f"{cfg.db} não existe")
    params = {**DEFAULTS, **kw}
    out = universe_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"n={params['n']}.parquet"
    con = duckdb.connect(str(cfg.db), read_only=True)
    for v in ("graph_events", "graph_nodes", "prices", "price_quality", "trading_calendar"):
        if not con.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'news' "
                           "AND table_name = ?", [v]).fetchone()[0]:
            raise ConfigError(f"news.{v} não existe; rode graph-build e prices antes")
    t0 = time.time()
    rows, short = [], []
    dates = rebalance_dates(con, start, end)
    for d in dates:
        r = select(con, d, **kw)
        rows += r
        if len(r) < params["n"]:
            short.append((d, len(r)))
    con.close()
    cols = ["rebalance_date", "ticker", "n_events", "n_docs", "n_ret", "last_close", "median_dollar_volume", "rank"]
    t = pa.Table.from_pylist([dict(zip(cols, r)) for r in rows])
    t = t.append_column("n_target", pa.array([params["n"]] * len(t), pa.int32()))
    tmp = dst.with_suffix(".tmp")
    import pyarrow.parquet as pq
    pq.write_table(t, tmp, compression="zstd")
    tmp.rename(dst)
    if views:
        register_views(cfg)
    turnover = _turnover(rows)
    print(f"[universe] n={params['n']}: {len(dates)} rebalanceamentos ({dates[0]} a {dates[-1]}), "
          f"{len({r[1] for r in rows}):,} tickers distintos, troca média {turnover:.1%}/mês ({time.time() - t0:.0f}s)")
    if short:
        print(f"[universe] {len(short)} datas com menos de {params['n']} ações elegíveis: "
              + ", ".join(f"{d}={k}" for d, k in short[:12]) + (" ..." if len(short) > 12 else ""))
    print(f"[universe] {dst}")
    return {"dates": len(dates), "rows": len(rows), "short": short, "turnover": turnover}


def _turnover(rows: list[tuple]) -> float:
    by = {}
    for r in rows:
        by.setdefault(r[0], set()).add(r[1])
    ds = sorted(by)
    ch = [1 - len(by[a] & by[b]) / max(len(by[b]), 1) for a, b in zip(ds, ds[1:])]
    return sum(ch) / len(ch) if ch else 0.0


def register_views(cfg: Config) -> None:
    db = duckdb.connect(str(cfg.db))
    db.execute("CREATE SCHEMA IF NOT EXISTS news")
    db.execute(f"CREATE OR REPLACE VIEW news.universe AS "
               f"SELECT * FROM read_parquet('{universe_dir(cfg) / 'n=*.parquet'}')")
    db.close()
