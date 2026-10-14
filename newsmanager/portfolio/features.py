"""Atributos de par (i, j) do universo numa data t, só com notícias < t.

Ver docs/metodo_covariancia.md, seção 4.1. Tudo devolvido como matrizes p x p
simétricas, na ordem de `tickers`, com diagonal 0.

  direct_<grupo>  soma de força x decaimento dos eventos entre i e j (os dois
                  sentidos) cuja classe cai no grupo
  exposure        cosseno entre os vetores de exposição de i e j às entidades
                  FORA do universo (Fed, petróleo, China, empresas menores...),
                  com peso IDF: "S&P 500", que liga todo mundo, pesa pouco
  cooccur         artigos (da amostra extraída) em que i e j aparecem em algum
                  evento, com qualquer relação entre eles ou nenhuma: o controle
                  que separa "o LLM achou a relação" de "saíram no mesmo texto"
  same_ff49/sic2  mesmo setor (Fama-French 49 / SIC de 2 dígitos): controle de
                  Elton e Gruber, não vem das notícias
  comention       artigos do FNSPID marcados com os dois tickers (a variável de
                  Tang, Zhou e Hong 2019). A marcação é irregular: em 2018 XOM
                  tem 101 artigos marcados, CVX 839 e AAPL nenhum
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import numpy as np

from ..config import Config

# Classes de relação (newsmanager/extract/graph.py) -> grupos com efeito
# econômico parecido sobre a co-movimentação.
GROUPS = {
    "chain": ("supply_chain", "partnership"),
    "rivalry": ("competition",),
    "control": ("ma", "ownership"),
    "impact": ("impact",),
}
OTHER = "other_rel"
FEATURES = tuple(f"direct_{g}" for g in (*GROUPS, OTHER)) + ("exposure", "cooccur", "comention",
                                                                "same_ff49", "same_sic2")
# Setor não é notícia: entra cru (0/1), não em log.
BINARY = ("same_ff49", "same_sic2")


def _group_case() -> str:
    whens = " ".join(f"WHEN rel_class IN ({', '.join(repr(c) for c in cs)}) THEN '{g}'"
                     for g, cs in GROUPS.items())
    return f"CASE {whens} ELSE '{OTHER}' END"


def ticker_aliases(con: duckdb.DuckDBPyConnection, tickers: list[str]) -> dict[str, str]:
    """Símbolo do FNSPID -> nó do universo (GOOGL -> GOOG, META -> FB)."""
    out = {t: t for t in tickers}
    for node, ts in con.execute("SELECT node_id, tickers FROM news.graph_nodes WHERE node_id IN (SELECT unnest(?))",
                                [tickers]).fetchall():
        for t in ts or []:
            out.setdefault(t, node)
    return out


def pair_features(con: duckdb.DuckDBPyConnection, cfg: Config, asof: date, tickers: list[str],
                  news_days: int = 365, half_life_days: float = 90.0) -> dict[str, np.ndarray]:
    p = len(tickers)
    idx = {t: k for k, t in enumerate(tickers)}
    lo = asof - timedelta(days=news_days)
    decay = f"power(0.5, (DATE '{asof}' - dt) / {float(half_life_days)})"
    win = f"dt >= DATE '{lo}' AND dt < DATE '{asof}'"
    out = {f: np.zeros((p, p)) for f in FEATURES}

    # 1. vínculo direto entre dois membros do universo
    for a, b, g, w in con.execute(f"""
            SELECT src, dst, {_group_case()} AS g, sum(strength * {decay})
            FROM news.graph_events
            WHERE {win} AND src IN (SELECT unnest(?)) AND dst IN (SELECT unnest(?)) AND src <> dst
            GROUP BY ALL""", [tickers, tickers]).fetchall():
        m = out[f"direct_{g}"]
        m[idx[a], idx[b]] += w
        m[idx[b], idx[a]] += w

    # 2. exposição a entidades fora do universo, ponderada por IDF. df conta
    #    quantos nós-ticker (de todo o grafo) se ligaram à entidade na janela.
    rows = con.execute(f"""
        WITH e AS (SELECT src AS a, dst AS b, strength * {decay} AS w FROM news.graph_events WHERE {win}
                   UNION ALL
                   SELECT dst, src, strength * {decay} FROM news.graph_events WHERE {win}),
             tk AS (SELECT node_id FROM news.graph_nodes WHERE ticker IS NOT NULL),
             df AS (SELECT b AS ent, count(DISTINCT a) AS df FROM e WHERE a IN (SELECT node_id FROM tk) GROUP BY 1),
             n AS (SELECT count(DISTINCT a) AS n FROM e WHERE a IN (SELECT node_id FROM tk))
        SELECT e.a, e.b, sum(e.w) * ln(any_value(n.n) / any_value(df.df)) AS x
        FROM e JOIN df ON df.ent = e.b CROSS JOIN n
        WHERE e.a IN (SELECT unnest(?)) AND e.b NOT IN (SELECT unnest(?))
        GROUP BY 1, 2""", [tickers, tickers]).fetchall()
    ents = {}
    for _, b, _ in rows:
        ents.setdefault(b, len(ents))
    if ents:
        B = np.zeros((p, len(ents)))
        for a, b, x in rows:
            B[idx[a], ents[b]] = max(x, 0.0)
        nrm = np.linalg.norm(B, axis=1, keepdims=True)
        B = np.divide(B, nrm, out=np.zeros_like(B), where=nrm > 0)
        out["exposure"] = B @ B.T

    # 3. co-ocorrência no mesmo artigo extraído (qualquer evento limpo)
    for a, b, n in con.execute(f"""
            WITH m AS (SELECT DISTINCT doc_id, node FROM (
                           SELECT doc_id, src AS node FROM news.graph_events WHERE {win}
                           UNION ALL SELECT doc_id, dst FROM news.graph_events WHERE {win})
                       WHERE node IN (SELECT unnest(?)))
            SELECT x.node, y.node, count(*) FROM m x JOIN m y ON x.doc_id = y.doc_id AND x.node < y.node
            GROUP BY 1, 2""", [tickers]).fetchall():
        out["cooccur"][idx[a], idx[b]] = out["cooccur"][idx[b], idx[a]] = n

    # 4. co-menção nas marcações de ticker do FNSPID (todos os artigos)
    alias = ticker_aliases(con, tickers)
    ments = cfg.curated / "mentions" / "*.parquet"
    for a, b, n in con.execute(f"""
            WITH m AS (SELECT DISTINCT doc_id, a.node AS node
                       FROM read_parquet('{ments}') r
                       JOIN (SELECT unnest(?) AS sym, unnest(?) AS node) a ON a.sym = r.symbol
                       WHERE r.published_at >= TIMESTAMP '{lo}' AND r.published_at < TIMESTAMP '{asof}')
            SELECT x.node, y.node, count(*) FROM m x JOIN m y ON x.doc_id = y.doc_id AND x.node < y.node
            GROUP BY 1, 2""", [list(alias), list(alias.values())]).fetchall():
        out["comention"][idx[a], idx[b]] = out["comention"][idx[b], idx[a]] = n

    # 5. mesmo setor (SIC atual da SEC; ver newsmanager/sectors.py). Sem setor = 0.
    sec = dict((t, (f49, s2)) for t, f49, s2 in con.execute(
        "SELECT ticker, ff49, sic2 FROM news.sectors WHERE ticker IN (SELECT unnest(?))", [tickers]).fetchall())
    f49 = np.array([sec.get(t, (None, None))[0] or -1 for t in tickers])
    s2 = np.array([sec.get(t, (None, None))[1] or -1 for t in tickers])
    out["same_ff49"] = ((f49[:, None] == f49[None, :]) & (f49[:, None] >= 0)).astype(float)
    out["same_sic2"] = ((s2[:, None] == s2[None, :]) & (s2[:, None] >= 0)).astype(float)

    for m in out.values():
        np.fill_diagonal(m, 0.0)
    return out
