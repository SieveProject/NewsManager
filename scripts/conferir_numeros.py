"""Recalcula, a partir dos dados locais, os números citados na monografia e no README.

Uso: .venv/bin/python scripts/conferir_numeros.py > logs/conferencia_numeros.txt

Só lê dados (data/corpus.duckdb, data/curated, logs/*.json). Números que só existem
nos logs da VM (corpus completo, contagem de unidades) estão em logs do Drive
(gdrive:NewsManager/logs/{corpus,units,collect,link-prep}.log) e são citados no fim.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
con = duckdb.connect(str(ROOT / "data/corpus.duckdb"), read_only=True)
con.execute("SET memory_limit='2GB'; SET threads=4")


def q(sql: str):
    return con.execute(sql).fetchall()


def one(sql: str):
    return q(sql)[0]


def sec(t: str) -> None:
    print(f"\n== {t}")


# ---------------------------------------------------------------- corpus (local)
sec("marcações de ticker (data/curated/mentions)")
M = f"read_parquet('{ROOT}/data/curated/mentions/**/*.parquet', hive_partitioning=true)"
n, docs, syms = one(f"SELECT count(*), count(DISTINCT doc_id), count(DISTINCT symbol) FROM {M}")
print(f"marcações={n:,}  artigos com ticker={docs:,}  tickers={syms:,}")
for y, d, m in q(f"SELECT year(published_at), count(DISTINCT doc_id), count(*) FROM {M} "
                 "WHERE year(published_at) BETWEEN 2009 AND 2023 GROUP BY 1 ORDER BY 1"):
    print(f"  {y}: artigos={d:,} marcações={m:,}")

sec("tamanho dos artigos nas unidades extraídas localmente (amostra de 25%)")
U = f"read_parquet('{ROOT}/data/curated/extraction_units/**/*.parquet', hive_partitioning=true)"
nu, med, kept, trunc = one(f"""SELECT count(*), median(body_chars),
    sum(least(body_chars, 8000))::DOUBLE / sum(body_chars), avg((body_chars > 8000)::INT) FROM {U}""")
print(f"unidades={nu:,}  mediana={med:,.0f} caracteres  texto preservado com 8.000={kept:.1%}  "
      f"unidades truncadas={trunc:.1%}")

# ---------------------------------------------------------------- extração
sec("extração (news.relations_flagged)")
tot, units = one("SELECT count(*), count(DISTINCT unit_id) FROM news.relations_flagged")
print(f"tuplas={tot:,}  unidades com ao menos uma tupla={units:,}  "
      f"({units / 572_762:.1%} de 572.762)  média={tot / units:.2f} tuplas por unidade com relação")
for motivo, c, p in q("SELECT coalesce(noise, 'limpa'), count(*), 100.0 * count(*) / sum(count(*)) OVER () "
                      "FROM news.relations_flagged GROUP BY 1 ORDER BY 2 DESC"):
    print(f"  {motivo:12s} {c:>10,} {p:5.1f}%")
print("tipos de relação distintos:", one("SELECT count(DISTINCT lower(relation_type)) FROM news.relations_flagged")[0])
for d, c, s in q("SELECT direction, count(*), avg(strength) FROM news.relations_flagged WHERE noise IS NULL GROUP BY 1"):
    print(f"  direção {d}: {c:,} (força média {s:.3f})")

sec("ligação a tickers (news.relations_linked, só limpas)")
cl, men = one("SELECT count(*), count(ticker_a) + count(ticker_b) FROM news.relations_linked WHERE noise IS NULL")
um, ambos = one("""SELECT avg(((ticker_a IS NOT NULL) OR (ticker_b IS NOT NULL))::INT),
                          avg(((ticker_a IS NOT NULL) AND (ticker_b IS NOT NULL))::INT)
                   FROM news.relations_linked WHERE noise IS NULL""")
print(f"menções com ticker={men:,} de {2 * cl:,} ({men / (2 * cl):.1%})  tuplas com ticker em um lado={um:.1%}  nos dois={ambos:.1%}")

# ---------------------------------------------------------------- grafo
sec("grafo (news.graph_*)")
ev, pares = one("SELECT count(*), count(DISTINCT (src, dst)) FROM news.graph_events")
nos, com_ticker, com_preco = one("SELECT count(*), count(ticker), sum(has_price::INT) FROM news.graph_nodes")
print(f"eventos limpos={ev:,}  pares distintos={pares:,}  nós={nos:,}  com ticker={com_ticker:,}  com preço={com_preco:,}")
print("eventos em news.graph_events_all:", f"{one('SELECT count(*) FROM news.graph_events_all')[0]:,}")
for r in q("SELECT label, kind, n_events, n_keys, n_mentions FROM news.graph_nodes ORDER BY n_events DESC LIMIT 15"):
    print("  ", r)
for c, p, s in q("SELECT rel_class, 100.0 * count(*) / sum(count(*)) OVER (), avg(sign) FROM news.graph_events GROUP BY 1 ORDER BY 2 DESC"):
    print(f"  {c:18s} {p:5.1f}%  sinal médio {s:+.2f}")

sec("arestas diretas x entidades em comum, 2018, 100 tickers com mais eventos")
print(one("""
WITH ev AS (SELECT e.src, e.dst, ns.ticker AS ts, nd.ticker AS td
            FROM news.graph_events e JOIN news.graph_nodes ns ON ns.node_id = e.src
            JOIN news.graph_nodes nd ON nd.node_id = e.dst WHERE year(e.dt) = 2018),
lado AS (SELECT src AS n, ts AS t, dst AS o, td AS ot FROM ev UNION ALL SELECT dst, td, src, ts FROM ev),
top AS (SELECT n FROM lado WHERE t IS NOT NULL GROUP BY n ORDER BY count(*) DESC LIMIT 100),
pares AS (SELECT a.n AS i, b.n AS j FROM top a JOIN top b ON a.n < b.n),
direta AS (SELECT DISTINCT least(src, dst) AS i, greatest(src, dst) AS j FROM ev),
expo AS (SELECT DISTINCT n, o FROM lado WHERE n IN (SELECT n FROM top) AND ot IS NULL),
comum AS (SELECT DISTINCT a.n AS i, b.n AS j FROM expo a JOIN expo b ON a.o = b.o AND a.n < b.n)
SELECT count(*) AS pares,
       avg(((p.i, p.j) IN (SELECT (i, j) FROM direta))::INT) AS frac_direta,
       avg(((p.i, p.j) IN (SELECT (i, j) FROM comum))::INT) AS frac_comum
FROM pares p"""))

# ---------------------------------------------------------------- preços
sec("preços (news.prices, news.returns, news.price_quality)")
print(one("SELECT count(DISTINCT ticker), min(date), max(date), count(*) FROM news.prices"))
for f, c in q("SELECT coalesce(flag, 'válido'), count(*) FROM news.prices GROUP BY 1 ORDER BY 2 DESC"):
    print(f"  {f:13s} {c:>12,}")
print("retornos válidos:", one("SELECT count(*), count(DISTINCT ticker) FROM news.returns"))
print("truncados:", one("SELECT sum(truncated::INT) FROM news.price_quality")[0])
for t, r in q("""SELECT ticker, ret_raw FROM news.prices WHERE date = DATE '2020-07-06'
                 AND ticker IN ('AAPL','GOOG','TSLA','KO','XOM','PFE') ORDER BY 1"""):
    print(f"  2020-07-06 {t}: {r:+.1%}")
saltos = q("""WITH f AS (SELECT ticker, date, adj_close / close AS f,
                         lag(adj_close / close) OVER (PARTITION BY ticker ORDER BY date) AS f0
                  FROM news.prices WHERE close > 0 AND adj_close > 0)
              SELECT date, count(*) FROM f WHERE abs(f / f0 - 1) > 0.05 GROUP BY 1 ORDER BY 2 DESC LIMIT 6""")
print("datas com mais saltos de f > 5%:", [(str(d), c) for d, c in saltos])

# ---------------------------------------------------------------- universos e setores
sec("universo das mais citadas (news.universe)")
for n, datas, tk in q("SELECT n_target, count(DISTINCT rebalance_date), count(DISTINCT ticker) FROM news.universe GROUP BY 1 ORDER BY 1"):
    troca = one(f"""WITH u AS (SELECT rebalance_date d, ticker FROM news.universe WHERE n_target = {n}),
                    m AS (SELECT DISTINCT d FROM u), p AS (SELECT d, lag(d) OVER (ORDER BY d) d0 FROM m)
                    SELECT avg(1 - (SELECT count(*) FROM u a JOIN u b ON a.ticker = b.ticker
                                    WHERE a.d = p.d AND b.d = p.d0)::DOUBLE / {n})
                    FROM p WHERE d0 IS NOT NULL""")[0]
    print(f"  n={n}: {datas} datas, {tk} tickers distintos, troca média {troca:.1%}/mês")
cob = one("""SELECT avg((s.ff49 IS NOT NULL)::INT) FROM news.universe u
             LEFT JOIN news.sectors s USING (ticker) WHERE u.n_target = 100""")[0]
print(f"setores: {one('SELECT count(*), count(ff12) FROM news.sectors')}  cobertura ação-mês n=100: {cob:.1%}")

sec("universo S&P 500 point-in-time (ações por mês)")
import pandas as pd
comp = pd.read_csv(ROOT / "data/reference/sp500/sp500_historical_components.csv", parse_dates=["date"]).sort_values("date")
comp["ts"] = comp["tickers"].str.split(",")
todos = sorted({t for ts in comp["ts"] for t in ts})
lista = ", ".join(f"'{t}'" for t in todos)
cal = pd.Series([d for (d,) in q("SELECT date FROM news.trading_calendar WHERE date >= DATE '2009-06-01' ORDER BY 1")])
cal = pd.to_datetime(cal)
ret = con.execute(f"SELECT ticker, date FROM news.returns WHERE date >= DATE '2009-06-01' AND ticker IN ({lista})").df()
ret["date"] = pd.to_datetime(ret["date"])
ok = pd.crosstab(ret["date"], ret["ticker"]).reindex(cal, fill_value=0).clip(upper=1)
janela = ok.rolling(252).sum().shift(1)          # pregões válidos nos 252 anteriores a cada data
reb = cal[(cal >= "2011-01-01")].groupby([cal.dt.year, cal.dt.month]).min()
por = {}
for t in reb:
    membros = comp[comp["date"] < t].iloc[-1]["ts"]
    linha = janela.loc[t]
    por[t] = int(sum(1 for m in membros if m in linha.index and linha[m] >= 0.95 * 252))
por = pd.Series(por)
print(f"min={por.min()} max={por.max()} mediana antes de 2020-07={por[por.index < '2020-07-01'].median():.0f} "
      f"depois de 2020-08={por[por.index >= '2020-08-01'].median():.0f}")
print("por ano (mediana):", por.groupby(por.index.year).median().astype(int).to_dict())

# ---------------------------------------------------------------- logs da seção 18
sec("auditoria e anonimização (logs/secao18_*.json)")
print(json.dumps(json.load(open(ROOT / "logs/secao18_auditoria.json")), ensure_ascii=False)[:900])
a = json.load(open(ROOT / "logs/secao18_anonimizacao.json"))
print({k: a[k] for k in ("unidades", "tuplas_orig", "tuplas_anon", "frac_orig_reencontradas", "pareadas",
                         "direcao_igual", "tipo_igual", "forca_media_orig", "forca_media_anon", "forca_dif_media_abs")})
d = json.load(open(ROOT / "logs/secao18_descritivas.json"))
print("portão:", d["portao_r2_g_em_setor"])
print("FF12 intra-setor (arestas):", {y: round(v["ff12"]["arestas"], 3) for y, v in d["intra_setor"].items()})
print("Jaccard ticker-ticker:", {y: round(v, 3) for y, v in d["jaccard_ticker_ticker"].items() if "2011" <= y <= "2023"})
print("FOMC:", one(f"SELECT count(*) FROM read_csv_auto('{ROOT}/data/reference/fomc_dates.csv')")[0])

print("""
== logs da VM (gdrive:NewsManager/logs)
corpus.log : 15.549.299 linhas -> 12.746.534 artigos (18,0% de duplicação), 5.730.655 marcações, 8.552 tickers;
             63,1% das linhas sem ticker, 75,9% sem texto (arquivo inteiro)
units.log  : 12.746.534 artigos; -10.322.619 com menos de 200 caracteres ou vazios; -799.037 do lenta.ru;
             amostra de 40% = 643.433 unidades
collect.log: 572.762 unidades (0 falhas), 1.270.582 relações; filtro: 1.093.431 limpas (86,1%)
link-prep  : 378.662 nomes -> 304.430 chaves (2.280.852 menções), universo 7.728 tickers (7.693 com preço)
README     : 1.607.982 unidades elegíveis (100%); 401.609 = 25,0%; 572.762 = 35,6%""")
