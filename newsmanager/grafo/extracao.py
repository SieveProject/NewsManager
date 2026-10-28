"""Etapas 1 e 2 do roteiro (seção 18.7): validação da extração e descritivas do grafo.

    python -m newsmanager.grafo auditoria       # presença dos agentes no texto (todas as tuplas)
    python -m newsmanager.grafo amostra         # planilhas de rotulagem (800 tuplas, 2 anotadores) e de recall
    python -m newsmanager.grafo metricas A.csv B.csv [--recall R.csv]
    python -m newsmanager.grafo anonimizacao    # 300 unidades mascaradas (só prepara; rodar exige GPU)
    python -m newsmanager.grafo descritivas

O texto vem de data/curated/extraction_units (cópia do Drive): é o que o LLM
recebeu (título + corpo; o corpo pode ter sido cortado em max_chars).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .. import query
from ..config import Config
from . import dados

UNITS = ("curated", "extraction_units", "*", "*.parquet")
GENERIC = ["the", "and", "inc", "corp", "corporation", "company", "companies", "group", "holdings", "shares",
           "stock", "stocks", "market", "markets", "bank", "banks", "global", "international", "national",
           "united", "states", "american", "sector", "industry", "index", "prices", "price", "economy"]
SEED_SQL = "hash(CAST(event_id AS VARCHAR) || 'secao18')"
PERIODS = (("2011-01-01", "2018-12-31"), ("2019-01-01", "2023-12-31"))
PER_CELL = 25
ANNOT_COLS = ["entidades_ok", "relacao_ok", "classe_correta", "sinal_ok", "forca_humana_0a3", "obs"]


def _units(cfg: Config) -> str:
    return str(cfg.root.joinpath(*UNITS))


def annot_dir(cfg: Config) -> Path:
    return cfg.root / "anotacao"


def _present(col: str) -> str:
    gen = "[" + ", ".join(f"'{g}'" for g in GENERIC) + "]"
    toks = (f"list_filter(regexp_split_to_array(lower({col}), '[^a-z0-9]+'), "
            f"x -> length(x) >= 4 AND NOT list_contains({gen}, x))")
    return (f"contains(u.txt, lower({col})) AS lit_{col[-1]}, "
            f"CASE WHEN len({toks}) = 0 THEN contains(u.txt, lower({col})) "
            f"ELSE list_bool_or(list_transform({toks}, x -> contains(u.txt, x))) END AS tok_{col[-1]}")


def auditoria(cfg: Config) -> dict:
    """Fração das tuplas do grafo com agent_a e agent_b no texto da unidade
    (literal; ou algum token distintivo de >= 4 letras), por classe e por ano."""
    con = query.connect(cfg)
    con.execute("SET memory_limit = '4GB'")
    rows = []
    for y in range(2009, 2025):
        r = con.execute(f"""
            WITH u AS (SELECT unit_id, lower(coalesce(title, '') || ' ' || coalesce(body, '')) AS txt
                       FROM read_parquet('{_units(cfg)}') WHERE year = {y}),
                 e AS (SELECT event_id, unit_id, rel_class, agent_a, agent_b FROM news.graph_events
                       WHERE year(dt) = {y})
            SELECT rel_class, count(*) AS n, count(txt) AS n_txt,
                   sum(CAST(lit_a AND lit_b AS INT)) AS lit, sum(CAST(tok_a AND tok_b AS INT)) AS tok
            FROM (SELECT e.rel_class, u.txt, {_present('e.agent_a')}, {_present('e.agent_b')}
                  FROM e LEFT JOIN u USING (unit_id))
            GROUP BY 1""").fetchall()
        rows += [(y, *x) for x in r]
    con.close()
    tot = np.array([[n, nt, l, t] for _, _, n, nt, l, t in rows], float).sum(axis=0)
    by_cls: dict = {}
    for _, c, n, nt, l, t in rows:
        a = by_cls.setdefault(c, np.zeros(4))
        a += [n, nt, l or 0, t or 0]
    by_year: dict = {}
    for y, _, n, nt, l, t in rows:
        a = by_year.setdefault(y, np.zeros(4))
        a += [n, nt, l or 0, t or 0]
    fmt = lambda a: {"tuplas": int(a[0]), "com_texto": int(a[1]), "literal": float(a[2] / max(a[1], 1)),  # noqa: E731
                     "token": float(a[3] / max(a[1], 1))}
    res = {"total": fmt(tot), "por_classe": {c: fmt(a) for c, a in sorted(by_cls.items())},
           "por_ano": {y: fmt(a) for y, a in sorted(by_year.items())}}
    out = cfg.root.parent / "logs" / "secao18_auditoria.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res["total"], ensure_ascii=False), f"\n[auditoria] {out}")
    return res


def amostra(cfg: Config) -> dict:
    """800 tuplas (25 por classe x período) para dois anotadores e 100 unidades
    para o recall. Sorteio determinístico (hash do event_id)."""
    import pandas as pd
    con = query.connect(cfg)
    con.execute("SET memory_limit = '4GB'")
    parts = []
    for lo, hi in PERIODS:
        parts.append(con.execute(f"""
            SELECT * FROM (
                SELECT event_id, unit_id, dt, rel_class, agent_a, agent_b, relation_type, sign, strength, src, dst,
                       row_number() OVER (PARTITION BY rel_class ORDER BY {SEED_SQL}) AS k
                FROM news.graph_events WHERE dt BETWEEN DATE '{lo}' AND DATE '{hi}')
            WHERE k <= {PER_CELL}""").df())
    s = pd.concat(parts)
    ids = ", ".join(f"'{u}'" for u in s["unit_id"].unique())
    txt = con.execute(f"""SELECT unit_id, title, body FROM read_parquet('{_units(cfg)}')
                          WHERE unit_id IN ({ids})""").df()
    s = s.merge(txt, on="unit_id", how="left").drop(columns="k")
    s["periodo"] = np.where(s["dt"].astype(str) <= PERIODS[0][1], "2011-2018", "2019-2023")
    s = s.sample(frac=1.0, random_state=7).reset_index(drop=True)
    s.insert(0, "id", np.arange(1, len(s) + 1))
    for c in ANNOT_COLS:
        s[c] = ""
    d = annot_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    for who in ("A", "B"):
        s.to_csv(d / f"secao18_tuplas_anotador_{who}.csv", index=False)
    # recall: 100 unidades (8 por ano de 2011 a 2022, 4 de 2023), todas as tuplas limpas da unidade
    rec = con.execute(f"""
        SELECT unit_id, y FROM (
            SELECT unit_id, year(min(dt)) AS y,
                   row_number() OVER (PARTITION BY year(min(dt)) ORDER BY hash(unit_id || 'secao18')) AS k
            FROM news.graph_events WHERE dt >= DATE '2011-01-01' GROUP BY unit_id)
        WHERE (y < 2023 AND k <= 8) OR (y = 2023 AND k <= 4)""").df()
    ids = ", ".join(f"'{u}'" for u in rec["unit_id"])
    tup = con.execute(f"""SELECT unit_id, agent_a, agent_b, relation_type, direction, strength
                          FROM news.relations_clean WHERE unit_id IN ({ids})""").df()
    txt = con.execute(f"""SELECT unit_id, published_at, title, body FROM read_parquet('{_units(cfg)}')
                          WHERE unit_id IN ({ids})""").df()
    con.close()
    tl = tup.groupby("unit_id").apply(lambda g: json.dumps(g.to_dict("records"),
                                                           ensure_ascii=False), include_groups=False)
    rec = txt.merge(tl.rename("tuplas_extraidas"), left_on="unit_id", right_index=True, how="left")
    rec["n_tuplas_corretas"] = ""
    rec["relacoes_faltantes"] = ""
    rec["n_faltantes"] = ""
    rec.to_csv(d / "secao18_recall.csv", index=False)
    (d / "LEIA-ME.txt").write_text(LEIA_ME)
    print(f"[amostra] {len(s)} tuplas e {len(rec)} unidades em {d}")
    return {"tuplas": len(s), "recall": len(rec)}


LEIA_ME = """Rotulagem da seção 18.7 (etapa 1 do roteiro grafo_noticias_covariancia.md)

secao18_tuplas_anotador_A.csv e _B.csv: as mesmas 800 tuplas (25 por classe x período), uma
planilha por anotador, preenchidas sem consulta entre os dois. Para cada linha, leia title + body:
  entidades_ok      1 se agent_a e agent_b estão corretos no texto; 0 se não
  relacao_ok        1 se a relação existe no texto (qualquer redação); 0 se não
  classe_correta    a classe certa entre as 16 de rel_class (deixe igual se estiver certa)
  sinal_ok          1 se o sinal (sign) está certo; 0 se não
  forca_humana_0a3  0 nenhuma, 1 fraca, 2 moderada, 3 forte
  obs               livre
secao18_recall.csv: 100 artigos. Leia o artigo inteiro e, comparando com tuplas_extraidas:
  n_tuplas_corretas   quantas das extraídas estão certas
  relacoes_faltantes  relações econômicas entre agentes que o artigo afirma e não foram extraídas
  n_faltantes         quantas
Depois: python -m newsmanager.grafo metricas data/anotacao/secao18_tuplas_anotador_A.csv \\
        data/anotacao/secao18_tuplas_anotador_B.csv --recall data/anotacao/secao18_recall.csv
"""


def _kappa(a: np.ndarray, b: np.ndarray) -> float:
    cats = np.union1d(a, b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def metricas(paths: list[str], recall: str | None = None) -> dict:
    """Métricas da tabela 2.2 do roteiro a partir das planilhas preenchidas."""
    import pandas as pd
    from scipy.stats import spearmanr
    frames = [pd.read_csv(p) for p in paths]
    res = {}
    for name, f in zip(paths, frames):
        f = f[f["relacao_ok"].astype(str).str.strip() != ""]
        num = lambda c: pd.to_numeric(f[c], errors="coerce")  # noqa: E731
        conf = pd.crosstab(f["rel_class"], f["classe_correta"].fillna(f["rel_class"]))
        res[Path(name).name] = {
            "n": int(len(f)), "precisao_entidade": float(num("entidades_ok").mean()),
            "precisao_relacao": float(num("relacao_ok").mean()),
            "acuracia_classe": float((f["classe_correta"].fillna(f["rel_class"]) == f["rel_class"]).mean()),
            "acuracia_sinal": float(num("sinal_ok").mean()),
            "spearman_forca": float(spearmanr(f["strength"], num("forca_humana_0a3"), nan_policy="omit").statistic),
            "confusao_classe": conf.to_dict()}
    if len(frames) == 2:
        a, b = frames
        m = a.merge(b, on="id", suffixes=("_a", "_b"))
        res["kappa"] = {c: _kappa(m[f"{c}_a"].astype(str).to_numpy(), m[f"{c}_b"].astype(str).to_numpy())
                        for c in ("entidades_ok", "relacao_ok", "sinal_ok", "classe_correta")}
    if recall:
        r = pd.read_csv(recall)
        ok = pd.to_numeric(r["n_tuplas_corretas"], errors="coerce")
        miss = pd.to_numeric(r["n_faltantes"], errors="coerce")
        res["recall"] = float(ok.sum() / (ok.sum() + miss.sum()))
    print(json.dumps({k: (v if not isinstance(v, dict) or "confusao_classe" not in v else
                          {x: y for x, y in v.items() if x != "confusao_classe"}) for k, v in res.items()},
                     indent=2, ensure_ascii=False))
    return res


def anonimizacao(cfg: Config, n: int = 300) -> Path:
    """Prepara 300 unidades com nomes de empresas e datas mascarados para
    reextrair com o mesmo prompt (teste 7.1.3). Não roda o LLM."""
    con = query.connect(cfg)
    u = con.execute(f"""
        WITH s AS (SELECT unit_id FROM (SELECT unit_id, row_number() OVER (ORDER BY hash(unit_id || 'anon')) AS k
                                        FROM (SELECT DISTINCT unit_id FROM news.graph_events
                                              WHERE dt >= DATE '2011-01-01')) WHERE k <= {n})
        SELECT x.unit_id, x.title, x.body FROM read_parquet('{_units(cfg)}') x JOIN s USING (unit_id)""").fetchall()
    names = con.execute("""SELECT unit_id, agent_a, kind_a, agent_b, kind_b FROM news.relations_linked
                           WHERE unit_id IN (SELECT unnest(?))""", [[x[0] for x in u]]).fetchall()
    con.close()
    comp: dict[str, set] = {}
    for uid, a, ka, b, kb in names:
        for nm, k in ((a, ka), (b, kb)):
            if k == "company" and nm and len(nm) >= 3:
                comp.setdefault(uid, set()).add(nm)
    date_rx = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{4})?|\b(19|20)\d{2}\b")
    out = annot_dir(cfg) / "secao18_anonimizacao.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for uid, title, body in u:
            txt = f"{title or ''}\n{body or ''}"
            mapping = {}
            for k, nm in enumerate(sorted(comp.get(uid, []), key=len, reverse=True)):
                tag = f"[EMPRESA_{chr(65 + k % 26)}{k // 26 or ''}]"
                mapping[tag] = nm
                txt = re.sub(re.escape(nm), tag, txt, flags=re.I)
            txt = date_rx.sub("[DATA]", txt)
            f.write(json.dumps({"unit_id": uid, "text": txt, "mascaras": mapping}, ensure_ascii=False) + "\n")
    print(f"[anonimizacao] {len(u)} unidades em {out} (reextração exige GPU: só com autorização)")
    return out


# ------------------------------------------------------------------ etapa 2

def descritivas(cfg: Config) -> dict:
    import pandas as pd
    con = query.connect(cfg)
    con.execute("SET memory_limit = '4GB'")
    by_year = con.execute("""
        WITH e AS (SELECT year(dt) AS y, least(src, dst) AS a, greatest(src, dst) AS b, sign, rel_class
                   FROM news.graph_events WHERE src <> dst)
        SELECT y, count(*) AS eventos, count(DISTINCT (a, b)) AS arestas,
               (SELECT count(*) FROM (SELECT a FROM e e2 WHERE e2.y = e.y UNION SELECT b FROM e e3 WHERE e3.y = e.y)) AS nos,
               avg(CAST(sign < 0 AS DOUBLE)) AS frac_neg
        FROM e GROUP BY y ORDER BY y""").df()
    neg = con.execute("""SELECT year(dt) AS y, rel_class, avg(CAST(sign < 0 AS DOUBLE)) AS f
                         FROM news.graph_events GROUP BY 1, 2""").df()
    deg = con.execute("""
        WITH e AS (SELECT DISTINCT year(dt) AS y, src AS a, dst AS b FROM news.graph_events WHERE src <> dst
                   UNION SELECT DISTINCT year(dt), dst, src FROM news.graph_events WHERE src <> dst),
             d AS (SELECT y, a, count(*) AS k FROM e GROUP BY 1, 2)
        SELECT y, median(k) AS mediana, quantile_cont(k, 0.9) AS p90, max(k) AS maximo, avg(k) AS media
        FROM d GROUP BY y ORDER BY y""").df()
    edges = con.execute("""SELECT DISTINCT year(dt) AS y, least(src, dst) AS a, greatest(src, dst) AS b
                           FROM news.graph_events WHERE src <> dst""").df()
    nodes = dict(con.execute("SELECT node_id, ticker IS NOT NULL FROM news.graph_nodes").fetchall())
    pn = dados.Panel(cfg, con)
    con.close()
    jac, jac_tk = {}, {}
    sets = {y: set(zip(g["a"], g["b"])) for y, g in edges.groupby("y")}
    for y in sorted(sets):
        if y - 1 in sets:
            a, b = sets[y - 1], sets[y]
            jac[int(y)] = len(a & b) / max(len(a | b), 1)
            ta = {e for e in a if nodes.get(e[0]) and nodes.get(e[1])}
            tb = {e for e in b if nodes.get(e[0]) and nodes.get(e[1])}
            jac_tk[int(y)] = len(ta & tb) / max(len(ta | tb), 1)
    # cobertura por quintil de tamanho e fração intra-setor (ações do S&P no início de cada ano)
    from .rede import Events
    con = query.connect(cfg)
    ev = Events(con)
    con.close()
    cov, intra = {}, {}
    for y in range(2011, 2024):
        i = next(k for k, d in enumerate(pn.days) if d.year == y)
        U = pn.universe(i)
        size = pn.size(i, U)
        snap = ev.snapshot(pn.days[min(i + 251, len(pn.days) - 1)], U, tau=1e9, window_days=365)
        m = snap.mentions()
        q = pd.qcut(pd.Series(size).rank(method="first"), 5, labels=False).to_numpy()
        cov[y] = {int(k): {"frac_com_evento": float((m[q == k] > 0).mean()), "eventos_medios": float(m[q == k].mean())}
                  for k in range(5)}
        W = snap.direct("n")
        iu = np.triu_indices(len(U), 1)
        e = W[iu] > 0
        sec = pn.sectors.arrays(U)
        intra[y] = {}
        for lvl in ("ff12", "ff49"):
            s = sec[lvl]
            same = ((s[:, None] == s[None, :]) & (s[:, None] >= 0))[iu]
            intra[y][lvl] = {"arestas": float(same[e].mean()) if e.any() else None, "pares": float(same.mean())}
    # portão: R² de g em setor (painel diádico principal)
    r2 = []
    d = dados.out_dir(cfg) / "diadico" / "sp500_tau=90_norm=idf"
    import pyarrow.parquet as pq
    for f in sorted(d.glob("pares_*.parquet")):
        t = pq.read_table(f, columns=["g", "same_ff12", "same_ff49", "same_sic2"]).to_pandas().dropna()
        X = np.column_stack([np.ones(len(t)), t["same_ff12"], t["same_ff49"], t["same_sic2"]])
        b, *_ = np.linalg.lstsq(X, t["g"].to_numpy(), rcond=None)
        res_ = t["g"].to_numpy() - X @ b
        r2.append(1 - res_.var() / t["g"].var())
    res = {"por_ano": by_year.to_dict("records"), "grau": deg.to_dict("records"),
           "frac_neg_por_classe": {c: g.set_index("y")["f"].to_dict() for c, g in neg.groupby("rel_class")},
           "jaccard": jac, "jaccard_ticker_ticker": jac_tk, "cobertura_tamanho": cov, "intra_setor": intra,
           "portao_r2_g_em_setor": {"media": float(np.mean(r2)) if r2 else None, "meses": len(r2),
                                    "passa(<0.5)": bool(np.mean(r2) < 0.5) if r2 else None}}
    out = cfg.root.parent / "logs" / "secao18_descritivas.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print(f"[descritivas] {out}; portão R² = {res['portao_r2_g_em_setor']}")
    return res
