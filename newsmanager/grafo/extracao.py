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
        # linha sem rótulo (NaN vira "nan" no astype(str)) fica fora
        f = f[pd.to_numeric(f["relacao_ok"], errors="coerce").notna()]
        num = lambda c: pd.to_numeric(f[c], errors="coerce")  # noqa: E731
        # classe e sinal só se julgam quando a relação existe no texto
        r = f[num("relacao_ok") == 1]
        cls = r["classe_correta"].fillna(r["rel_class"])
        conf = pd.crosstab(r["rel_class"], cls)
        res[Path(name).name] = {
            "n": int(len(f)), "precisao_entidade": float(num("entidades_ok").mean()),
            "precisao_relacao": float(num("relacao_ok").mean()),
            "acuracia_classe": float((cls == r["rel_class"]).mean()),
            "acuracia_sinal": float(pd.to_numeric(r["sinal_ok"], errors="coerce").mean()),
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


SUFFIX_RX = re.compile(r"[\s,]+(?:inc|incorporated|corp|corporation|co|company|ltd|limited|llc|plc|lp|l\.p|n\.v|s\.a|"
                       r"ag|se|sa|nv|holdings?|group|& co|the)\.?$", re.I)
DATE_RX = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{4})?"
                     r"|\b(?:19|20)\d{2}\b")


def _name_variants(nm: str) -> set[str]:
    """'Ansys Inc. (ANSS)' -> {'Ansys Inc.', 'Ansys'}: sem o ticker entre parênteses e sem sufixo societário."""
    base = re.sub(r"\s*\([^)]*\)", "", nm).strip()
    out = {base}
    cur = base
    while True:
        nxt = SUFFIX_RX.sub("", cur).strip(" ,.")
        if nxt == cur or len(nxt) < 3:
            break
        out.add(nxt)
        cur = nxt
    if cur.lower().startswith("the ") and len(cur) > 7:
        out.add(cur[4:])
    return {v for v in out if len(v) >= 3 and v.lower() not in GENERIC}


def mask_unit(title: str, body: str, companies: list[tuple[str, str | None]], symbols: list[str]) -> tuple[str, str, dict, list[str]]:
    """Mascara empresas (nomes, variantes sem sufixo e tickers) e datas. companies: (nome, ticker).
    Devolve título, corpo, máscara -> {nomes, ticker} e a lista de tickers do arquivo já mascarada."""
    groups: dict[str, dict] = {}
    for nm, tk in companies:
        m = re.search(r"\(([A-Z][A-Z.\-]{0,5})\)", nm or "")
        tk = tk or (m.group(1) if m else None)
        key = tk or SUFFIX_RX.sub("", re.sub(r"\s*\([^)]*\)", "", nm)).strip().lower()
        g = groups.setdefault(key, {"nomes": set(), "ticker": tk})
        g["nomes"] |= _name_variants(nm)
    for s in symbols or []:
        if s and s not in groups and not any(g["ticker"] == s for g in groups.values()):
            groups[s] = {"nomes": set(), "ticker": s}
    mapping, subs = {}, []
    for k, (key, g) in enumerate(sorted(groups.items())):
        tag = f"[EMPRESA_{chr(65 + k % 26)}{k // 26 or ''}]"
        mapping[tag] = {"nomes": sorted(g["nomes"]), "ticker": g["ticker"], "chave": key}
        subs += [(re.compile(rf"(?<![\w\[]){re.escape(v)}(?:'s)?(?!\w)", re.I), tag) for v in g["nomes"]]
        if g["ticker"]:
            # ticker só em caixa alta (evita 'ON', 'ALL' como palavra) e em formatos de cotação
            subs += [(re.compile(rf"(?:NYSE|NASDAQ|Nasdaq|AMEX)\s*:\s*{re.escape(g['ticker'])}\b"), tag),
                     (re.compile(rf"(?<![\w\[]){re.escape(g['ticker'])}(?![\w\]])"), tag)]
    subs.sort(key=lambda s: -len(s[0].pattern))                 # nomes longos antes dos curtos

    def apply(t: str) -> str:
        for rx, tag in subs:
            t = rx.sub(tag, t)
        return DATE_RX.sub("[DATA]", t)

    sym = [next((tg for tg, m in mapping.items() if m["ticker"] == s), "[EMPRESA]") for s in symbols or []]
    return apply(title or ""), apply(body or ""), mapping, sym


def anonimizacao(cfg: Config, n: int = 300) -> Path:
    """Prepara 300 unidades com nomes de empresas, tickers e datas mascarados para
    reextrair com o mesmo prompt (teste 7.1.3). Não roda o LLM (ver anon_run)."""
    con = query.connect(cfg)
    u = con.execute(f"""
        WITH s AS (SELECT unit_id FROM (SELECT unit_id, row_number() OVER (ORDER BY hash(unit_id || 'anon')) AS k
                                        FROM (SELECT DISTINCT unit_id FROM news.graph_events
                                              WHERE dt >= DATE '2011-01-01')) WHERE k <= {n})
        SELECT x.unit_id, x.title, x.body, x.all_symbols FROM read_parquet('{_units(cfg)}') x JOIN s USING (unit_id)
        ORDER BY x.unit_id""").fetchall()
    names = con.execute("""SELECT unit_id, agent_a, kind_a, ticker_a, agent_b, kind_b, ticker_b FROM news.relations_linked
                           WHERE unit_id IN (SELECT unnest(?))""", [[x[0] for x in u]]).fetchall()
    comp: dict[str, set] = {}
    for uid, a, ka, ta, b, kb, tb in names:
        for nm, k, tk in ((a, ka, ta), (b, kb, tb)):
            if (k == "company" or tk) and nm and len(nm) >= 2:
                comp.setdefault(uid, set()).add((nm, tk))
    # nome canônico de cada ticker da unidade (agentes e arquivo), para mascarar empresas
    # citadas no texto que não viraram agente ("Cadence Design Systems ( CDNS )")
    tks = sorted({tk for s in comp.values() for _, tk in s if tk} | {s for x in u for s in (x[3] or [])})
    canon: dict[str, set] = {}
    for tk, nm in con.execute("""SELECT DISTINCT ticker, canonical FROM news.entity_map
                                 WHERE link = 'issuer' AND ticker_check = 'name' AND ticker IN (SELECT unnest(?))""",
                              [tks]).fetchall():
        canon.setdefault(tk, set()).add(nm)
    con.close()
    for uid, *_, syms in u:
        tk_u = {tk for _, tk in comp.get(uid, set()) if tk} | set(syms or [])
        for tk in tk_u:
            comp.setdefault(uid, set()).update((nm, tk) for nm in canon.get(tk, ()))
    out = annot_dir(cfg) / "secao18_anonimizacao.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    resid = 0
    with out.open("w") as f:
        for uid, title, body, syms in u:
            t, b, mapping, sym = mask_unit(title, body, sorted(comp.get(uid, [])), list(syms or []))
            left = [v for m in mapping.values() for v in ([m["ticker"]] if m["ticker"] else []) + m["nomes"]
                    if re.search(rf"(?<!\w){re.escape(v)}(?!\w)", t + "\n" + b, re.I if v != m["ticker"] else 0)]
            resid += bool(left)
            f.write(json.dumps({"unit_id": uid, "title": t, "body": b, "symbols": sym, "mascaras": mapping,
                                "residuo": left}, ensure_ascii=False) + "\n")
    print(f"[anonimizacao] {len(u)} unidades em {out}; {resid} com nome/ticker residual "
          f"(reextração exige GPU: só com autorização)")
    return out


def anon_run(cfg: Config, host: str, model: str, concurrency: int = 8) -> Path:
    """Reextrai as unidades mascaradas com o mesmo prompt, schema, max_chars e modelo da
    extração (roda na GPU). Data -> '[DATA]'; tickers do arquivo -> máscaras."""
    import asyncio
    from ..extract.cli import _repo, _required_num_ctx
    from ..extract.client import OllamaClient, OllamaConfig
    from ..extract.prompt import DEFAULT_MAX_CHARS, load as load_prompt

    pr = load_prompt(_repo() / "prompts" / "extraction.txt", _repo() / "prompts" / "schema.json",
                     DEFAULT_MAX_CHARS, model=model)
    num_ctx = -(-_required_num_ctx(DEFAULT_MAX_CHARS, pr.template, 768) // 256) * 256
    oll = OllamaConfig(host=host, model=model, num_ctx=num_ctx, num_predict=768, concurrency=concurrency,
                       timeout_s=300.0)
    units = [json.loads(x) for x in (annot_dir(cfg) / "secao18_anonimizacao.jsonl").read_text().splitlines()]
    out = annot_dir(cfg) / "secao18_anon_saida.jsonl"

    async def go():
        async with OllamaClient(oll) as cl:
            h = await cl.health()
            if not h["has_model"]:
                raise RuntimeError(f"{model} não está em {host}: {h['models']}")
            await cl.warm()

            async def one(u):
                text, trunc = pr.render(article=u["body"], title=u["title"], date="[DATA]",
                                        symbols=", ".join(u["symbols"]) or None)
                r = await cl.generate(text, pr.schema)
                try:
                    tup = json.loads(r.text)["tuples"] if r.ok else None
                except (ValueError, KeyError, TypeError):
                    tup = None
                return {"unit_id": u["unit_id"], "ok": tup is not None, "tuples": tup, "truncated": trunc,
                        "prompt_version": pr.version, "model": model}
            return await asyncio.gather(*(one(u) for u in units))

    res = asyncio.run(go())
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res))
    print(f"[anon_run] {sum(r['ok'] for r in res)}/{len(res)} ok, prompt_version {pr.version}: {out}")
    return out


def anon_comparar(cfg: Config, h: int = 21) -> dict:
    """Compara a reextração mascarada com a original (7.1.3): pareamento por par de agentes
    (máscara -> chave da empresa), concordância de direção, tipo e força, e se as
    divergências se alinham com o retorno futuro do agente afetado (agent_b)."""
    from scipy import stats
    units = {d["unit_id"]: d for d in map(json.loads, (annot_dir(cfg) / "secao18_anonimizacao.jsonl").read_text().splitlines())}
    anon = {d["unit_id"]: d for d in map(json.loads, (annot_dir(cfg) / "secao18_anon_saida.jsonl").read_text().splitlines())}
    con = query.connect(cfg)
    orig = con.execute("""SELECT unit_id, published_at, agent_a, ticker_a, agent_b, ticker_b, relation_type, direction,
                                 strength, noise FROM news.relations_linked WHERE unit_id IN (SELECT unnest(?))""",
                       [list(anon)]).df()

    def key_orig(nm, tk, masks):
        for m in masks.values():
            if (tk and m["ticker"] == tk) or SUFFIX_RX.sub("", re.sub(r"\s*\([^)]*\)", "", nm or "")).strip().lower() == m["chave"] \
                    or re.sub(r"\s*\([^)]*\)", "", nm or "").strip() in m["nomes"]:
                return m["chave"]
        return (nm or "").strip().lower()

    def key_anon(nm, masks):
        m = re.search(r"\[EMPRESA_[A-Z]\d*\]", nm or "")
        if m and m.group(0) in masks:
            return masks[m.group(0)]["chave"]
        return (nm or "").strip().lower()

    rows = []
    n_orig = n_anon = 0
    for uid, a in anon.items():
        if not a["ok"]:
            continue
        masks = units[uid]["mascaras"]
        o = orig[orig.unit_id == uid]
        n_orig += len(o)
        n_anon += len(a["tuples"])
        at = {}
        for t in a["tuples"]:
            at.setdefault(frozenset((key_anon(t["agent_a"], masks), key_anon(t["agent_b"], masks))), []).append(t)
        for r in o.itertuples():
            ka, kb = key_orig(r.agent_a, r.ticker_a, masks), key_orig(r.agent_b, r.ticker_b, masks)
            cand = at.get(frozenset((ka, kb)), [])
            t = cand[0] if cand else None
            rows.append({"unit_id": uid, "dt": r.published_at, "ticker_b": r.ticker_b, "limpa": r.noise is None,
                         "par_achado": t is not None, "s_o": 1 if r.direction == "positive" else -1,
                         "f_o": r.strength, "tipo_o": (r.relation_type or "").lower(),
                         "s_a": None if t is None else (1 if t["direction"] == "positive" else -1),
                         "f_a": None if t is None else t["strength"],
                         "tipo_a": None if t is None else (t["relation_type"] or "").lower()})
    import pandas as pd
    df = pd.DataFrame(rows)
    m = df[df.par_achado].copy()
    # retorno futuro (h pregões a partir do dia seguinte à publicação) do agente afetado
    tk = sorted({x for x in m.ticker_b.dropna()})
    if tk:
        lst = ", ".join("'" + x.replace("'", "") + "'" for x in tk)
        ret = con.execute(f"SELECT ticker, date, ret FROM news.returns WHERE ticker IN ({lst})").df()
        ret["date"] = pd.to_datetime(ret["date"])
        fut = []
        for r in m.itertuples():
            s = ret[(ret.ticker == r.ticker_b) & (ret.date > pd.Timestamp(r.dt).normalize())].sort_values("date").head(h)
            fut.append(float(np.prod(1 + s.ret) - 1) if len(s) >= int(0.7 * h) else np.nan)
        m["r_fut"] = fut
    else:
        m["r_fut"] = np.nan
    con.close()
    m["d"] = m.s_o * m.f_o - m.s_a * m.f_a
    ok = m.dropna(subset=["r_fut"])

    def corr(x, y):
        if len(x) < 10 or np.std(x) == 0:
            return {"rho": None, "p": None, "n": int(len(x))}
        r, p = stats.spearmanr(x, y)
        return {"rho": float(r), "p": float(p), "n": int(len(x))}

    res = {
        "unidades": len(anon), "unidades_ok": int(sum(a["ok"] for a in anon.values())),
        "unidades_com_residuo": int(sum(bool(u["residuo"]) for u in units.values())),
        "tuplas_orig": n_orig, "tuplas_anon": n_anon,
        "frac_orig_reencontradas": float(df.par_achado.mean()) if len(df) else None,
        "frac_orig_limpas_reencontradas": float(df[df.limpa].par_achado.mean()) if len(df) else None,
        "pareadas": int(len(m)),
        "direcao_igual": float((m.s_o == m.s_a).mean()) if len(m) else None,
        "tipo_igual": float((m.tipo_o == m.tipo_a).mean()) if len(m) else None,
        "forca_media_orig": float(m.f_o.mean()) if len(m) else None,
        "forca_media_anon": float(m.f_a.mean()) if len(m) else None,
        "forca_dif_media_abs": float((m.f_o - m.f_a).abs().mean()) if len(m) else None,
        f"alinhamento_divergencia_retorno_{h}d": corr(ok.d, ok.r_fut),
        f"alinhamento_original_retorno_{h}d": corr(ok.s_o * ok.f_o, ok.r_fut),
        f"alinhamento_anon_retorno_{h}d": corr(ok.s_a * ok.f_a, ok.r_fut),
    }
    out = cfg.root.parent / "logs" / "secao18_anonimizacao.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    df.to_csv(annot_dir(cfg) / "secao18_anon_pareamento.csv", index=False)
    print(f"[anon_comparar] {json.dumps(res, ensure_ascii=False)}\n-> {out}")
    return res


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
