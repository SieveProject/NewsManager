"""Grafo de conhecimento dinâmico -- etapa `graph-build` (sem GPU).

Cada tupla limpa vira um EVENTO datado `src --classe(sinal, força)--> dst`
entre NÓS únicos: um nó por empresa (o ticker), um para o Fed, um para o
petróleo... O grafo "em t" é a agregação dos eventos de uma janela antes de t
(`newsmanager.graph.snapshot`); nada é materializado por data.

Por que uma camada de nós por cima do `entity_id` do link:

  - o `canonical` do modelo leve às vezes junta demais: "mexico central bank",
    "nigerian banks" e "dfs" saíram como "Federal Reserve". Sem ticker, o
    canonical só vale se tiver algo em comum com o próprio nome (`compatible`);
    senão o nó é o nome normalizado.
  - e às vezes junta de menos: US_FEDERAL_RESERVE vs FEDERAL_RESERVE, GOOG vs
    GOOGL. Prefixo de país (US_) cai quando o nome sem ele existe com o mesmo
    tipo; classes de ação da mesma empresa viram um nó; o resto é manual, em
    `prompts/graph_aliases.csv`.
  - o slug de quem não tem ticker colide com tickers: CORN (milho) com o ETF
    CORN, FTC (a agência) com a FTC Solar. Sem ticker, o nó leva o tipo:
    CORN_COMMODITY.

Datas: TODO `published_at` do corpus é meia-noite -- não há hora. Notícia do
dia d pode ter saído depois do fechamento, então o grafo usado para prever o
retorno de d só pode conter notícias ATÉ d-1 (`snapshot` usa `dt < asof`).
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import pyarrow as pa

from ..config import Config, ConfigError
from .link import _write, clean_security_name, entities_dir, norm, slug

# Mudar regras de nós ou de classes muda o grafo: entra na versão.
GRAPH_REV = "1"

# Prefixo de país só cai nestes tipos ("US Steel" é empresa, não aço).
PREFIX_KINDS = ("central_bank", "government", "currency", "macro", "sector", "index", "commodity")
_COUNTRY_PREFIX = re.compile(r"^(?:US|U_S|USA|UNITED_STATES|AMERICAN)_(?=.)")

# Palavras que não bastam para dizer que um nome e o canônico são a mesma coisa.
_GENERIC = set("""
the and of for us usa american united states national international global world new
bank banks central federal government department ministry market markets stock stocks
shares share price prices index sector industry group company companies inc corp
officials policy policies economy economic financial finance institutions
""".split())

# Classe da relação: 137 mil verbos distintos -> 16 classes. Ordem importa
# (primeira que casa): "raises price target on" é analista, não política
# monetária; "buys shares of" é participação, não aquisição.
REL_CLASSES: tuple[tuple[str, str], ...] = (
    ("analyst", r"upgrad|downgrad|price target|\bratings?\b|\brates\b.*\b(buy|sell|hold|neutral)\b|"
                r"coverage|reiterat|recommend|overweight|underweight|initiat"),
    ("monetary", r"\brates?\b|monetary|stimulus|quantitative|easing|tighten|taper|liquidity|"
                 r"bond[- ]buying|asset purchase|money supply"),
    ("capital_return", r"dividend|buyback|repurchas|distribut(es|ion) to shareholders"),
    ("ownership", r"invest|stake|shares (of|in|to)|sells shares|position|portfolio|holds|holding|"
                  r"owns|allocation|exposure|\bshort|insider|outflow|inflow|controls"),
    ("ma", r"acqui|merg|takeover|take over|buyout|\bbids?\b|tender offer|\bbuys?\b|purchas|"
           r"divest|spin|sells (its |a )?(unit|business|division|stake|assets)"),
    ("partnership", r"partner|collaborat|allian|joint venture|teams? up|agreement|deal|licens|"
                    r"cooperat|works with|signs|integrat|co-develop|negotiat|in talks|meets with|"
                    r"discusses with|joins"),
    ("supply_chain", r"suppl|sells to|buys from|customer|client|serves|provid|contract|order|"
                     r"deliver|distribut|ships|vendor|sources|manufactur|outsourc|^sells$|uses"),
    ("competition", r"compet|rival|lags|outpac|outperform|underperform|beats|trails|challeng|"
                    r"vies|steals|disrupt|overtak|gains share|loses share|market share|dominat|"
                    r"compares|contrasts"),
    ("trade_policy", r"tariff|trade|import|export|embargo|quota|sanction"),
    ("legal_regulatory", r"regulat|\bsu(e|es|ing)\b|lawsuit|litigat|investigat|probe|approv|fine|"
                         r"penal|\bban|block|antitrust|charg|settl|court|rul(es|ing)|subpoena|"
                         r"accus|reject|clear|permit|audit|scrutin|oversee|enforc|"
                         r"files? (a )?(complaint|patent)|infring"),
    ("financing", r"lend|loan|financ|credit|debt|bond|underwrit|refinanc|default|bail|"
                  r"guarantee|issues|ipo|raises (capital|funds|money)|funds|funding|borrows|pays"),
    ("impact", r"affect|impact|driv|boost|hurt|weaken|strengthen|increas|reduc|decreas|benefit|"
               r"pressur|drag|influenc|support|lift|lower|rais|cut|weigh|push|pull|fuel|threat|"
               r"harm|damag|help|caus|lead|spur|dampen|depress|hit|rall|slump|fall|rise|gain|"
               r"los|bolster|undermin|improv|worsen|stabiliz|threaten|signal|correlat|track|"
               r"depend|relies|exposed|sensitive|volatil|demand|cool|heat|slow|accelerat|stimulat|"
               r"buoy|encourag|limit|restrict|curb|protect|enabl|advanc|contribut|attract|declin|"
               r"grow|disappoint|delay"),
    # o que um agente diz ou espera de outro, sem efeito descrito
    ("stance", r"critic|oppos|predict|forecast|warn|comment|expect|project|estimat|analy[sz]|"
               r"review|backs|advocat|calls for|urg|prais|endors|respond|concern|monitor|"
               r"disput|reflect|cites|says|believ|questions"),
    # atuação da própria empresa num mercado ou produto
    ("business", r"operat|expand|enter|launch|develop|target|focus|produc|build|lists on|"
                 r"offers|introduc|open|shuts|clos|treats|grows"),
    ("governance", r"hires|appoint|manag|advis|replac|represent|names|elect|nominat"),
)
_REL_RX = [(c, re.compile(rx)) for c, rx in REL_CLASSES]


def rel_class(relation_type: str | None) -> str:
    t = (relation_type or "").lower()
    for c, rx in _REL_RX:
        if rx.search(t):
            return c
    return "other"


def _tokens(s: str) -> set[str]:
    return {t for t in norm(s).split() if len(t) >= 3 and t not in _GENERIC}


def compatible(key: str, canonical: str) -> bool:
    """O canônico do modelo é mesmo um nome desta entidade?

    Basta um token distintivo em comum ("crude oil" ~ "oil prices"), um ser
    prefixo do outro ("fed" ~ "federal reserve"), sigla ("ecb" ~ "european
    central bank") ou um contido no outro sem espaços. "mexico central bank" e
    "nigerian banks" NÃO são compatíveis com "Federal Reserve".
    """
    nk, nc = norm(key), norm(canonical)
    if not nc or nk == nc:
        return True
    tk, tc = _tokens(nk), _tokens(nc)
    if tk & tc:
        return True
    words_k, words_c = nk.split(), nc.split()
    # o lado do canônico pode ser genérico: "fed" ~ "federal"
    if any(len(a) >= 3 and b.startswith(a) for a in words_k if a not in _GENERIC
           for b in words_c):
        return True
    # "eia" ~ "U.S. Energy Information Administration"
    initials = "".join(w[0] for w in words_c if w not in ("and", "of", "the", "us"))
    if re.fullmatch(r"[a-z]{2,6}", nk) and nk == initials:
        return True
    jk, jc = nk.replace(" ", ""), nc.replace(" ", "")
    short, long_ = sorted((jk, jc), key=len)
    return len(short) >= 4 and short in long_


# --- aliases manuais ---------------------------------------------------------

def load_aliases(path: Path) -> dict[str, str]:
    """`prompts/graph_aliases.csv`: from,to,note. `from` é um nó ou `key:<chave>`."""
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(line for line in f if not line.lstrip().startswith("#")):
            a, b = (row.get("from") or "").strip(), (row.get("to") or "").strip()
            if a and b:
                out[a] = b
    return out


def _resolve(m: dict[str, str], x: str) -> str:
    seen = set()
    while x in m and x not in seen:
        seen.add(x)
        x = m[x]
    return x


def graph_dir(cfg: Config, relations_version: str) -> Path:
    return cfg.curated / "graph" / f"v={relations_version}"


def graph_version(link_version: str, aliases_path: Path) -> str:
    h = hashlib.sha256()
    for part in (GRAPH_REV, link_version, json.dumps(REL_CLASSES), json.dumps(PREFIX_KINDS),
                 aliases_path.read_text() if aliases_path.exists() else ""):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()[:12]


# --- build -------------------------------------------------------------------

def _node_map(con: duckdb.DuckDBPyConnection, d: Path, aliases: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """Chave do link -> nó; e a tabela de nós."""
    em = con.execute(f"""SELECT key, n, kind, canonical, ticker, has_price, variants
                         FROM read_parquet('{d / 'entity_map.parquet'}')""").fetchall()
    tk = con.execute(f"""SELECT ticker, name, is_etf, has_price, news_mentions
                         FROM read_parquet('{d / 'tickers.parquet'}')""").fetchall()
    price = {t: bool(hp) for t, _, _, hp, _ in tk}

    # 1. classes de ação: mesmo nome de empresa (sem "Class A"), não ETF ->
    #    o ticker mais citado com preço. Ações preferenciais quase não aparecem.
    used = Counter()
    for _, n, _, _, t, _, _ in em:
        if t:
            used[t] += n
    by_name = defaultdict(list)
    for t, name, etf, _, _ in tk:
        nm = norm(clean_security_name(name or ""))
        if nm and not etf and used[t]:
            by_name[nm].append(t)
    share = {}
    for ts in by_name.values():
        if len(ts) > 1:
            best = max(ts, key=lambda t: (price.get(t, False), used[t]))
            share.update({t: best for t in ts if t != best})

    tickers_used = set(used)
    tmap = {**share, **{a: b for a, b in aliases.items() if a in tickers_used}}

    # 2. chave -> nó provisório
    raw = {}
    for key, n, kind, can, t, _, _ in em:
        if t:
            raw[key] = (_resolve(tmap, t), kind, True, n)
        else:
            base = slug(can) if can and compatible(key, can) else slug(key)
            raw[key] = (base or slug(key), kind or "other", False, n)

    # tipo de cada nó sem ticker: o mais frequente entre as suas chaves
    kinds = defaultdict(Counter)
    for nid, kind, has_t, n in raw.values():
        if not has_t:
            kinds[nid][kind] += n
    kind_of = {nid: c.most_common(1)[0][0] for nid, c in kinds.items()}
    size = Counter()
    for nid, _, has_t, n in raw.values():
        if not has_t:
            size[nid] += n

    # 3. prefixo de país: US_TREASURY -> TREASURY quando ambos existem com o
    #    mesmo tipo; fica o mais citado.
    pmap = {}
    for nid in list(size):
        bare = _COUNTRY_PREFIX.sub("", nid)
        if bare != nid and bare in size and kind_of[nid] == kind_of[bare] and kind_of[nid] in PREFIX_KINDS:
            a, b = sorted((nid, bare), key=lambda x: -size[x])
            pmap[b] = a

    # 4. manuais (nós e key:<chave>), depois o resto
    nmap = {**pmap, **{a: b for a, b in aliases.items() if not a.startswith("key:")}}
    keyover = {a[4:]: b for a, b in aliases.items() if a.startswith("key:")}

    out, label_votes, members = [], defaultdict(Counter), defaultdict(set)
    final_kind = defaultdict(Counter)
    for key, n, kind, can, t, _, variants in em:
        nid, k, has_t, _ = raw[key]
        nid = keyover.get(key) or _resolve(nmap, nid)
        method = ("manual" if key in keyover else
                  "ticker" if has_t else
                  "canonical" if can and compatible(key, can) else "key")
        out.append({"key": key, "n": n, "node_tmp": nid, "has_ticker": has_t, "method": method})
        label_votes[nid][(can or (variants or [key])[0])] += n
        final_kind[nid][kind or "other"] += n
        if has_t:
            members[nid].add(t)

    # 5. colisão slug sem ticker x ticker -> leva o tipo
    ticker_nodes = {r["node_tmp"] for r in out if r["has_ticker"]}
    nodes_n = Counter()
    for r in out:
        nid = r["node_tmp"]
        if not r["has_ticker"] and nid in ticker_nodes:
            nid = f"{nid}_{final_kind[nid].most_common(1)[0][0].upper()}"
        r["node_id"] = nid
        nodes_n[nid] += r["n"]

    node_rows = {}
    for r in out:
        nid, old = r["node_id"], r["node_tmp"]
        if nid in node_rows:
            node_rows[nid]["n_keys"] += 1
            continue
        is_t = r["has_ticker"]
        ts = sorted(members[old]) if is_t else []
        node_rows[nid] = {
            "node_id": nid,
            "label": label_votes[old].most_common(1)[0][0],
            "kind": final_kind[old].most_common(1)[0][0],
            "ticker": nid if is_t else None,
            "tickers": ts,
            "has_price": price.get(nid, False) if is_t else False,
            "n_mentions": nodes_n[nid],
            "n_keys": 1,
        }
    keymap = [{"key": r["key"], "node_id": r["node_id"], "method": r["method"]} for r in out]
    return keymap, list(node_rows.values())


def build(cfg: Config, relations_version: str, aliases_path: Path, views: bool = True) -> dict:
    d = entities_dir(cfg, relations_version)
    for f in ("entity_map.parquet", "names.parquet", "tickers.parquet", "relations_linked.parquet"):
        if not (d / f).exists():
            raise ConfigError(f"{d / f} não existe; rode link-build antes")
    out = graph_dir(cfg, relations_version)
    out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    lv = con.execute(f"SELECT any_value(link_version) FROM read_parquet('{d / 'entity_map.parquet'}')").fetchone()[0]
    gv = graph_version(lv, aliases_path)
    aliases = load_aliases(aliases_path)

    keymap, nodes = _node_map(con, d, aliases)
    _write(keymap, pa.schema([("key", pa.string()), ("node_id", pa.string()), ("method", pa.string())]),
           out / "node_keys.parquet")

    rels = [r[0] for r in con.execute(
        f"SELECT DISTINCT relation_type FROM read_parquet('{d / 'relations_linked.parquet'}')").fetchall()]
    _write([{"relation_type": t, "rel_class": rel_class(t)} for t in rels],
           pa.schema([("relation_type", pa.string()), ("rel_class", pa.string())]), out / "rel_classes.parquet")

    # Eventos: todas as tuplas, com o ruído marcado (como relations_flagged).
    # Laço criado pela fusão de nós (Fed -> FOMC) vira ruído 'self_loop'.
    nk, nm = out / "node_keys.parquet", d / "names.parquet"
    con.execute(f"""COPY (
        SELECT row_number() OVER (ORDER BY r.published_at, r.unit_id, r.agent_a, r.agent_b, r.relation_type) AS event_id,
               CAST(r.published_at AS DATE) AS dt, r.unit_id, r.repr_doc_id AS doc_id,
               ka.node_id AS src, kb.node_id AS dst,
               r.relation_type, rc.rel_class,
               CASE r.direction WHEN 'positive' THEN 1 WHEN 'negative' THEN -1 ELSE 0 END::TINYINT AS sign,
               r.strength,
               (CASE r.direction WHEN 'positive' THEN 1 WHEN 'negative' THEN -1 ELSE 0 END) * r.strength AS w,
               r.agent_a, r.agent_b,
               CASE WHEN r.noise IS NULL AND ka.node_id = kb.node_id THEN 'self_loop' ELSE r.noise END AS noise
        FROM read_parquet('{d / 'relations_linked.parquet'}') r
        LEFT JOIN read_parquet('{nm}') na ON na.raw = r.agent_a
        LEFT JOIN read_parquet('{nk}') ka ON ka.key = na.key
        LEFT JOIN read_parquet('{nm}') nb ON nb.raw = r.agent_b
        LEFT JOIN read_parquet('{nk}') kb ON kb.key = nb.key
        LEFT JOIN read_parquet('{out / 'rel_classes.parquet'}') rc ON rc.relation_type = r.relation_type
        ORDER BY dt
    ) TO '{out / 'events.parquet'}' (FORMAT parquet, COMPRESSION zstd)""")

    # Nós com datas e grau (só eventos limpos).
    con.register("nodes_arrow", pa.Table.from_pylist(nodes, schema=pa.schema([
        ("node_id", pa.string()), ("label", pa.string()), ("kind", pa.string()), ("ticker", pa.string()),
        ("tickers", pa.list_(pa.string())), ("has_price", pa.bool_()), ("n_mentions", pa.int64()),
        ("n_keys", pa.int64())])))
    ev = f"read_parquet('{out / 'events.parquet'}')"
    con.execute(f"""COPY (
        WITH e AS (SELECT dt, src AS node_id, dst AS other FROM {ev} WHERE noise IS NULL
                   UNION ALL SELECT dt, dst, src FROM {ev} WHERE noise IS NULL),
             s AS (SELECT node_id, count(*) AS n_events, count(DISTINCT other) AS degree,
                          min(dt) AS first_dt, max(dt) AS last_dt FROM e GROUP BY 1)
        SELECT n.*, coalesce(s.n_events, 0) AS n_events, coalesce(s.degree, 0) AS degree,
               s.first_dt, s.last_dt, '{gv}' AS graph_version
        FROM nodes_arrow n LEFT JOIN s USING (node_id)
    ) TO '{out / 'nodes.parquet'}' (FORMAT parquet, COMPRESSION zstd)""")
    (out / "meta.json").write_text(json.dumps({
        "graph_version": gv, "graph_rev": GRAPH_REV, "link_version": lv,
        "relations_version": relations_version, "aliases": str(aliases_path)}, indent=2))

    st = con.execute(f"""SELECT count(*), count(*) FILTER (WHERE noise IS NULL),
                                count(*) FILTER (WHERE noise = 'self_loop'),
                                count(DISTINCT (src, dst)) FILTER (WHERE noise IS NULL),
                                count(*) FILTER (WHERE noise IS NULL AND rel_class = 'other')
                         FROM {ev}""").fetchone()
    nst = con.execute(f"""SELECT count(*), count(*) FILTER (WHERE n_events > 0),
                                 count(*) FILTER (WHERE ticker IS NOT NULL AND n_events > 0)
                          FROM read_parquet('{out / 'nodes.parquet'}')""").fetchone()
    ids_before = con.execute(f"SELECT count(DISTINCT entity_id) FROM read_parquet('{d / 'entity_map.parquet'}')").fetchone()[0]
    classes = con.execute(f"""SELECT rel_class, count(*) FROM {ev} WHERE noise IS NULL
                              GROUP BY 1 ORDER BY 2 DESC""").fetchall()
    methods = con.execute(f"SELECT method, count(*) FROM read_parquet('{nk}') GROUP BY 1 ORDER BY 2 DESC").fetchall()
    con.close()

    if views:
        register_views(cfg, relations_version)

    total, clean, loops, pairs, other = st
    print(f"[graph-build] g={gv} (link l={lv})")
    print(f"[graph-build] nós: {nst[0]:,} (entity_id do link: {ids_before:,}); "
          f"com evento limpo {nst[1]:,}, dos quais com ticker {nst[2]:,}")
    print("[graph-build] chave -> nó: " + ", ".join(f"{m} {c:,}" for m, c in methods))
    print(f"[graph-build] eventos: {total:,}; limpos {clean:,} ({100 * clean / total:.1f}%); "
          f"laços novos pela fusão {loops:,}; pares (src,dst) distintos {pairs:,}")
    print("[graph-build] classes: " + ", ".join(f"{c} {100 * k / clean:.1f}%" for c, k in classes))
    print(f"[graph-build] {out}")
    return {"graph_version": gv, "nodes": nst[0], "events": total, "clean": clean,
            "pairs": pairs, "other_class": other}


def register_views(cfg: Config, relations_version: str) -> None:
    out = graph_dir(cfg, relations_version)
    ev, nd = out / "events.parquet", out / "nodes.parquet"
    db = duckdb.connect(str(cfg.db))
    db.execute("CREATE SCHEMA IF NOT EXISTS news")
    db.execute(f"CREATE OR REPLACE VIEW news.graph_nodes AS SELECT * FROM read_parquet('{nd}')")
    db.execute(f"CREATE OR REPLACE VIEW news.graph_node_keys AS SELECT * FROM read_parquet('{out / 'node_keys.parquet'}')")
    db.execute(f"CREATE OR REPLACE VIEW news.graph_events_all AS SELECT * FROM read_parquet('{ev}')")
    db.execute("CREATE OR REPLACE VIEW news.graph_events AS "
               "SELECT * EXCLUDE (noise) FROM news.graph_events_all WHERE noise IS NULL")
    # Arestas por período: uma linha por (período, src, dst, classe).
    for name, unit in (("weekly", "week"), ("monthly", "month")):
        db.execute(f"""CREATE OR REPLACE VIEW news.graph_edges_{name} AS
            SELECT date_trunc('{unit}', dt)::DATE AS period, src, dst, rel_class,
                   count(*) AS n_events, count(DISTINCT doc_id) AS n_docs,
                   count(*) FILTER (WHERE sign > 0) AS n_pos, count(*) FILTER (WHERE sign < 0) AS n_neg,
                   sum(w) AS w_sum, avg(w) AS w_mean, avg(strength) AS strength_mean
            FROM news.graph_events GROUP BY ALL""")
    db.close()
