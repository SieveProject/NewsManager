"""Liga os agentes das tuplas a tickers -- etapa `link`.

Um backtest precisa de `ticker_a`/`ticker_b`, não de "Apple", "Apple Inc." e
"AAPL" como três agentes. Três passos:

  link-prep   (sem GPU) normaliza os nomes, monta o universo de tickers e, para
              cada nome distinto, uma LISTA FECHADA de candidatos.
  link-run    (GPU) um modelo leve escolhe o candidato -- ou NONE -- e classifica
              o tipo de entidade e um nome canônico.
  link-build  (sem GPU) grava entity_map + relations_linked e as views.

Por que lista fechada: pedido a "dar o ticker", um modelo de 3-7B inventa
tickers plausíveis. Com os candidatos como `enum` do schema, a tarefa vira
classificação e um ticker fora do universo é impossível de emitir.

Universo = tickers com preço no FNSPID (Stock_price/full_history.zip, lido por
HTTP range: só o diretório central) UNIÃO os da coluna `symbols` das notícias.
Medido na run v=a05d0c0dc189: 7.693 tickers com preço, e só 35 das notícias
ficam de fora (2,8% das menções a símbolos -- BRK, SNAP e warrants espúrios).
Cada ticker leva `has_price`; o backtest filtra.

Candidatos, por ordem de força da evidência:
  explicit   o próprio nome traz o ticker: "Vanguard Tax-Exempt Bond (VTEB)",
             "NASDAQ: AAPL". Só evidência, não decisão: "crude oil (WTI)" é West
             Texas Intermediate, não W&T Offshore, e "Facebook (FB)" aponta
             para um ticker que hoje é de um ETF
  exact      nome normalizado igual ao da listagem da Nasdaq Trader
  symbol     o nome É um ticker do universo ("AMD", "IBM")
  cooc       tickers da coluna `symbols` dos artigos onde o nome aparece. Medido
             nos nomes frequentes: Tesla->TSLA 48%, Disney->DIS 65%, Deutsche
             Bank AG->DB 79%. Também traz concorrentes (Nvidia->AMD 27%): o
             prompt diz que co-menção sozinha não é motivo
  fuzzy      nome parecido na listagem (mesmo primeiro token + Jaro-Winkler, ou
             prefixo: "Keysight" ~ "Keysight Technologies")

A listagem da Nasdaq Trader é um retrato de HOJE: empresas deslistadas não têm
nome nela. Para essas o candidato leva o nome mais escrito para o ticker nas
próprias tuplas ("usually written as ...").

Entidades que não são empresa (Fed, China, petróleo, S&P 500) não ganham
ticker: ganham `kind` e um `entity_id` canônico (FEDERAL_RESERVE, CHINA...).
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
import time
import unicodedata
import zipfile
from pathlib import Path

import duckdb
import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import Config, ConfigError
from .client import OllamaClient, OllamaConfig

PRICE_ZIP_URL = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip"
DIRECTORY_URLS = {
    "nasdaqlisted.txt": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "otherlisted.txt": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
}

# Mudar a geração de candidatos muda o que o modelo vê: entra na versão.
CANDIDATES_REV = "1"
MAX_CANDIDATES = 8

KINDS = ("company", "fund", "index", "central_bank", "government", "country",
         "currency", "commodity", "macro", "sector", "person", "other")
# Só estes podem ter ticker; para o resto o ticker do modelo é descartado.
TICKER_KINDS = ("company", "fund")

# Mesmos nomes que o `filter` aceita: ruído não precisa de entidade. Sem
# comparison/membership, que a análise pode querer manter.
NOISE_SKIP = ("self_loop", "source", "metric", "generic", "orgchart")


# --- normalização ------------------------------------------------------------

_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies",
    "ltd", "limited", "plc", "llc", "lp", "llp", "ag", "sa", "nv", "se", "spa",
    "holdings", "holding", "group", "the", "cos",
}
_EXCHANGES = r"(?:NYSE(?:\s*(?:Arca|American|MKT))?|NASDAQ|Nasdaq|NYSEARCA|NYSEMKT|AMEX|BATS|OTC(?:QX|QB)?)"
_EXPLICIT = (
    re.compile(rf"\(\s*(?:{_EXCHANGES}\s*:\s*)?([A-Z]{{1,5}}(?:[.\-][A-Z])?)\s*\)"),
    re.compile(rf"\b{_EXCHANGES}\s*:\s*([A-Z]{{1,5}}(?:[.\-][A-Z])?)\b"),
)
_TICKERISH = re.compile(r"^[A-Z]{1,5}(?:[.\-][A-Z])?$")


def norm(s: str) -> str:
    """Chave de agrupamento: 'The Apple Inc.' == 'Apple' == 'apple, inc'."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"['’`]s\b", "", s)          # possessivo: "Apple's"
    s = s.replace("&", " and ").replace(".", "")  # "U.S." -> "us", "Inc." -> "inc"
    toks = re.sub(r"[^a-z0-9]+", " ", s).split()
    while toks and toks[0] == "the":
        toks = toks[1:]
    while len(toks) > 1 and toks[-1] in _SUFFIXES:
        toks = toks[:-1]
    return " ".join(toks)


def explicit_ticker(raw: str, universe: set[str]) -> str | None:
    for rx in _EXPLICIT:
        for m in rx.finditer(raw or ""):
            t = m.group(1).replace(".", "-").upper()
            if t in universe:
                return t
    return None


def clean_security_name(name: str) -> str:
    """'Alcoa Corporation Common Stock ' -> 'Alcoa Corporation'."""
    n = (name or "").split(" - ")[0]
    n = re.sub(r"\s+(Common Stock|Ordinary Shares|Common Shares|Class [A-Z]\b.*|"
               r"American Depositary.*|Depositary Shares.*|Units?|Warrants?|Rights?)\s*$",
               "", n.strip(), flags=re.I)
    return n.strip()


def slug(s: str) -> str:
    return norm(s).upper().replace(" ", "_")


# --- referências -------------------------------------------------------------

class _RemoteFile(io.RawIOBase):
    """Arquivo HTTP com seek: o zipfile lê só o diretório central (~1 MB) dos 590 MB."""

    def __init__(self, url: str):
        self._c = httpx.Client(follow_redirects=True, timeout=120)
        self._u, self._p = url, 0
        self._n = int(self._c.head(url).headers["content-length"])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self._p

    def seek(self, off, whence=0):
        self._p = {0: off, 1: self._p + off, 2: self._n + off}[whence]
        return self._p

    def read(self, k=-1):
        if k is None or k < 0:
            k = self._n - self._p
        if k == 0:
            return b""
        r = self._c.get(self._u, headers={"Range": f"bytes={self._p}-{self._p + k - 1}"})
        r.raise_for_status()
        self._p += len(r.content)
        return r.content


def ref_dir(cfg: Config) -> Path:
    return cfg.root / "reference"


def fetch_reference(cfg: Config, refresh: bool = False) -> dict[str, Path]:
    """Baixa (uma vez) a lista de tickers com preço e a listagem da Nasdaq Trader.

    Ficam em data/reference/ e entram no hash da versão: a listagem muda todo
    dia, e uma rodada nova com listagem nova não pode se misturar à antiga.
    """
    d = ref_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    out = {"price_tickers.txt": d / "price_tickers.txt"}
    if refresh or not out["price_tickers.txt"].exists():
        z = zipfile.ZipFile(_RemoteFile(PRICE_ZIP_URL))
        ticks = sorted({Path(n).stem.upper() for n in z.namelist()
                        if n.startswith("full_history/") and n.endswith(".csv")})
        out["price_tickers.txt"].write_text("\n".join(ticks) + "\n")
        print(f"[link] {len(ticks):,} tickers com preço em {PRICE_ZIP_URL.rsplit('/', 1)[-1]}")
    for name, url in DIRECTORY_URLS.items():
        out[name] = d / name
        if refresh or not out[name].exists():
            r = httpx.get(url, timeout=120, follow_redirects=True)
            r.raise_for_status()
            out[name].write_bytes(r.content)
    return out


def read_directory(paths: dict[str, Path]) -> dict[str, tuple[str, bool]]:
    """ticker -> (nome da empresa, é ETF). Formato '|' da Nasdaq Trader."""
    out: dict[str, tuple[str, bool]] = {}
    for name in DIRECTORY_URLS:
        lines = paths[name].read_text(encoding="latin-1").splitlines()
        head = lines[0].split("|")
        i_sym, i_name, i_etf = 0, head.index("Security Name"), head.index("ETF")
        for line in lines[1:]:
            if line.startswith("File Creation"):
                continue
            p = line.split("|")
            if len(p) <= max(i_name, i_etf):
                continue
            t = p[i_sym].replace(".", "-").upper()
            out.setdefault(t, (clean_security_name(p[i_name]), p[i_etf] == "Y"))
    return out


def ref_hash(paths: dict[str, Path]) -> str:
    h = hashlib.sha256()
    for k in sorted(paths):
        h.update(paths[k].read_bytes())
    return h.hexdigest()[:12]


# --- link-prep ---------------------------------------------------------------

def entities_dir(cfg: Config, relations_version: str) -> Path:
    return cfg.curated / "entities" / f"v={relations_version}"


def _relations_path(cfg: Config, relations_version: str) -> Path:
    p = cfg.curated / "relations" / f"v={relations_version}" / "relations_flagged.parquet"
    if not p.exists():
        raise ConfigError(f"{p} não existe; rode `filter` antes")
    return p


def _write(rows: list[dict], schema: pa.Schema, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp, compression="zstd")
    tmp.rename(path)


def prepare(cfg: Config, relations_version: str, refresh: bool = False,
            reference: dict[str, Path] | None = None) -> dict:
    """Grava tickers.parquet, names.parquet e candidates.parquet."""
    src = _relations_path(cfg, relations_version)
    paths = reference or fetch_reference(cfg, refresh)
    out = entities_dir(cfg, relations_version)
    out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    skip = ", ".join(f"'{r}'" for r in NOISE_SKIP)
    con.execute(f"CREATE TEMP VIEW r AS SELECT * FROM read_parquet('{src}') "
                f"WHERE noise IS NULL OR noise NOT IN ({skip})")

    # Universo
    price = {x.strip().upper() for x in paths["price_tickers.txt"].read_text().split() if x.strip()}
    news = dict(con.execute(
        "SELECT upper(s), count(*) FROM (SELECT unnest(symbols) s FROM r) WHERE s IS NOT NULL GROUP BY 1"
    ).fetchall())
    directory = read_directory(paths)
    universe = price | set(news)

    # Nomes crus -> chave normalizada (+ ticker explícito no próprio nome)
    raw = con.execute(
        "SELECT x, count(*) FROM (SELECT agent_a x FROM r UNION ALL SELECT agent_b FROM r) "
        "WHERE x IS NOT NULL AND trim(x) <> '' GROUP BY 1"
    ).fetchall()
    names = []
    for x, n in raw:
        k = norm(x)
        if k:
            names.append({"raw": x, "key": k, "n": n, "explicit": explicit_ticker(x, universe)})
    names_schema = pa.schema([("raw", pa.string()), ("key", pa.string()), ("n", pa.int64()),
                              ("explicit", pa.string())])
    _write(names, names_schema, out / "names.parquet")
    con.execute(f"CREATE TABLE names AS SELECT * FROM read_parquet('{out / 'names.parquet'}')")

    con.execute("""CREATE TABLE keys AS
        SELECT key, sum(n)::BIGINT n, list(raw ORDER BY n DESC, raw)[1:5] variants,
               arg_max(explicit, n) FILTER (WHERE explicit IS NOT NULL) explicit
        FROM names GROUP BY key""")

    # Cada menção (chave, artigo, tuplas) -- base dos exemplos e da co-menção
    con.execute("""CREATE TABLE m AS
        SELECT n.key, r.unit_id, r.symbols, r.agent_a, r.relation_type, r.agent_b, r.year
        FROM r JOIN names n ON n.raw = r.agent_a
        UNION ALL
        SELECT n.key, r.unit_id, r.symbols, r.agent_a, r.relation_type, r.agent_b, r.year
        FROM r JOIN names n ON n.raw = r.agent_b""")
    examples = dict(con.execute("""
        SELECT key, list(left(agent_a, 70) || ' --' || left(relation_type, 40) || '--> '
                         || left(agent_b, 70) || ' (' || coalesce(year::VARCHAR, '?') || ')'
                         ORDER BY h)[1:3]
        FROM (SELECT *, hash(unit_id || agent_a || agent_b) h FROM m) GROUP BY key""").fetchall())

    con.execute("""CREATE TABLE co AS
        WITH ku AS (SELECT DISTINCT key, unit_id, symbols FROM m),
        kn AS (SELECT key, count(*) units FROM ku WHERE len(symbols) > 0 GROUP BY 1),
        kt AS (SELECT key, unit_id, upper(unnest(symbols)) t FROM ku),
        c AS (SELECT key, t, count(DISTINCT unit_id) k FROM kt GROUP BY 1, 2),
        tn AS (SELECT t, count(DISTINCT unit_id) units FROM kt GROUP BY 1)
        SELECT c.key, c.t, c.k, c.k / kn.units AS share, c.k / tn.units AS t_share,
               row_number() OVER (PARTITION BY c.key ORDER BY c.k DESC, c.t) rk
        FROM c JOIN kn USING (key) JOIN tn USING (t)""")
    cooc: dict[str, list[tuple[str, float]]] = {}
    for k, t, share in con.execute("SELECT key, t, share FROM co WHERE rk <= 5 ORDER BY key, rk").fetchall():
        cooc.setdefault(k, []).append((t, share))

    # Como as notícias chamam cada ticker: o nome mais escrito entre os que o
    # têm em >= 30% dos próprios artigos E aparecem em >= 15% dos artigos do
    # ticker. Dá nome a deslistados (OCN -> Ocwen) e denuncia ticker
    # reutilizado (FB é hoje um ETF da ProShares; nas notícias, Facebook). Sem
    # o segundo corte, warrants marcados em artigos aleatórios (ARTLW, BCDAW)
    # herdavam nomes como "AI technologies".
    alias = dict(con.execute("""
        SELECT co.t, arg_max(keys.variants[1], co.k) FROM co JOIN keys USING (key)
        WHERE co.share >= 0.3 AND co.t_share >= 0.15 AND co.k >= 3 GROUP BY 1""").fetchall())

    # Listagem normalizada, restrita ao universo
    dir_rows = [{"t": t, "dkey": norm(nm), "first": norm(nm).split(" ")[0]}
                for t, (nm, _) in directory.items() if t in universe and norm(nm)]
    con.register("dir_arrow", pa.Table.from_pylist(
        dir_rows, schema=pa.schema([("t", pa.string()), ("dkey", pa.string()), ("first", pa.string())])))
    con.execute("CREATE TABLE dirn AS SELECT * FROM dir_arrow")
    exact: dict[str, list[str]] = {}
    for k, t in con.execute("SELECT k.key, d.t FROM keys k JOIN dirn d ON k.key = d.dkey ORDER BY d.t").fetchall():
        exact.setdefault(k, []).append(t)
    fuzzy: dict[str, list[str]] = {}
    for k, t in con.execute("""
        WITH p AS (
          SELECT k.key, d.t, jaro_winkler_similarity(k.key, d.dkey) sim
          FROM keys k JOIN dirn d ON split_part(k.key, ' ', 1) = d.first
          WHERE k.key <> d.dkey AND (
                jaro_winkler_similarity(k.key, d.dkey) >= 0.92
             OR (length(k.key) >= 4 AND starts_with(d.dkey, k.key || ' '))
             OR (length(d.dkey) >= 4 AND starts_with(k.key, d.dkey || ' '))))
        SELECT key, t FROM (SELECT *, row_number() OVER (PARTITION BY key ORDER BY sim DESC, t) rk FROM p)
        WHERE rk <= 3 ORDER BY key, rk""").fetchall():
        fuzzy.setdefault(k, []).append(t)
    key_rows = con.execute("SELECT key, n, variants, explicit FROM keys ORDER BY n DESC, key").fetchall()
    con.close()

    def label(t: str) -> str:
        if t in directory:
            nm, etf = directory[t]
            out = f"{nm}{' (ETF)' if etf else ''}"
            a = alias.get(t)
            if a and norm(a).split(" ")[0] != norm(nm).split(" ")[0]:
                out = f"listed today as {out}; in these news it is '{a[:40]}'"
            return out
        if t in alias:
            return f"(no current listing; usually written as '{alias[t]}')"
        return "(name unknown)"

    # Universo gravado para o backtest saber o que tem preço
    tick_rows = [{"ticker": t, "name": directory.get(t, (alias.get(t), False))[0],
                  "is_etf": directory.get(t, ("", False))[1], "has_price": t in price,
                  "news_mentions": int(news.get(t, 0))} for t in sorted(universe)]
    _write(tick_rows, pa.schema([("ticker", pa.string()), ("name", pa.string()), ("is_etf", pa.bool_()),
                                 ("has_price", pa.bool_()), ("news_mentions", pa.int64())]),
           out / "tickers.parquet")

    cand_rows = []
    n_with = 0
    for key, n, variants, expl in key_rows:
        cands: list[dict] = []
        seen: set[str] = set()

        def add(t: str, ev: str) -> None:
            if t in universe and t not in seen and len(cands) < MAX_CANDIDATES:
                seen.add(t)
                cands.append({"ticker": t, "name": label(t), "evidence": ev, "has_price": t in price})

        if expl:
            add(expl, "the entity name contains this ticker")
        for t in exact.get(key, []):
            add(t, "same company name")
        for v in variants:
            if _TICKERISH.match(v.strip()):
                add(v.strip().replace(".", "-"), "the name is this ticker")
        for t, share in cooc.get(key, []):
            add(t, f"tagged on {share:.0%} of the articles that mention the entity")
        for t in fuzzy.get(key, []):
            add(t, "similar company name")
        n_with += bool(cands)
        cand_rows.append({"key": key, "n": n, "variants": variants, "explicit": expl,
                          "examples": examples.get(key, []), "candidates": cands})

    cand_schema = pa.schema([
        ("key", pa.string()), ("n", pa.int64()), ("variants", pa.list_(pa.string())),
        ("explicit", pa.string()), ("examples", pa.list_(pa.string())),
        ("candidates", pa.list_(pa.struct([("ticker", pa.string()), ("name", pa.string()),
                                           ("evidence", pa.string()), ("has_price", pa.bool_())]))),
    ])
    _write(cand_rows, cand_schema, out / "candidates.parquet")
    (out / "reference.json").write_text(json.dumps(
        {"ref_hash": ref_hash(paths), "candidates_rev": CANDIDATES_REV,
         "files": {k: str(v) for k, v in paths.items()}}, indent=2))

    total = sum(r["n"] for r in cand_rows)
    stats = {
        "universe": len(universe), "with_price": len(price & universe), "names": len(names),
        "keys": len(cand_rows), "keys_with_candidates": n_with,
        "explicit": sum(1 for r in cand_rows if r["explicit"]),
        "mentions": total,
    }
    print(f"[link-prep] universo: {len(universe):,} tickers ({len(price & universe):,} com preço)")
    print(f"[link-prep] {len(names):,} nomes crus -> {len(cand_rows):,} chaves normalizadas "
          f"({total:,} menções)")
    print(f"[link-prep] com candidato: {n_with:,} chaves; ticker entre parênteses no nome: {stats['explicit']:,}")
    print(f"[link-prep] {out}")
    return stats


# --- link-run ----------------------------------------------------------------

def load_template(path: Path) -> str:
    t = Path(path).read_text(encoding="utf-8")
    for ph in ("{{NAME}}", "{{CANDIDATES}}"):
        if ph not in t:
            raise ConfigError(f"{path} precisa do marcador {ph}")
    return t


def link_version(template: str, model: str, cand_dir: Path) -> str:
    meta = json.loads((cand_dir / "reference.json").read_text())
    h = hashlib.sha256()
    for part in (template, model, json.dumps(KINDS), meta["ref_hash"], meta["candidates_rev"]):
        h.update(part.encode())
    return h.hexdigest()[:12]


def render(template: str, row: dict) -> str:
    variants = [v[:60] for v in row["variants"]]
    cands = row["candidates"] or []
    lines = [f"  {c['ticker']:<7} | {c['name'][:70]} | {c['evidence']}" for c in cands]
    out = template
    for ph, val in (
        ("{{NAME}}", row["variants"][0][:80] if row["variants"] else row["key"]),
        ("{{VARIANTS}}", "; ".join(variants[1:]) or "(none)"),
        ("{{EXAMPLES}}", "\n".join(f"  {e}" for e in row["examples"]) or "  (none)"),
        ("{{CANDIDATES}}", "\n".join(lines) or "  (no candidates -- answer NONE)"),
    ):
        out = out.replace(ph, val)
    return out


def schema_for(row: dict) -> dict:
    tickers = [c["ticker"] for c in (row["candidates"] or [])]
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": list(KINDS)},
            "ticker": {"type": "string", "enum": tickers + ["NONE"]},
            "link": {"type": "string", "enum": ["issuer", "parent", "none"]},
            "canonical": {"type": "string"},
        },
        "required": ["kind", "ticker", "link", "canonical"],
    }


def max_prompt_chars(template: str) -> int:
    """Pior caso de render(): cada campo no limite de corte."""
    return (len(template) + 80 + 4 * 62 + 3 * (70 + 40 + 70 + 20)
            + MAX_CANDIDATES * (2 + 7 + 3 + 70 + 3 + 70 + 1))


RESULT_SCHEMA = pa.schema([
    ("key", pa.string()), ("ok", pa.bool_()), ("kind", pa.string()), ("ticker", pa.string()),
    ("link", pa.string()), ("canonical", pa.string()), ("n_candidates", pa.int32()),
    ("error", pa.string()), ("raw", pa.string()), ("latency_s", pa.float64()),
    ("model", pa.string()), ("link_version", pa.string()),
])


def runs_dir(cfg: Config, relations_version: str, version: str) -> Path:
    return entities_dir(cfg, relations_version) / "runs" / f"l={version}"


def _done(d: Path) -> set[str]:
    if not any(d.glob("part-*.parquet")):
        return set()
    c = duckdb.connect()
    keys = {k for (k,) in c.execute(f"SELECT key FROM read_parquet('{d}/part-*.parquet') WHERE ok").fetchall()}
    c.close()
    return keys


async def _one(cl: OllamaClient, template: str, row: dict, version: str) -> dict:
    res = await cl.generate(render(template, row), schema=schema_for(row))
    p = res.parsed if res.ok and isinstance(res.parsed, dict) else None
    ok = p is not None and p.get("kind") in KINDS
    allowed = {c["ticker"] for c in (row["candidates"] or [])}
    t = p.get("ticker") if ok else None
    return {
        "key": row["key"], "ok": ok,
        "kind": p.get("kind") if ok else None,
        "ticker": t if t in allowed else None,
        "link": p.get("link") if ok else None,
        "canonical": (p.get("canonical") or "").strip()[:120] if ok else None,
        "n_candidates": len(allowed), "error": res.error or (None if ok else "schema"),
        "raw": res.text[:500], "latency_s": res.latency_s,
        "model": cl.cfg.model, "link_version": version,
    }


async def _run(cfg: Config, relations_version: str, oll: OllamaConfig, template: str,
               limit: int | None, order: str, min_n: int, segment: int) -> dict:
    cand_dir = entities_dir(cfg, relations_version)
    version = link_version(template, oll.model, cand_dir)
    out = runs_dir(cfg, relations_version, version)
    out.mkdir(parents=True, exist_ok=True)
    done = _done(out)
    rows = pq.read_table(cand_dir / "candidates.parquet").to_pylist()
    todo = [r for r in rows if r["n"] >= min_n and r["key"] not in done]
    if order == "hash":
        todo.sort(key=lambda r: hashlib.md5(r["key"].encode()).hexdigest())
    if limit:
        todo = todo[:limit]
    print(f"[link-run] l={version}  modelo {oll.model}  já feitas {len(done):,}  "
          f"a fazer {len(todo):,} (n >= {min_n}, ordem {order})")
    print(f"[link-run] {out}")

    t0, n_ok, n_fail = time.time(), 0, 0
    async with OllamaClient(oll) as cl:
        h = await cl.health()
        if not h["has_model"]:
            raise ConfigError(f"modelo {oll.model} ausente no Ollama ({h['models']}); `ollama pull` antes")
        await cl.warm()
        for i in range(0, len(todo), segment):
            chunk = todo[i:i + segment]
            res = await asyncio.gather(*(_one(cl, template, r, version) for r in chunk))
            name = f"part-{int(time.time() * 1000)}-{i // segment:05d}.parquet"
            _write(res, RESULT_SCHEMA, out / name)
            n_ok += sum(r["ok"] for r in res)
            n_fail += sum(not r["ok"] for r in res)
            dt = time.time() - t0
            rate = (n_ok + n_fail) / dt if dt else 0
            left = len(todo) - (n_ok + n_fail)
            print(f"[link-run] {n_ok + n_fail:,}/{len(todo):,}  ok {n_ok:,}  falhas {n_fail:,}  "
                  f"{rate:.1f}/s  ETA {left / rate / 60 if rate else 0:.0f} min", flush=True)
    return {"version": version, "ok": n_ok, "failed": n_fail, "seconds": time.time() - t0}


def run(cfg: Config, relations_version: str, oll: OllamaConfig, template_path: Path,
        limit: int | None = None, order: str = "freq", min_n: int = 1, segment: int = 2000) -> dict:
    template = load_template(template_path)
    need = max_prompt_chars(template) // 2 + oll.num_predict + 64
    if oll.num_ctx < need:
        raise ConfigError(f"--num-ctx {oll.num_ctx} pequeno demais para o prompt de link: precisa de ~{need}")
    if not (entities_dir(cfg, relations_version) / "candidates.parquet").exists():
        raise ConfigError("candidates.parquet ausente; rode `link-prep` antes")
    return asyncio.run(_run(cfg, relations_version, oll, template, limit, order, min_n, segment))


# --- link-verify -------------------------------------------------------------
#
# Segunda pergunta, só para os tickers aceitos por evidência forte com nome
# diferente da listagem (ticker_check = 'evidence'). Medido nas 60 mais citadas
# desse grupo: ~87% certos, e os erros são o ticker da bolsa de origem que nos
# EUA é de outra empresa (Tesco -> TSCO = Tractor Supply, Roche -> ROG = Rogers).
# Sem listagem atual não há nome a comparar: esses ficam como estão.

# Ticker reutilizado depois do período dos preços: a listagem de hoje mostra
# outro título e o modelo, com razão, responde "no". FB.csv do FNSPID é o
# Facebook (2012-05 a 2020-07); desde 2022 o FB é um ETF da ProShares. Medido:
# sem isto o link-verify derrubava Facebook e Meta Platforms (~3 mil menções).
REUSED_TICKERS = {"FB": "Facebook, Inc. (renamed Meta Platforms in 2021; the symbol FB was later reassigned)"}

_FUNDISH = re.compile(r"(?i)etf|fund|spdr|ishares|powershares|trust")


def verify_exempt(variants: list[str], ticker: str, link: str | None, is_etf: bool) -> bool:
    """Casos em que a resposta do link-verify não vale (medido na run c49291db2953).

    - controladora (link 'parent'): o modelo rejeitava Sandoz -> NVS, Waymo ->
      GOOG, Burger King -> QSR, Mobileye -> INTC; das 35 rejeições, quase todas
      eram mapeamentos certos
    - ETF citado como fundo ou pelo símbolo: "Technology Select Sector SPDR Fund
      (XLK)" contra "State Street Technology Select Sector SPDR ETF" -> "no"
    """
    if link == "parent":
        return True
    return bool(is_etf) and (any(x.strip().upper() == ticker for x in variants)
                             or any(_FUNDISH.search(x) for x in variants))


VERIFY_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
                 "required": ["answer"]}


def verify_dir(cfg: Config, relations_version: str, version: str) -> Path:
    return runs_dir(cfg, relations_version, version) / "verify"


def _verify_todo(cfg: Config, relations_version: str, version: str) -> list[dict]:
    """Chaves do grupo 'evidence' cujo ticker tem nome na listagem atual."""
    d = entities_dir(cfg, relations_version)
    c = duckdb.connect()
    rows = c.execute(f"""
        SELECT em.key, em.variants, em.ticker, t.name, t.is_etf, em.llm_link
        FROM read_parquet('{d / 'entity_map.parquet'}') em
        JOIN read_parquet('{d / 'tickers.parquet'}') t USING (ticker)
        WHERE em.ticker_check = 'evidence' AND em.link_version = ?
        ORDER BY em.n DESC, em.key""", [version]).fetchall()
    c.close()
    out = []
    for k, v, t, nm, etf, lk in rows:
        if not nm or verify_exempt(v, t, lk, etf):
            continue
        out.append({"key": k, "variants": v, "ticker": t, "listed": REUSED_TICKERS.get(t, nm)})
    return out


def render_verify(template: str, row: dict) -> str:
    out = template
    for ph, val in (("{{NAME}}", row["variants"][0][:80]),
                    ("{{VARIANTS}}", "; ".join(x[:60] for x in row["variants"][1:]) or "(none)"),
                    ("{{TICKER}}", row["ticker"]), ("{{LISTED}}", row["listed"][:80])):
        out = out.replace(ph, val)
    return out


async def _verify(cfg, relations_version, version, oll, template, rows) -> dict:
    out = verify_dir(cfg, relations_version, version)
    out.mkdir(parents=True, exist_ok=True)
    tv = hashlib.sha256((template + oll.model).encode()).hexdigest()[:8]
    dst = out / f"verify-{tv}.parquet"
    async with OllamaClient(oll) as cl:
        h = await cl.health()
        if not h["has_model"]:
            raise ConfigError(f"modelo {oll.model} ausente no Ollama; `ollama pull` antes")
        res = await asyncio.gather(*(cl.generate(render_verify(template, r), schema=VERIFY_SCHEMA)
                                     for r in rows))
    recs = [{"key": r["key"], "ticker": r["ticker"],
             "answer": (g.parsed or {}).get("answer") if g.ok else None,
             "model": oll.model, "verify_version": tv} for r, g in zip(rows, res)]
    _write(recs, pa.schema([("key", pa.string()), ("ticker", pa.string()), ("answer", pa.string()),
                            ("model", pa.string()), ("verify_version", pa.string())]), dst)
    n_no = sum(r["answer"] == "no" for r in recs)
    n_fail = sum(r["answer"] is None for r in recs)
    print(f"[link-verify] {len(recs):,} tickers conferidos: {n_no:,} rejeitados, {n_fail:,} falhas")
    print(f"[link-verify] {dst}")
    return {"checked": len(recs), "rejected": n_no, "failed": n_fail, "path": str(dst)}


def verify(cfg: Config, relations_version: str, version: str, oll: OllamaConfig,
           template_path: Path, limit: int | None = None) -> dict:
    """Precisa do entity_map de um link-build anterior desta versão."""
    template = Path(template_path).read_text(encoding="utf-8")
    rows = _verify_todo(cfg, relations_version, version)[:limit]
    return asyncio.run(_verify(cfg, relations_version, version, oll, template, rows))


# --- link-build --------------------------------------------------------------

# Evidência forte: o ticker escolhido veio do próprio nome ou da listagem.
STRONG_EVIDENCE = ("the entity name contains", "same company name", "the name is this ticker")
# Palavras que não identificam uma empresa: sem isso "Bank of X" casaria com
# qualquer banco e o rótulo do candidato ("listed today as...") com tudo.
_STOP = set("""
and the inc corp company group holdings holding international global american national
united financial capital bank banks technologies technology systems industries energy
resources partners management fund funds trust etf shares index ishares spdr invesco
vanguard proshares select sector first new general china chinese usa north south east
west plc ltd limited corporation incorporated services service solutions communications
pharmaceuticals therapeutics health healthcare insurance investment investments asset
assets securities markets market equity stock stocks oil gas motor motors airlines air
petroleum mining gold
""".split())


def _label_names(label: str) -> list[str]:
    """Nome contra o qual conferir: o da listagem; o das notícias só sem listagem.

    O nome das notícias vem das marcações do FNSPID, que erram: artigos da
    Airbus são marcados AIR (nos EUA, AAR Corp.), e aceitar o apelido deixava
    "Airbus" -> AIR passar. Ticker reutilizado (FB) passa pela evidência forte.
    """
    if label.startswith("(name unknown"):
        return []
    m = re.match(r"listed today as (.*?)(?: \(ETF\))?; in these news", label)
    if m:
        return [m.group(1)]
    if label.startswith("("):
        return re.findall(r"'([^']+)'", label)
    return [re.sub(r" \(ETF\)$", "", label)]


def name_matches(variants: list[str], label: str) -> bool:
    """O nome da entidade e o do ticker têm algo distintivo em comum?

    Token >= 3 letras fora de _STOP ("goldman"), token >= 5 letras prefixo de
    outro ("quidel" ~ "quidelortho"), ou um nome contido no outro sem espaços
    ("walmart" em "walmartstores", "exxonmobil" == "exxon mobil").
    """
    names = _label_names(label)
    for v in variants:
        nv = norm(v)
        # Sigla = iniciais do nome: TSMC, IBM, UPS, AIG
        # (iniciais do nome cru: norm() tiraria o "Company" de TSMC)
        if re.fullmatch(r"[a-z]{3,6}", nv) and any(
                nv == "".join(w[0] for w in re.findall(r"[a-z0-9]+", nm.lower())
                              if w not in ("and", "of", "the"))[:len(nv)] for nm in names):
            return True
        tv = {t for t in nv.split() if len(t) >= 3 and t not in _STOP}
        jv = nv.replace(" ", "")
        for nm in names:
            nn = norm(nm)
            tn = {t for t in nn.split() if len(t) >= 3 and t not in _STOP}
            if tv & tn:
                return True
            if any(len(a) >= 5 and len(b) >= 5 and (a.startswith(b) or b.startswith(a))
                   for a in tv for b in tn):
                return True
            jn = nn.replace(" ", "")
            short, long_ = sorted((jv, jn), key=len)
            if len(short) >= 4 and short in long_:
                return True
    return False


def check_ticker(variants: list[str], candidates: list[dict], ticker: str | None) -> str | None:
    """Por que aceitar o ticker do modelo ('name' | 'evidence'), ou None.

    'name': o nome bate com o do ticker. 'evidence': o nome NÃO bate, mas o
    ticker veio do próprio nome ou da listagem -- Google -> GOOG (Alphabet),
    Facebook -> FB (hoje um ETF), e também Roche -> ROG (o ticker suíço; nos EUA
    é a Rogers Corp.). Estes são os que valem revisão manual.

    Medido no piloto de 400 entidades (qwen2.5:7b): o modelo acerta quase tudo
    nas citadas e erra na cauda, escolhendo por co-menção ("companies" -> GS,
    "Forte Capital" -> VOE, "contract drug manufacturers" -> TMO) ou pelo ticker
    estrangeiro que conhece ("Airbus" -> AIR, que nos EUA é a AAR Corp.). Só a
    co-menção nunca basta.
    """
    if not ticker:
        return None
    c = next((c for c in candidates if c["ticker"] == ticker), None)
    if c is None:
        return None
    if name_matches(variants, c["name"]):
        return "name"
    if c["evidence"].startswith(STRONG_EVIDENCE):
        return "evidence"
    return None


def build(cfg: Config, relations_version: str, version: str | None = None, views: bool = True) -> dict:
    d = entities_dir(cfg, relations_version)
    runs = sorted((d / "runs").glob("l=*")) if version is None else [d / "runs" / f"l={version}"]
    runs = [r for r in runs if any(r.glob("part-*.parquet"))]
    if not runs:
        raise ConfigError(f"nenhum resultado de link-run em {d / 'runs'}")
    if version is None and len(runs) > 1:
        raise ConfigError("mais de uma versão de link-run; passe --link-version: "
                          + ", ".join(r.name for r in runs))
    rd = runs[0]
    version = rd.name.split("=", 1)[1]
    src = _relations_path(cfg, relations_version)
    tmap, tlinked = d / "entity_map.parquet", d / "relations_linked.parquet"

    con = duckdb.connect()
    tk = ", ".join(f"'{k}'" for k in TICKER_KINDS)
    # Última resposta ok por chave (uma re-execução pode ter repetido a chave).
    con.execute(f"""CREATE TABLE llm AS
        SELECT * FROM read_parquet('{rd}/part-*.parquet', filename=true) WHERE ok
        QUALIFY row_number() OVER (PARTITION BY key ORDER BY filename DESC) = 1""")
    con.execute(f"""CREATE TABLE em AS
        SELECT c.key, c.n, c.variants, l.kind,
               -- ticker só para empresa/fundo: o modelo às vezes classifica
               -- "S&P 500" como índice e ainda assim escolhe um ETF
               CASE WHEN l.kind IN ({tk}) AND l.link IN ('issuer', 'parent') THEN l.ticker END AS llm_ticker,
               l.link AS llm_link, l.canonical,
               CASE WHEN l.key IS NULL THEN 'pending' ELSE 'llm' END AS method,
               c.candidates
        FROM read_parquet('{d / 'candidates.parquet'}') c LEFT JOIN llm l USING (key)""")
    # Checagem determinística do ticker do modelo, e entity_id: o ticker; sem
    # ticker, o nome canônico normalizado (FEDERAL_RESERVE).
    # Rejeições do link-verify (se rodou): o ticker 'evidence' que o modelo
    # disse não ser da entidade cai, salvo os casos isentos (verify_exempt).
    # Vale o arquivo mais recente.
    rejected: set[str] = set()
    vfiles = sorted((rd / "verify").glob("verify-*.parquet"), key=lambda p: p.stat().st_mtime)
    if vfiles:
        etf = dict(con.execute(f"SELECT ticker, is_etf FROM read_parquet('{d / 'tickers.parquet'}')").fetchall())
        rows = con.execute(f"""SELECT v.key, v.ticker, em.variants, em.llm_link
            FROM read_parquet('{vfiles[-1]}') v JOIN em USING (key) WHERE v.answer = 'no'""").fetchall()
        rejected = {k for k, t, var, lk in rows if not verify_exempt(var, t, lk, etf.get(t, False))}
        print(f"[link-build] link-verify: {len(rejected):,} tickers rejeitados "
              f"({len(rows) - len(rejected):,} isentos) -- {vfiles[-1].name}")
    ids = []
    for k, v, cands, lt, ll, can in con.execute(
            "SELECT key, variants, candidates, llm_ticker, llm_link, canonical FROM em").fetchall():
        why = check_ticker(v, cands or [], lt)
        if why == "evidence" and k in rejected:
            why = None
        t = lt if why else None
        ids.append({"key": k, "ticker": t, "link": ll if t else "none", "ticker_check": why,
                    "entity_id": t or slug(can or "") or slug(k)})
    con.register("ids_arrow", pa.Table.from_pylist(ids, schema=pa.schema([
        ("key", pa.string()), ("ticker", pa.string()), ("link", pa.string()),
        ("ticker_check", pa.string()), ("entity_id", pa.string())])))
    con.execute(f"""COPY (
        SELECT em.* EXCLUDE (candidates), ids.ticker, ids.link, ids.ticker_check, ids.entity_id,
               t.has_price, '{version}' AS link_version
        FROM em JOIN ids_arrow ids USING (key)
        LEFT JOIN read_parquet('{d / 'tickers.parquet'}') t ON t.ticker = ids.ticker
    ) TO '{tmap}' (FORMAT parquet, COMPRESSION zstd)""")
    con.execute(f"""COPY (
        SELECT r.*,
               ea.entity_id AS entity_a, ea.ticker AS ticker_a, ea.kind AS kind_a, ea.link AS link_a,
               eb.entity_id AS entity_b, eb.ticker AS ticker_b, eb.kind AS kind_b, eb.link AS link_b
        FROM read_parquet('{src}') r
        LEFT JOIN read_parquet('{d / 'names.parquet'}') na ON na.raw = r.agent_a
        LEFT JOIN read_parquet('{tmap}') ea ON ea.key = na.key
        LEFT JOIN read_parquet('{d / 'names.parquet'}') nb ON nb.raw = r.agent_b
        LEFT JOIN read_parquet('{tmap}') eb ON eb.key = nb.key
    ) TO '{tlinked}' (FORMAT parquet, COMPRESSION zstd)""")

    s = con.execute(f"""SELECT count(*), sum(n), count(*) FILTER (WHERE method = 'pending'),
                               sum(n) FILTER (WHERE ticker IS NOT NULL),
                               sum(n) FILTER (WHERE ticker IS NOT NULL AND has_price)
                        FROM read_parquet('{tmap}')""").fetchone()
    kinds = con.execute(f"""SELECT coalesce(kind, '(pendente)'), count(*), sum(n) FROM read_parquet('{tmap}')
                            GROUP BY 1 ORDER BY 3 DESC""").fetchall()
    r = con.execute(f"""SELECT count(*),
                               count(*) FILTER (WHERE ticker_a IS NOT NULL OR ticker_b IS NOT NULL),
                               count(*) FILTER (WHERE ticker_a IS NOT NULL AND ticker_b IS NOT NULL)
                        FROM read_parquet('{tlinked}') WHERE noise IS NULL""").fetchone()
    con.close()

    if views:
        db = duckdb.connect(str(cfg.db))
        db.execute("CREATE SCHEMA IF NOT EXISTS news")
        db.execute(f"CREATE OR REPLACE VIEW news.entity_map AS SELECT * FROM read_parquet('{tmap}')")
        db.execute(f"CREATE OR REPLACE VIEW news.tickers AS SELECT * FROM read_parquet('{d / 'tickers.parquet'}')")
        db.execute(f"CREATE OR REPLACE VIEW news.relations_linked AS SELECT * FROM read_parquet('{tlinked}')")
        db.close()

    keys, mentions, pending, m_tick, m_price = s
    print(f"[link-build] l={version}: {keys:,} entidades, {mentions:,} menções ({pending:,} chaves pendentes)")
    print(f"[link-build] menções com ticker: {m_tick or 0:,} ({100 * (m_tick or 0) / mentions:.1f}%), "
          f"com preço: {m_price or 0:,} ({100 * (m_price or 0) / mentions:.1f}%)")
    for k, nk, nm in kinds:
        print(f"  {k:13} {nk:9,} entidades  {nm:10,} menções")
    print(f"[link-build] relações limpas: {r[0]:,}; com ticker em algum lado {r[1]:,} "
          f"({100 * r[1] / max(r[0], 1):.1f}%), nos dois {r[2]:,} ({100 * r[2] / max(r[0], 1):.1f}%)")
    print(f"[link-build] {tmap}\n[link-build] {tlinked}")
    return {"version": version, "keys": keys, "pending": pending, "relations": r[0],
            "any_ticker": r[1], "both_tickers": r[2]}
