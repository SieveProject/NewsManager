"""Setor de cada ticker do universo -- etapa `sectors` (sem GPU).

Controle da H1 e alvo de referência do backtest (correlação constante por
setor, Elton e Gruber 1973). Fontes, nesta ordem:

  1. SEC `company_tickers.json`: ticker -> CIK. É um retrato de HOJE: 95 dos
     681 tickers do universo (ATVI, MYL, CERN, PXD...) já não estão nele, e um
     ticker pode ter sido reutilizado -- o nome da SEC tem de bater com o nosso.
  2. Pesquisa de entidades do EDGAR (a do campo de busca do site) pelo nome
     da empresa, para os que sobraram.
  3. API de submissões (data.sec.gov): código SIC e descrição.
  4. SIC -> indústrias de Fama-French (12 e 49), definições do site do Kenneth
     French (Siccodes12/49.zip).

O SIC é o ATUAL da empresa (a SEC não guarda histórico): mudanças de setor no
período ficam como limitação. A SEC exige User-Agent com contato:
NM_SEC_USER_AGENT="Nome e-mail" (não vai para o código, que é público).
Limite de 10 requisições/s: usamos ~8.
"""

from __future__ import annotations

import io
import json
import os
import re
import time
import zipfile
from pathlib import Path

import duckdb
import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from .config import Config, ConfigError
from .extract.link import norm

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes{k}.zip"
MIN_INTERVAL = 0.125  # 8 req/s
# Renomeadas depois do período: a busca do EDGAR só acha o nome ATUAL. É só um
# termo de busca -- o casamento ainda é conferido contra os nomes anteriores
# que a SEC registra (formerNames), então um palpite errado não passa.
RENAMED = {"RDS-A": "Shell plc", "SQ": "Block, Inc.", "ADS": "Bread Financial",
           "K": "Kellanova", "FB": "Meta Platforms", "RCI": "Rogers Communications"}

# Palavras que não identificam a empresa na comparação de nomes.
_GENERIC = set("""inc corp corporation co company companies ltd limited plc sa ag nv se group holdings
holding the and of class common stock shares ordinary adr ads american depositary new""".split())


def sec_dir(cfg: Config) -> Path:
    return cfg.root / "reference" / "sec"


def sectors_dir(cfg: Config) -> Path:
    return cfg.curated / "sectors"


class _Sec:
    def __init__(self, user_agent: str):
        self.c = httpx.Client(headers={"User-Agent": user_agent}, timeout=60, follow_redirects=True)
        self.t = 0.0

    def get(self, url: str, **params) -> httpx.Response:
        for attempt in range(5):
            wait = self.t + MIN_INTERVAL - time.time()
            if wait > 0:
                time.sleep(wait)
            self.t = time.time()
            r = self.c.get(url, params=params or None)
            if r.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            return r
        return r


def _tokens(s: str) -> set[str]:
    return {re.sub(r"(?<=..)s$", "", t) for t in norm(s).split() if t not in _GENERIC and len(t) > 1}


def _bare(s: str) -> str:
    """Sem espaços nem 's' final de possessivo: 'Amazon.com' = 'AMAZON COM',
    "McDonald's" = 'MCDONALDS'."""
    return "".join(re.sub(r"(?<=..)s$", "", t) for t in norm(s).split() if t not in _GENERIC)


def name_score(a: str, b: str, initials: bool = False) -> float:
    """1 se os nomes coincidem (inclusive sem espaços); 0,9 se um é prefixo
    longo do outro sem espaços ou se um é a sigla do outro (GE ~ General Electric); senão o
    Jaccard dos tokens distintivos."""
    if norm(a) == norm(b):
        return 1.0
    ba, bb = _bare(a), _bare(b)
    if ba and ba == bb:
        return 1.0
    # prefixo cobrindo >= 60% do mais longo ("walmartstores" ~ "walmart" não,
    # "pershingsquare" ~ "square" não, "wkkellogg" ~ "kellogg" não)
    short, long_ = sorted((ba, bb), key=len)
    if len(short) >= 4 and long_.startswith(short) and len(short) >= 0.6 * len(long_):
        return 0.9
    # sigla só quando o ticker já aponta para a empresa: na busca por nome,
    # "J.C. Penney" casava com "JE Cleantech"
    for x, y in (((a, b), (b, a)) if initials else ()):
        tx = norm(x).split()
        initials = "".join(w[0] for w in norm(y).split() if w not in _GENERIC)
        if tx and 2 <= len(tx[0]) <= 4 and tx[0] == initials[:len(tx[0])] and len(initials) == len(tx[0]):
            return 0.9
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def ff_industries(k: int) -> list[tuple[int, int, int, str]]:
    """(sic_lo, sic_hi, número, nome curto) das definições de Fama-French."""
    r = httpx.get(FF_URL.format(k=k), timeout=60, follow_redirects=True)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    text = z.read(z.namelist()[0]).decode("latin-1")
    out, cur = [], None
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s+(\w+)\s", line)
        if m and not re.match(r"\s*\d{4}-\d{4}", line):
            cur = (int(m.group(1)), m.group(2))
            continue
        m = re.match(r"\s*(\d{4})-(\d{4})", line)
        if m and cur:
            out.append((int(m.group(1)), int(m.group(2)), *cur))
    return out


def classify(sic: int | None, ranges: list[tuple[int, int, int, str]], other: tuple[int, str]) -> tuple[int, str]:
    if sic is None:
        return (None, None)
    for lo, hi, num, name in ranges:
        if lo <= sic <= hi:
            return (num, name)
    return other


def build(cfg: Config, user_agent: str | None = None, tickers: list[str] | None = None) -> dict:
    ua = user_agent or os.environ.get("NM_SEC_USER_AGENT")
    if not ua or "@" not in ua:
        raise ConfigError('a SEC exige contato: NM_SEC_USER_AGENT="Seu Nome seu@email"')
    con = duckdb.connect(str(cfg.db), read_only=True)
    if tickers is None:
        tickers = [t for (t,) in con.execute("SELECT DISTINCT ticker FROM news.universe ORDER BY 1").fetchall()]
    names = dict(con.execute("SELECT ticker, name FROM news.tickers WHERE ticker IN (SELECT unnest(?))",
                             [tickers]).fetchall())
    labels = dict(con.execute("SELECT node_id, label FROM news.graph_nodes WHERE node_id IN (SELECT unnest(?))",
                              [tickers]).fetchall())
    con.close()

    sec = _Sec(ua)
    d = sec_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    r = sec.get(TICKERS_URL)
    r.raise_for_status()
    (d / "company_tickers.json").write_bytes(r.content)
    current = {v["ticker"].upper().replace(".", "-"): (int(v["cik_str"]), v["title"]) for v in r.json().values()}

    cache_p = d / "submissions.jsonl"
    cache = {}
    if cache_p.exists():
        for line in cache_p.read_text().splitlines():
            x = json.loads(line)
            cache[x["cik"]] = x

    def submission(cik: int) -> dict | None:
        if cik in cache:
            return cache[cik]
        r = sec.get(SUBMISSIONS_URL.format(cik=cik))
        if r.status_code != 200:
            return None
        s = r.json()
        x = {"cik": cik, "name": s.get("name"), "sic": int(s["sic"]) if s.get("sic") else None,
             "sic_desc": s.get("sicDescription"), "tickers": s.get("tickers"),
             "former_names": [f.get("name") for f in s.get("formerNames") or []]}
        cache[cik] = x
        with cache_p.open("a") as f:
            f.write(json.dumps(x) + "\n")
        return x

    def best_of(cands: list[int], ours: list[str], initials: bool = False) -> tuple[dict | None, float]:
        best, score = None, 0.0
        for cik in dict.fromkeys(cands):
            s = submission(cik)
            if not s or s["sic"] is None:
                continue
            sc = max(name_score(o, n, initials) for o in ours for n in [s["name"] or "", *[f for f in s["former_names"] if f]])
            if sc > score:
                best, score = s, sc
        return best, score

    rows = []
    for k, t in enumerate(tickers):
        # O nome da listagem de HOJE pode ser de outra empresa com o mesmo
        # ticker (ADS: Adidas hoje, Alliance Data no FNSPID): vale o rótulo do grafo também.
        ours = list(dict.fromkeys(x for x in (names.get(t), labels.get(t)) if x)) or [t]
        hit, method, score = None, None, 0.0
        if t in current:
            s, sc = best_of([current[t][0]], ours, initials=True)
            if s and sc >= 0.34:
                hit, method, score = s, "ticker", sc
        if hit is None:
            cands = []
            queries = [RENAMED.get(t), *ours, *(" ".join(sorted(_tokens(o))[:2]) for o in ours),
                       *(norm(o).split()[0] for o in ours if norm(o))]
            for q in dict.fromkeys(q for q in queries if q):
                r = sec.get(SEARCH_URL, keysTyped=q)
                if r.status_code == 200:
                    cands += [int(h["_id"]) for h in r.json().get("hits", {}).get("hits", [])[:5]]
                if cands:
                    s, sc = best_of(cands, ours)
                    # 0,67: com 0,5 passavam Square -> Pershing Square, Rogers
                    # Communications -> Rogers Corp, Kellogg -> WK Kellogg
                    if s and sc >= 0.67:
                        hit, method, score = s, "name", sc
                        break
        rows.append({"ticker": t, "our_name": " | ".join(ours), "cik": hit["cik"] if hit else None,
                     "sec_name": hit["name"] if hit else None, "sic": hit["sic"] if hit else None,
                     "sic_desc": hit["sic_desc"] if hit else None, "method": method,
                     "name_score": round(score, 3)})
        if (k + 1) % 100 == 0:
            print(f"[sectors] {k + 1}/{len(tickers)}")

    ff12, ff49 = ff_industries(12), ff_industries(49)
    for x in rows:
        x["ff12"], x["ff12_name"] = classify(x["sic"], ff12, (12, "Other"))
        x["ff49"], x["ff49_name"] = classify(x["sic"], ff49, (49, "Other"))
        x["sic2"] = x["sic"] // 100 if x["sic"] is not None else None

    out = sectors_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out / "sectors.parquet", compression="zstd")
    db = duckdb.connect(str(cfg.db))
    db.execute(f"CREATE OR REPLACE VIEW news.sectors AS SELECT * FROM read_parquet('{out / 'sectors.parquet'}')")
    db.close()
    by = {}
    for x in rows:
        by[x["method"]] = by.get(x["method"], 0) + 1
    print(f"[sectors] {len(rows)} tickers: " + ", ".join(f"{m or 'sem setor'} {n}" for m, n in by.items()))
    print(f"[sectors] {out / 'sectors.parquet'}")
    return {"tickers": len(rows), **{str(m): n for m, n in by.items()}}
