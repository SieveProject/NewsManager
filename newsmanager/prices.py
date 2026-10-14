"""Painel de preços do FNSPID -- etapa `prices` (sem GPU).

    fetch   baixa Stock_price/full_history.zip (590 MB, retomável) e confere o
            sha256 publicado pelo Hugging Face. O zip é a camada bruta: nunca é
            extraído em disco (2,9 GB de CSV), a limpeza lê membro a membro.
    build   zip -> data/curated/prices/{daily,quality,calendar}.parquet + views.

O que o arquivo tem (medido em 2026-10-14, 7.693 tickers, 29,7 milhões de linhas):

  - Duas ordens de coluna (`date,volume,open,...` em 5.705 arquivos,
    `date,open,...,volume` em 1.988): lido pelo cabeçalho.
  - COLETAS EMENDADAS (a principal descrita abaixo; ao todo 4 datas, ver
    SPLICE_DATES: em 2015-01-02 a AT&T "sobe" 95%, em 2020-04-02 a NEE "cai" 76%). 3.232 tickers terminam em 2020 (2.351 em
    2020-07-02); os 4.459 que vão a 2023-12-28 são o lote antigo até
    2020-07-02 colado a um lote novo a partir de 2020-07-06. Cada lote é
    ajustado (desdobramento + dividendos) só pelos eventos até a SUA coleta,
    então o retorno do dia da emenda é lixo: AAPL -74,9% (4:1), GOOG -94,9%
    (20:1), e também KO -9,5%, XOM -15,5%, PFE -17% (dividendos pós-2020). Dentro
    de cada lote os retornos estão certos (escala constante não muda retorno).
    Varredura da dispersão cross-section: 2020-07-06 é a única data anômala.
  - Fim de série em massa na mesma data (2020-07-02, 2020-06-19, 2020-04-01...)
    é fim de coleta, não deslistagem: `truncated` em quality.
  - Picos que se revertem no dia seguinte (erros de cotação, quase só em
    small caps: 11,8 mil dias em 529 tickers).
  - Grandes quedas reais ficam: AAPL -52% em 2000-09-29, NFLX -41% em 2004.

open/high/low ficam só no zip (camada bruta). `ret` é o retorno simples do `adj_close` entre pregões CONSECUTIVOS do
calendário, ou NULL com o motivo em `flag`. `ret_raw` guarda o valor bruto.
"""

from __future__ import annotations

import hashlib
import io
import time
import zipfile
from pathlib import Path

import duckdb
import httpx
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .config import Config, ConfigError

# Notícias úteis começam em 2010-2011; janelas in-sample de até 2 anos.
START = "2005-01-01"
# Dias de emenda entre coletas (ver docstring). O retorno destes dias é
# inválido para todo ticker. Achados pelo fator de ajuste f = adj_close/close,
# que numa série consistente só muda em data ex-dividendo e pouco: contagem de
# tickers com |f_t/f_t-1 - 1| > 5% no dia --
#   2020-07-06: 1.918   2015-01-02: 481   2020-04-02: 151   2020-07-02: 100
# e o resto do histórico <= 60 (ex-dividendos de fundos de distribuição mensal).
# O build recalcula a contagem e avisa se outra data passar de SPLICE_WARN.
SPLICE_DATES = ("2015-01-02", "2020-04-02", "2020-07-02", "2020-07-06")
SPLICE_WARN = 90
# Pico: |r_t| >= SPIKE_ABS e o dia seguinte desfaz quase tudo,
# |(1+r_t)(1+r_t+1) - 1| <= SPIKE_REVERT. Com 0,25 a regra pegava reversões
# REAIS (GME 2021-03-24, PCG 2018-11-15); os erros de cotação são de 15x a 40x
# (ACB 2006-2016 intercala dois papéis com o mesmo ticker).
SPIKE_ABS = 0.5
SPIKE_REVERT = 0.05
# Pregão = dia útil com ao menos metade dos tickers da vizinhança (±15 pregões).
CALENDAR_MIN_FRAC = 0.5
# Fim de coleta: data em que ao menos tantos tickers terminam juntos.
BATCH_END_MIN_TICKERS = 50
MEMORY_LIMIT = "4GB"
MAX_SPILL = "3GB"

PRICE_ZIP_URL = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip"
# Conferido em 2026-10-14. O sha256 é o oid LFS, que o Hugging Face devolve em
# `x-linked-etag` no HEAD SEM seguir o redirect. O `etag` depois do redirect é
# o hash xet do CDN (7550...), outra coisa. Se o upstream republicar, o fetch
# falha em vez de misturar duas versões.
EXPECTED_BYTES = 589_525_596
EXPECTED_SHA256 = "03da4fce7ebea90d5715ba3501773d410ae663b617027b338ef000a9955dab91"


def raw_dir(cfg: Config) -> Path:
    return cfg.raw / "prices"


def zip_path(cfg: Config) -> Path:
    return raw_dir(cfg) / "full_history.zip"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(cfg: Config) -> Path:
    """Baixa o zip uma vez; retoma um download interrompido (.part + Range)."""
    dst = zip_path(cfg)
    ok = dst.with_suffix(".zip.sha256")
    if dst.exists() and ok.exists() and ok.read_text().strip() == EXPECTED_SHA256:
        print(f"[prices] {dst} já baixado e conferido")
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(".zip.part")
    if dst.exists() and not part.exists():
        dst.rename(part)  # baixado mas não conferido: confere abaixo
    with httpx.Client(follow_redirects=True, timeout=120) as c:
        head = c.head(PRICE_ZIP_URL, follow_redirects=False)
        size = int(head.headers.get("x-linked-size") or c.head(PRICE_ZIP_URL).headers["content-length"])
        etag = head.headers.get("x-linked-etag", "").strip('"')
        if size != EXPECTED_BYTES or (etag and etag != EXPECTED_SHA256):
            raise ConfigError(f"o zip de preços mudou no Hugging Face ({size:,} bytes, etag {etag}); "
                              "atualize EXPECTED_* em newsmanager/prices.py conscientemente")
        have = part.stat().st_size if part.exists() else 0
        if have < size:
            print(f"[prices] baixando {size / 1e6:,.0f} MB (já tenho {have / 1e6:,.0f} MB)")
            with c.stream("GET", PRICE_ZIP_URL, headers={"Range": f"bytes={have}-"}) as r:
                r.raise_for_status()
                if have and r.status_code != 206:
                    raise ConfigError("o servidor ignorou o Range; apague o .part e rode de novo")
                with part.open("ab") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
    got = _sha256(part)
    if got != EXPECTED_SHA256:
        part.unlink()
        raise ConfigError(f"sha256 não confere ({got}); arquivo apagado, rode de novo")
    part.rename(dst)
    ok.write_text(got + "\n")
    print(f"[prices] {dst} ({size / 1e6:,.0f} MB, sha256 ok)")
    return dst


def prices_dir(cfg: Config) -> Path:
    return cfg.curated / "prices"


_TYPES = {"date": pa.string(), "open": pa.float64(), "high": pa.float64(), "low": pa.float64(),
          "close": pa.float64(), "adj close": pa.float64(), "volume": pa.float64()}
_OUT = ["date", "open", "high", "low", "close", "adj close", "volume"]


def _stage(cfg: Config, dst: Path) -> int:
    """zip -> um parquet de preparação (só >= START), lendo cada CSV pelo cabeçalho."""
    z = zipfile.ZipFile(zip_path(cfg))
    names = [n for n in z.namelist() if n.startswith("full_history/") and n.endswith(".csv")]
    w, n_rows = None, 0
    for name in names:
        t = pacsv.read_csv(io.BytesIO(z.read(name)),
                           convert_options=pacsv.ConvertOptions(column_types=_TYPES, include_columns=_OUT))
        t = t.select(_OUT).rename_columns(["date", "open", "high", "low", "close", "adj_close", "volume"])
        t = t.filter(pc.greater_equal(t["date"], START))
        t = t.append_column("ticker", pa.array([Path(name).stem.upper()] * len(t), pa.string()))
        if w is None:
            w = pq.ParquetWriter(dst, t.schema, compression="zstd")
        w.write_table(t)
        n_rows += len(t)
    if w is not None:
        w.close()
    return n_rows


def build(cfg: Config, views: bool = True) -> dict:
    zp = zip_path(cfg)
    if not zp.exists():
        raise ConfigError(f"{zp} não existe; rode `python -m newsmanager prices --fetch` antes")
    out = prices_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    cfg.tmp.mkdir(parents=True, exist_ok=True)
    stage = cfg.tmp / "prices_stage.parquet"
    t0 = time.time()
    n_stage = _stage(cfg, stage)
    print(f"[prices] {n_stage:,} linhas >= {START} lidas do zip ({time.time() - t0:.0f}s)")

    # Teto de memória e de transbordo: sem ele o DuckDB encheu o disco do Mac
    # (21,5 milhões de linhas em tabelas intermediárias). Agora as etapas são
    # CTEs de uma consulta só, gravada direto em Parquet, e o transbordo para em
    # MAX_SPILL em vez de consumir o disco inteiro.
    con = duckdb.connect(config={"memory_limit": MEMORY_LIMIT, "temp_directory": str(cfg.tmp / "duckdb"),
                                 "max_temp_directory_size": MAX_SPILL, "preserve_insertion_order": False})
    con.execute("SET enable_progress_bar = false")
    src = f"""(SELECT ticker, CAST(date AS DATE) AS date, open, high, low, close, adj_close, volume
              FROM read_parquet('{stage}'))"""
    bad = con.execute(f"""SELECT count(*) FILTER (WHERE close IS NULL OR adj_close IS NULL OR close <= 0 OR adj_close <= 0),
                                 count(*) FILTER (WHERE dayofweek(date) IN (0, 6))
                          FROM {src}""").fetchone()

    # Calendário de pregões: dia útil com cobertura >= metade da vizinhança.
    con.execute(f"""CREATE TABLE calendar AS
        WITH d AS (SELECT date, count(*) AS n FROM {src}
                   WHERE dayofweek(date) NOT IN (0, 6) AND close > 0 AND adj_close > 0 GROUP BY 1),
             m AS (SELECT date, n, median(n) OVER (ORDER BY date ROWS BETWEEN 15 PRECEDING AND 15 FOLLOWING) AS med FROM d)
        SELECT date, n AS n_tickers, row_number() OVER (ORDER BY date) AS cal_i
        FROM m WHERE n >= {CALENDAR_MIN_FRAC} * med ORDER BY date""")

    splice = ", ".join(f"DATE '{d}'" for d in SPLICE_DATES)
    daily_tmp = out / "daily.parquet.tmp"
    con.execute(f"""COPY (
        WITH p1 AS (
            SELECT r.*, c.cal_i, lag(r.adj_close) OVER w AS prev_adj, c.cal_i - lag(c.cal_i) OVER w - 1 AS gap
            FROM {src} r JOIN calendar c USING (date)
            WHERE r.close > 0 AND r.adj_close > 0
            WINDOW w AS (PARTITION BY r.ticker ORDER BY r.date)),
        p2 AS (
            SELECT *, adj_close / prev_adj - 1 AS ret_raw,
                   CASE WHEN prev_adj IS NULL THEN 'first'
                        WHEN gap > 0 THEN 'gap'
                        WHEN date IN ({splice}) THEN 'splice'
                        WHEN volume = 0 AND adj_close = prev_adj THEN 'stale' END AS flag0
            FROM p1),
        -- pico e reversão: decidido com o retorno do pregão seguinte (consecutivo)
        p3 AS (
            SELECT *, lead(ret_raw) OVER w AS nxt, lead(gap) OVER w AS nxt_gap,
                      lag(ret_raw) OVER w AS prv, lag(flag0) OVER w AS prv_flag
            FROM p2 WINDOW w AS (PARTITION BY ticker ORDER BY date)),
        p4 AS (
            SELECT *, coalesce(flag0,
                     CASE WHEN abs(ret_raw) >= {SPIKE_ABS} AND nxt_gap = 0
                               AND abs((1 + ret_raw) * (1 + nxt) - 1) <= {SPIKE_REVERT} THEN 'spike'
                          WHEN prv_flag IS NULL AND abs(prv) >= {SPIKE_ABS} AND gap = 0
                               AND abs((1 + prv) * (1 + ret_raw) - 1) <= {SPIKE_REVERT} THEN 'spike_revert' END) AS flag
            FROM p3)
        -- open/high/low ficam só no zip: o painel curado tem de caber no Mac
        SELECT ticker, date, close, adj_close, volume, close * volume AS dollar_volume,
               ret_raw, CASE WHEN flag IS NULL THEN ret_raw END AS ret, flag,
               CASE WHEN date <= DATE '2020-07-02' THEN 'old' ELSE 'new' END AS batch
        FROM p4 ORDER BY ticker, date
    ) TO '{daily_tmp}' (FORMAT parquet, COMPRESSION zstd)""")
    daily_tmp.rename(out / "daily.parquet")
    # Guarda: emenda nova (datas fora de SPLICE_DATES com muitos saltos de f).
    extra = con.execute(f"""
        WITH f AS (SELECT date, adj_close / close AS f, lag(adj_close / close) OVER w AS pf, flag
                   FROM read_parquet('{out / 'daily.parquet'}') WINDOW w AS (PARTITION BY ticker ORDER BY date))
        SELECT date, count(*) FROM f WHERE pf IS NOT NULL AND flag IS DISTINCT FROM 'gap'
               AND abs(f / pf - 1) > 0.05 AND date NOT IN ({splice})
        GROUP BY 1 HAVING count(*) >= {SPLICE_WARN} ORDER BY 2 DESC""").fetchall()
    if extra:
        print(f"[prices] AVISO: possíveis emendas fora de SPLICE_DATES: {extra[:5]}")
    stage.unlink(missing_ok=True)
    con.execute(f"CREATE VIEW daily AS SELECT * FROM read_parquet('{out / 'daily.parquet'}')")

    # Qualidade por ticker. Fim de coleta = data de fim compartilhada por muitos.
    con.execute(f"""CREATE TABLE quality AS
        WITH s AS (SELECT ticker, min(date) AS first_date, max(date) AS last_date, count(*) AS n_obs,
                          count(ret) AS n_ret,
                          count(*) FILTER (WHERE flag = 'gap') AS n_gap,
                          count(*) FILTER (WHERE flag IN ('spike', 'spike_revert')) AS n_spike,
                          count(*) FILTER (WHERE flag = 'splice') AS n_splice,
                          count(*) FILTER (WHERE flag = 'stale') AS n_stale,
                          max(abs(ret)) AS max_abs_ret,
                          median(close) AS median_close, median(dollar_volume) AS median_dollar_volume
                   FROM daily GROUP BY 1),
             ends AS (SELECT last_date FROM s GROUP BY 1
                      HAVING count(*) >= {BATCH_END_MIN_TICKERS} AND last_date < (SELECT max(date) FROM calendar)),
             span AS (SELECT s.ticker, count(c.date) AS n_cal FROM s
                      JOIN calendar c ON c.date BETWEEN s.first_date AND s.last_date GROUP BY 1)
        SELECT s.*, span.n_cal, s.n_obs / span.n_cal AS coverage,
               s.last_date IN (SELECT last_date FROM ends) AS truncated
        FROM s JOIN span USING (ticker) ORDER BY ticker""")

    for name in ("quality", "calendar"):
        tmp = out / f"{name}.parquet.tmp"
        con.execute(f"COPY {name} TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)")
        tmp.rename(out / f"{name}.parquet")
    st = con.execute("""SELECT count(*), count(DISTINCT ticker), count(ret), min(date), max(date) FROM daily""").fetchone()
    flags = con.execute("SELECT flag, count(*) FROM daily WHERE flag IS NOT NULL GROUP BY 1 ORDER BY 2 DESC").fetchall()
    q = con.execute("""SELECT count(*) FILTER (WHERE truncated), list(DISTINCT last_date) FILTER (WHERE truncated)
                       FROM quality""").fetchone()
    ncal = con.execute("SELECT count(*) FROM calendar").fetchone()[0]
    con.close()

    if views:
        register_views(cfg)
    print(f"[prices] descartadas: {bad[0]:,} linhas com preço nulo/<=0; {bad[1]:,} em fim de semana")
    print(f"[prices] {st[0]:,} linhas, {st[1]:,} tickers, {ncal:,} pregões ({st[3]} a {st[4]}); "
          f"{st[2]:,} retornos válidos")
    print("[prices] retornos invalidados: " + ", ".join(f"{f} {n:,}" for f, n in flags))
    print(f"[prices] {q[0]:,} tickers truncados por fim de coleta em {sorted(q[1] or [])}")
    print(f"[prices] {out} ({time.time() - t0:.0f}s)")
    return {"rows": st[0], "tickers": st[1], "returns": st[2], "truncated": q[0]}


def register_views(cfg: Config) -> None:
    out = prices_dir(cfg)
    db = duckdb.connect(str(cfg.db))
    db.execute("CREATE SCHEMA IF NOT EXISTS news")
    db.execute(f"CREATE OR REPLACE VIEW news.prices AS SELECT * FROM read_parquet('{out / 'daily.parquet'}')")
    db.execute("CREATE OR REPLACE VIEW news.returns AS SELECT ticker, date, ret FROM news.prices WHERE ret IS NOT NULL")
    db.execute(f"CREATE OR REPLACE VIEW news.price_quality AS SELECT * FROM read_parquet('{out / 'quality.parquet'}')")
    db.execute(f"CREATE OR REPLACE VIEW news.trading_calendar AS SELECT * FROM read_parquet('{out / 'calendar.parquet'}')")
    db.close()
