"""Silver layer: typed, deduplicated, partitioned.

The data model change that matters
----------------------------------
The source is one row per (article, ticker). Measured on the first 200k rows:
200,000 rows carry only 156,761 distinct URLs and 155,379 distinct article
bodies -- roughly 22% of rows are the *same* article repeated under each ticker
it mentions. Left flat, that inflates storage, and any per-article statistic
(sentiment means, publisher counts, embedding costs) silently double-counts
multi-ticker stories.

So the flat table is split into:

  documents  one row per article -- text, headline, url, publisher, timestamp
  mentions   one row per (doc_id, symbol) -- the many-to-many link
  summaries  one row per article -- the four extractive summaries, kept apart
             because they are derived and rarely queried alongside the body

Join `mentions` to `documents` for per-ticker work; query `documents` alone for
per-article work. Both are correct, which the flat table cannot offer.

Memory
------
Dedup is done one year at a time. Duplicates of a story always share its
publication date, so they never straddle a year boundary -- which makes
per-year processing exact, not approximate, and caps peak memory at the largest
single year rather than the whole corpus.
"""

from __future__ import annotations

import time
from pathlib import Path

from .config import SUMMARY_COLUMNS, Config
from .ingest import connect

# The Date column is 'YYYY-MM-DD HH:MM:SS UTC'. Trailing ' UTC' is stripped
# rather than parsed as a zone: it is constant across the file, and TRY_STRPTIME
# returns NULL instead of aborting the run on a malformed value.
_TS = "TRY_STRPTIME(regexp_replace(TRIM(\"Date\"), ' UTC$', ''), '%Y-%m-%d %H:%M:%S')"

# Identity: the URL when present, else a hash of the normalised body. URL is the
# stronger key -- the same story syndicated under several tickers keeps one URL,
# and whitespace-only text differences would defeat a pure content hash.
_DOC_ID = """
md5(
  CASE
    WHEN NULLIF(TRIM("Url"), '') IS NOT NULL THEN 'u:' || TRIM("Url")
    ELSE 'c:' || md5(regexp_replace(COALESCE("Article", ''), '\\s+', ' ', 'g'))
  END
)
"""


def _raw_glob(cfg: Config) -> str:
    return str(cfg.raw / "*.parquet")


def _years(con, cfg: Config) -> list[int]:
    rows = con.execute(
        f"""
        SELECT DISTINCT EXTRACT(year FROM {_TS})::INT AS y
        FROM read_parquet('{_raw_glob(cfg)}')
        WHERE {_TS} IS NOT NULL
        ORDER BY y
        """
    ).fetchall()
    return [r[0] for r in rows]


def run(cfg: Config) -> dict:
    """Build documents/, mentions/ and summaries/ partitioned by year."""
    cfg.ensure_dirs()
    if not list(Path(cfg.raw).glob("*.parquet")):
        raise RuntimeError(f"no raw shards in {cfg.raw}; run `nm ingest` first")

    con = connect(cfg, memory_limit="9GB")
    years = _years(con, cfg)
    if not years:
        raise RuntimeError("no parseable dates in raw layer")
    print(f"[curate] {len(years)} years: {years[0]}..{years[-1]}")

    docs_dir = cfg.curated / "documents"
    ment_dir = cfg.curated / "mentions"
    summ_dir = cfg.curated / "summaries"
    for d in (docs_dir, ment_dir, summ_dir):
        d.mkdir(parents=True, exist_ok=True)

    keep_summaries = not cfg.drop_summaries
    totals = {"documents": 0, "mentions": 0, "rows_in": 0}
    t0 = time.time()

    summary_select = ""
    if keep_summaries:
        summary_select = ", " + ", ".join(
            f"NULLIF(\"{c}\", '') AS {c.lower()}" for c in SUMMARY_COLUMNS
        )

    for year in years:
        yt = time.time()
        con.execute("DROP TABLE IF EXISTS stg")
        con.execute(
            f"""
            CREATE TEMP TABLE stg AS
            SELECT
                {_DOC_ID}                                    AS doc_id,
                {_TS}                                        AS published_at,
                NULLIF(TRIM("Stock_symbol"), '')             AS symbol,
                NULLIF(TRIM("Article_title"), '')            AS title,
                NULLIF(TRIM("Url"), '')                      AS url,
                NULLIF(TRIM("Publisher"), '')                AS publisher,
                NULLIF(TRIM("Author"), '')                   AS author,
                NULLIF("Article", '')                        AS body
                {summary_select}
            FROM read_parquet('{_raw_glob(cfg)}')
            WHERE EXTRACT(year FROM {_TS}) = {year}
            """
        )
        rows_in = con.execute("SELECT count(*) FROM stg").fetchone()[0]

        # One row per article. The picked representative is deterministic:
        # earliest timestamp, then lowest symbol, so a rebuild is byte-stable.
        con.execute(
            f"""
            COPY (
                SELECT
                    doc_id, published_at, {year} AS year,
                    title, url, publisher, author, body,
                    length(body)                          AS body_chars,
                    length(body) < {cfg.min_article_chars} AS is_stub,
                    n_symbols
                FROM (
                    SELECT *,
                        count(DISTINCT symbol) OVER (PARTITION BY doc_id) AS n_symbols,
                        row_number() OVER (
                            PARTITION BY doc_id
                            ORDER BY published_at NULLS LAST, symbol NULLS LAST
                        ) AS rn
                    FROM stg
                )
                WHERE rn = 1
            ) TO '{docs_dir / f"year={year}"}.parquet'
              (FORMAT parquet, COMPRESSION '{cfg.compression}', ROW_GROUP_SIZE 50000)
            """
        )

        con.execute(
            f"""
            COPY (
                SELECT DISTINCT doc_id, symbol, published_at, {year} AS year
                FROM stg WHERE symbol IS NOT NULL
            ) TO '{ment_dir / f"year={year}"}.parquet'
              (FORMAT parquet, COMPRESSION '{cfg.compression}', ROW_GROUP_SIZE 100000)
            """
        )

        if keep_summaries:
            cols = ", ".join(c.lower() for c in SUMMARY_COLUMNS)
            con.execute(
                f"""
                COPY (
                    SELECT doc_id, {cols} FROM (
                        SELECT doc_id, {cols},
                               row_number() OVER (PARTITION BY doc_id ORDER BY published_at NULLS LAST) rn
                        FROM stg
                    ) WHERE rn = 1
                ) TO '{summ_dir / f"year={year}"}.parquet'
                  (FORMAT parquet, COMPRESSION '{cfg.compression}', ROW_GROUP_SIZE 50000)
                """
            )

        ndocs = con.execute(f"SELECT count(*) FROM read_parquet('{docs_dir / f'year={year}'}.parquet')").fetchone()[0]
        nment = con.execute(f"SELECT count(*) FROM read_parquet('{ment_dir / f'year={year}'}.parquet')").fetchone()[0]
        totals["documents"] += ndocs
        totals["mentions"] += nment
        totals["rows_in"] += rows_in
        dedup = 100 * (1 - ndocs / rows_in) if rows_in else 0
        print(
            f"[curate] {year}: {rows_in:,} rows -> {ndocs:,} docs "
            f"({dedup:.1f}% dedup), {nment:,} mentions [{time.time()-yt:.0f}s]"
        )

    con.close()
    totals["seconds"] = time.time() - t0
    overall = 100 * (1 - totals["documents"] / totals["rows_in"]) if totals["rows_in"] else 0
    print(
        f"[curate] done in {totals['seconds']/60:.1f}m: "
        f"{totals['rows_in']:,} rows -> {totals['documents']:,} documents "
        f"({overall:.1f}% deduplicated), {totals['mentions']:,} mentions"
    )
    return totals
