"""Build `extraction_units`: the exact set of texts to send to the LLM.

Every unit is one paid inference. Anything collapsed here is money not spent,
so this stage is cost control rather than tidiness.

Three reductions, in order of confidence:

1. `documents` is already one row per article (curate collapsed the
   one-row-per-ticker source layout). A story filed under five tickers is one
   unit, carrying all five symbols along for downstream joins.

2. `text_key` -- a hash of the whitespace/punctuation-normalised body -- collapses
   the same story republished under different URLs. Measured on a 19,320-article
   sample, normalisation found no duplicates that the exact body hash missed,
   so this is cheap insurance rather than a large win on its own.

3. Stubs and empty bodies are excluded. There is nothing to extract from 40
   characters of boilerplate, and they would still cost a request each.

How much total duplication exists corpus-wide is *not* reliably estimable from a
sample: duplicates of a story sit far apart in a symbol-sorted file, so sampled
windows systematically miss them. Two honest measurements from this corpus
disagree for that reason -- 4.3% over 60 thin windows spread across the file,
~22% over a contiguous 200k-row prefix of heavily co-mentioned A-tickers. This
stage reports the true figure once, over the whole corpus, and that number is
the one to trust.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..config import Config
from ..ingest import connect, heavy_memory_limit

# Normalisation for the near-duplicate key: fold case, collapse all whitespace,
# drop punctuation. Catches re-encodings and whitespace-mangled republications
# while staying an exact-match test -- no similarity threshold to tune, and no
# risk of collapsing two genuinely different articles.
_NORM = r"lower(regexp_replace(regexp_replace(body, '[^\w\s]', '', 'g'), '\s+', ' ', 'g'))"


def build(cfg: Config, *, min_chars: int | None = None, near_dedup: bool = True) -> dict:
    """Write data/curated/extraction_units/ -- one row per LLM call to make."""
    docs = cfg.curated / "documents" / "*.parquet"
    ments = cfg.curated / "mentions" / "*.parquet"
    if not list((cfg.curated / "documents").glob("*.parquet")):
        raise RuntimeError(f"no curated documents in {cfg.curated}; run `nm curate` first")

    min_chars = cfg.min_article_chars if min_chars is None else min_chars
    out_dir = cfg.curated / "extraction_units"
    out_dir.mkdir(parents=True, exist_ok=True)
    con = connect(cfg, memory_limit=heavy_memory_limit())

    n_docs = con.execute(f"SELECT count(*) FROM read_parquet('{docs}')").fetchone()[0]
    t0 = time.time()

    # Symbols are aggregated per document so a unit carries every ticker the
    # story was filed under -- the LLM sees the full ticker context in one call
    # instead of the article being sent once per ticker.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW doc_syms AS
        SELECT doc_id, list_sort(list(DISTINCT symbol)) AS symbols
        FROM read_parquet('{ments}')
        WHERE symbol IS NOT NULL
        GROUP BY doc_id
        """
    )

    key_expr = f"md5({_NORM})" if near_dedup else "doc_id"

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW eligible AS
        SELECT d.doc_id, d.published_at, d.year, d.title, d.url, d.publisher,
               d.body, d.body_chars, COALESCE(s.symbols, []) AS symbols,
               {key_expr} AS text_key
        FROM read_parquet('{docs}') d
        LEFT JOIN doc_syms s USING (doc_id)
        WHERE d.body IS NOT NULL AND d.body_chars >= {min_chars}
        """
    )
    n_eligible = con.execute("SELECT count(*) FROM eligible").fetchone()[0]

    # One unit per distinct text. The representative doc is chosen
    # deterministically (earliest publication, then lowest doc_id) so the unit
    # list is stable across rebuilds -- workers can be restarted mid-run without
    # the partitioning shifting under them.
    con.execute(
        f"""
        COPY (
            SELECT
                text_key                                  AS unit_id,
                doc_id                                    AS repr_doc_id,
                published_at, year, title, url, publisher,
                body, body_chars,
                symbols,
                n_docs                                    AS n_source_docs,
                all_symbols
            FROM (
                SELECT *,
                    count(*)   OVER (PARTITION BY text_key) AS n_docs,
                    flatten(list(symbols) OVER (PARTITION BY text_key)) AS all_symbols,
                    row_number() OVER (
                        PARTITION BY text_key
                        ORDER BY published_at NULLS LAST, doc_id
                    ) AS rn
                FROM eligible
            )
            WHERE rn = 1
        ) TO '{out_dir}' (
            FORMAT parquet, COMPRESSION '{cfg.compression}',
            PARTITION_BY (year), OVERWRITE_OR_IGNORE 1
        )
        """
    )

    units_glob = str(out_dir / "**" / "*.parquet")
    n_units = con.execute(f"SELECT count(*) FROM read_parquet('{units_glob}')").fetchone()[0]
    stats = con.execute(
        f"""
        SELECT sum(body_chars)::BIGINT, avg(body_chars)::INT, max(body_chars)
        FROM read_parquet('{units_glob}')
        """
    ).fetchone()
    con.close()

    dropped = n_docs - n_eligible
    collapsed = n_eligible - n_units
    print(
        f"[units] {n_docs:,} documents\n"
        f"        -{dropped:,} below {min_chars} chars or empty\n"
        f"        -{collapsed:,} collapsed as duplicate text "
        f"({100*collapsed/n_eligible if n_eligible else 0:.1f}% of eligible)\n"
        f"        = {n_units:,} extraction units ({100*n_units/n_docs if n_docs else 0:.1f}% of documents)"
    )
    print(f"[units] total {stats[0]:,} chars, mean {stats[1]:,}/unit, max {stats[2]:,} [{time.time()-t0:.0f}s]")
    return {
        "documents": n_docs,
        "eligible": n_eligible,
        "units": n_units,
        "dropped_short": dropped,
        "collapsed_duplicate": collapsed,
        "total_chars": stats[0],
        "mean_chars": stats[1],
        "max_chars": stats[2],
    }


def units_glob(cfg: Config) -> str:
    return str(cfg.curated / "extraction_units" / "**" / "*.parquet")
