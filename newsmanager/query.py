"""Serving layer: a DuckDB database of views over the curated Parquet.

The .duckdb file holds views, not copies -- it stays a few hundred KB and never
duplicates the corpus. Rebuild it freely; it is derived state.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from .config import Config


def build(cfg: Config) -> Path:
    """(Re)create the database of views. Safe to call repeatedly."""
    cfg.ensure_dirs()
    docs = cfg.curated / "documents" / "*.parquet"
    ments = cfg.curated / "mentions" / "*.parquet"
    summs = cfg.curated / "summaries" / "*.parquet"

    if not list((cfg.curated / "documents").glob("*.parquet")):
        raise RuntimeError(f"no curated data in {cfg.curated}; run `nm curate` first")

    con = duckdb.connect(str(cfg.db))
    con.execute("CREATE SCHEMA IF NOT EXISTS news")

    con.execute(f"CREATE OR REPLACE VIEW news.documents AS SELECT * FROM read_parquet('{docs}')")
    con.execute(f"CREATE OR REPLACE VIEW news.mentions AS SELECT * FROM read_parquet('{ments}')")
    if list((cfg.curated / "summaries").glob("*.parquet")):
        con.execute(f"CREATE OR REPLACE VIEW news.summaries AS SELECT * FROM read_parquet('{summs}')")

    # The flat shape the source had, reconstructed on demand. Prefer the
    # normalised views; this exists so existing per-ticker code keeps working.
    con.execute(
        """
        CREATE OR REPLACE VIEW news.articles AS
        SELECT m.symbol, d.*
        FROM news.mentions m
        JOIN news.documents d USING (doc_id)
        """
    )

    # Per-symbol daily counts: the usual join key against a price panel.
    con.execute(
        """
        CREATE OR REPLACE VIEW news.daily_symbol_counts AS
        SELECT symbol,
               CAST(published_at AS DATE) AS dt,
               count(*)                   AS n_articles,
               count(DISTINCT doc_id)     AS n_documents
        FROM news.mentions
        WHERE published_at IS NOT NULL
        GROUP BY 1, 2
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW news.coverage AS
        SELECT symbol,
               count(*)                        AS n_mentions,
               min(published_at)               AS first_seen,
               max(published_at)               AS last_seen,
               count(DISTINCT CAST(published_at AS DATE)) AS active_days
        FROM news.mentions
        GROUP BY 1
        """
    )
    con.close()
    return cfg.db


def connect(cfg: Config, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open the serving database. Use read_only for concurrent analysis sessions."""
    if not cfg.db.exists():
        raise RuntimeError(f"{cfg.db} does not exist; run `nm serve` first")
    return duckdb.connect(str(cfg.db), read_only=read_only)


def build_marts(cfg: Config) -> dict:
    """Gold layer: small, fully-materialised tables for repeated analysis.

    Written as real Parquet rather than views because a thesis loop re-reads
    these constantly and they are small enough to fit in memory.
    """
    con = connect(cfg, read_only=False)
    cfg.marts.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, sql in {
        "daily_symbol_counts": "SELECT * FROM news.daily_symbol_counts",
        "coverage": "SELECT * FROM news.coverage",
        "document_index": (
            "SELECT doc_id, published_at, year, title, url, publisher, author, "
            "body_chars, is_stub, n_symbols FROM news.documents"
        ),
    }.items():
        path = cfg.marts / f"{name}.parquet"
        con.execute(f"COPY ({sql}) TO '{path}' (FORMAT parquet, COMPRESSION '{cfg.compression}')")
        out[name] = con.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
        print(f"[marts] {name}: {out[name]:,} rows -> {path}")
    con.close()
    return out
