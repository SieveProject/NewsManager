"""Configuration loading and the pinned source schema."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

# The 12 source columns, in file order.
#
# Everything is read as VARCHAR on purpose. DuckDB's sniffer can infer these,
# but inference over a remote CSV means either trusting a small prefix sample
# or setting sample_size=-1, which streams all 23 GB just to pick types. Pinning
# the schema makes every shard parse identically and keeps a malformed row in
# one shard from changing another shard's column types. Casting happens once,
# explicitly, in the curate stage.
SOURCE_COLUMNS: dict[str, str] = {
    "Unnamed: 0": "VARCHAR",
    "Date": "VARCHAR",
    "Article_title": "VARCHAR",
    "Stock_symbol": "VARCHAR",
    "Url": "VARCHAR",
    "Publisher": "VARCHAR",
    "Author": "VARCHAR",
    "Article": "VARCHAR",
    "Lsa_summary": "VARCHAR",
    "Luhn_summary": "VARCHAR",
    "Textrank_summary": "VARCHAR",
    "Lexrank_summary": "VARCHAR",
}

SUMMARY_COLUMNS = ("Lsa_summary", "Luhn_summary", "Textrank_summary", "Lexrank_summary")

CSV_HEADER = ",".join(SOURCE_COLUMNS) + "\n"


@dataclass(frozen=True)
class Config:
    url: str
    expected_bytes: int
    expected_etag: str
    root: Path
    manifest: Path
    raw: Path
    curated: Path
    marts: Path
    db: Path
    tmp: Path
    shard_bytes: int
    boundary_probe_bytes: int
    workers: int
    memory_limit: str
    compression: str
    drop_summaries: bool
    partition_by: str
    min_article_chars: int

    def ensure_dirs(self) -> None:
        for p in (self.root, self.raw, self.curated, self.marts, self.tmp):
            p.mkdir(parents=True, exist_ok=True)

    @property
    def ingest_columns(self) -> dict[str, str]:
        """Source columns actually written to the raw layer."""
        if not self.drop_summaries:
            return dict(SOURCE_COLUMNS)
        return {k: v for k, v in SOURCE_COLUMNS.items() if k not in SUMMARY_COLUMNS}


def load(path: str | os.PathLike[str] | None = None) -> Config:
    cfg_path = Path(path or os.environ.get("NEWSMANAGER_CONFIG", "config.toml"))
    if not cfg_path.is_absolute():
        cfg_path = _repo_root() / cfg_path
    with cfg_path.open("rb") as fh:
        raw = tomllib.load(fh)

    base = cfg_path.parent
    src, paths, ing, cur = raw["source"], raw["paths"], raw["ingest"], raw["curate"]

    def p(key: str) -> Path:
        v = Path(paths[key])
        return v if v.is_absolute() else base / v

    return Config(
        url=src["url"],
        expected_bytes=int(src["expected_bytes"]),
        expected_etag=str(src["expected_etag"]),
        root=p("root"),
        manifest=p("manifest"),
        raw=p("raw"),
        curated=p("curated"),
        marts=p("marts"),
        db=p("db"),
        tmp=p("tmp"),
        shard_bytes=int(ing["shard_bytes"]),
        boundary_probe_bytes=int(ing["boundary_probe_bytes"]),
        workers=int(ing["workers"]),
        memory_limit=str(ing["memory_limit"]),
        compression=str(ing["compression"]),
        drop_summaries=bool(ing["drop_summaries"]),
        partition_by=str(cur["partition_by"]),
        min_article_chars=int(cur["min_article_chars"]),
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent
