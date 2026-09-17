"""Configuration loading and the pinned source schema."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# tomllib is 3.11+. Ubuntu 22.04 LTS -- the most common rented-GPU image --
# ships 3.10, where this module cannot even be imported. Fail with one readable
# line instead of a ModuleNotFoundError traceback on a machine billing by the
# hour, and say how to fix it.
if sys.version_info < (3, 11):  # pragma: no cover - depends on the interpreter
    raise SystemExit(
        f"newsmanager needs Python 3.11+ (tomllib); this interpreter is "
        f"{sys.version_info.major}.{sys.version_info.minor}.\n"
        "  Ubuntu 22.04: sudo add-apt-repository -y ppa:deadsnakes/ppa && "
        "sudo apt install -y python3.11 python3.11-venv\n"
        "  then re-run with: PY=python3.11 ./deploy/run_all.sh"
    )

import tomllib  # noqa: E402  -- guarded above


# Exit code reserved for failures a retry can never fix: a model that was never
# pulled, a context window too small for the prompt, a corpus that was never
# synced. `run_worker.sh` aborts on this instead of restarting, so it must stay
# distinct from an ordinary crash (1), which restarting *does* fix.
EXIT_CONFIG = 2


class ConfigError(Exception):
    """A misconfiguration. Deterministic -- the same command will fail again."""

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
