"""Data-quality gate between stages.

Checks that would otherwise surface as a wrong thesis result rather than an
error: dropped shards, unparseable dates, dedup that collapsed too much or too
little, symbols that vanished between layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .ingest import connect
from .shard import Manifest


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    # Distribution heuristics (dedup rate, ticker count) depend on how much of
    # the corpus is in scope -- a partial or subset build trips them without
    # anything being wrong. They report but never fail the run. Only structural
    # invariants (uniqueness, referential integrity, coverage) are fatal.
    warn_only: bool = False

    @property
    def fatal(self) -> bool:
        return not self.ok and not self.warn_only

    def __str__(self) -> str:
        tag = "PASS" if self.ok else ("WARN" if self.warn_only else "FAIL")
        return f"[{tag}] {self.name}: {self.detail}"


def validate_raw(cfg: Config) -> list[Check]:
    checks: list[Check] = []
    con = connect(cfg, memory_limit="4GB")
    glob = str(cfg.raw / "*.parquet")
    files = sorted(Path(cfg.raw).glob("*.parquet"))

    if not files:
        return [Check("raw_present", False, f"no parquet in {cfg.raw}")]
    checks.append(Check("raw_present", True, f"{len(files)} shard files"))

    # Every planned shard must have produced a file. A missing index means a
    # silently dropped slice of the corpus.
    if cfg.manifest.exists():
        man = Manifest.load(cfg.manifest)
        have = {int(p.stem.split("-")[1]) for p in files if "-" in p.stem}
        missing = sorted({s.index for s in man.shards} - have)
        checks.append(
            Check(
                "all_shards_present",
                not missing,
                "all shards ingested" if not missing else f"{len(missing)} missing: {missing[:10]}",
            )
        )

    n = con.execute(f"SELECT count(*) FROM read_parquet('{glob}')").fetchone()[0]
    checks.append(Check("raw_rowcount", n > 0, f"{n:,} rows"))

    bad_date = con.execute(
        f"""SELECT count(*) FROM read_parquet('{glob}')
            WHERE TRY_STRPTIME(regexp_replace(TRIM("Date"), ' UTC$', ''), '%Y-%m-%d %H:%M:%S') IS NULL"""
    ).fetchone()[0]
    pct = 100 * bad_date / n if n else 0
    checks.append(Check("dates_parseable", pct < 1.0, f"{bad_date:,} unparseable ({pct:.3f}%)"))

    nosym = con.execute(
        f"""SELECT count(*) FROM read_parquet('{glob}') WHERE NULLIF(TRIM("Stock_symbol"),'') IS NULL"""
    ).fetchone()[0]
    checks.append(Check("symbol_present", nosym == 0, f"{nosym:,} rows without a symbol"))

    nobody = con.execute(
        f"""SELECT count(*) FROM read_parquet('{glob}') WHERE NULLIF("Article", '') IS NULL"""
    ).fetchone()[0]
    checks.append(Check("body_present", 100 * nobody / n < 5 if n else False, f"{nobody:,} rows with empty Article"))

    con.close()
    return checks


def validate_curated(cfg: Config) -> list[Check]:
    checks: list[Check] = []
    con = connect(cfg, memory_limit="4GB")
    docs = str(cfg.curated / "documents" / "*.parquet")
    ments = str(cfg.curated / "mentions" / "*.parquet")

    if not list((cfg.curated / "documents").glob("*.parquet")):
        return [Check("curated_present", False, "no curated documents; run `nm curate`")]

    ndocs = con.execute(f"SELECT count(*) FROM read_parquet('{docs}')").fetchone()[0]
    nment = con.execute(f"SELECT count(*) FROM read_parquet('{ments}')").fetchone()[0]
    checks.append(Check("curated_present", True, f"{ndocs:,} documents, {nment:,} mentions"))

    dupes = con.execute(
        f"SELECT count(*) FROM (SELECT doc_id FROM read_parquet('{docs}') GROUP BY doc_id HAVING count(*)>1)"
    ).fetchone()[0]
    checks.append(Check("doc_id_unique", dupes == 0, f"{dupes:,} duplicated doc_id"))

    # Every mention must resolve to a document, or per-ticker joins lose rows.
    orphans = con.execute(
        f"""SELECT count(*) FROM read_parquet('{ments}') m
            ANTI JOIN read_parquet('{docs}') d USING (doc_id)"""
    ).fetchone()[0]
    checks.append(Check("no_orphan_mentions", orphans == 0, f"{orphans:,} mentions without a document"))

    # Raw rows and mentions are both one-per-(article,ticker); they should agree
    # up to exact-duplicate rows removed by DISTINCT.
    if list(Path(cfg.raw).glob("*.parquet")):
        nraw = con.execute(f"SELECT count(*) FROM read_parquet('{cfg.raw / '*.parquet'}')").fetchone()[0]
        keep = 100 * nment / nraw if nraw else 0
        checks.append(
            Check("mention_coverage", keep > 90, f"{nment:,}/{nraw:,} raw rows retained as mentions ({keep:.1f}%)")
        )

    # On the full corpus this lands near 0.78; measured duplication over the
    # first 200k rows was ~22%. On a partial build it sits near 1.0 simply
    # because co-mentioned tickers fall outside the ingested slice.
    ratio = ndocs / nment if nment else 0
    checks.append(
        Check(
            "dedup_plausible",
            0.4 < ratio < 0.95,
            f"documents/mentions = {ratio:.3f} (full corpus ~0.78; near 1.0 means a partial build)",
            warn_only=True,
        )
    )

    rng = con.execute(f"SELECT min(published_at), max(published_at) FROM read_parquet('{docs}')").fetchone()
    checks.append(Check("date_range", rng[0] is not None, f"{rng[0]} .. {rng[1]}"))

    # The file is sorted by ticker, so a partial build covers an alphabetical
    # prefix of symbols rather than a random sample.
    nsym = con.execute(f"SELECT count(DISTINCT symbol) FROM read_parquet('{ments}')").fetchone()[0]
    checks.append(
        Check("symbol_coverage", nsym > 100, f"{nsym:,} distinct tickers (full corpus is thousands)", warn_only=True)
    )

    con.close()
    return checks


def run(cfg: Config, stage: str = "all") -> bool:
    checks: list[Check] = []
    if stage in ("all", "raw"):
        checks += validate_raw(cfg)
    if stage in ("all", "curated"):
        checks += validate_curated(cfg)
    for c in checks:
        print(c)
    fatal = [c for c in checks if c.fatal]
    warned = [c for c in checks if not c.ok and c.warn_only]
    passed = sum(1 for c in checks if c.ok)
    print(f"\n{passed}/{len(checks)} checks passed, {len(warned)} warning(s), {len(fatal)} failure(s)")
    return not fatal
