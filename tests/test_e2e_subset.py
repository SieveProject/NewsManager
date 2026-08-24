"""End-to-end test over a bounded slice of the live remote file.

Exercises every stage for real -- range fetch, boundary detection, parquet
write, dedup, validation, views -- while touching ~200 MB instead of 23 GB.
Requires network.

    python tests/test_e2e_subset.py [n_shards] [shard_mib]
"""

from __future__ import annotations

import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from newsmanager import config, curate, ingest, query, shard, source, validate  # noqa: E402


def main() -> int:
    n_shards = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    shard_mib = int(sys.argv[2]) if len(sys.argv) > 2 else 48

    base = config.load()
    root = Path(__file__).resolve().parent.parent / "data" / "_test"
    if root.exists():
        shutil.rmtree(root)
    cfg = replace(
        base,
        root=root,
        manifest=root / "manifest.json",
        raw=root / "raw",
        curated=root / "curated",
        marts=root / "marts",
        db=root / "test.duckdb",
        tmp=root / "tmp",
        shard_bytes=shard_mib * 1024 * 1024,
        workers=4,
        memory_limit="1GB",
    )
    cfg.ensure_dirs()

    print(f"=== probe ===")
    info = source.verify(cfg.url, cfg.expected_bytes, cfg.expected_etag)
    print(f"ok: {info.size:,} bytes, etag {info.etag[:16]}...")

    print(f"\n=== plan (first {n_shards} shards of {shard_mib} MiB) ===")
    t = time.time()
    starts = [source.fetch_range(cfg.url, 0, 65535).find(b"\n") + 1]
    for _ in range(n_shards):
        starts.append(shard.find_boundary(cfg, starts[-1] + cfg.shard_bytes))
    shards = [
        shard.Shard(index=i, start=s, end=starts[i + 1] - 1)
        for i, s in enumerate(starts[:-1])
    ]
    man = shard.Manifest(url=cfg.url, size=cfg.expected_bytes, etag=info.etag,
                         shard_bytes=cfg.shard_bytes, shards=shards)
    man.save(cfg.manifest)
    covered = sum(s.nbytes for s in shards)
    print(f"{len(shards)} shards, {covered/1e6:.0f} MB, boundaries found in {time.time()-t:.0f}s")
    for a, b in zip(shards, shards[1:]):
        assert a.end + 1 == b.start, f"gap between shard {a.index} and {b.index}"
    print("boundary contiguity: OK")

    print(f"\n=== ingest ===")
    stats = ingest.run_sharded(cfg, man)
    assert not stats.get("failed"), stats["failed"]
    assert stats["rows"] > 0
    parquet_bytes = sum(p.stat().st_size for p in cfg.raw.glob("*.parquet"))
    print(f"{stats['rows']:,} rows; parquet {parquet_bytes/1e6:.0f} MB "
          f"vs csv {covered/1e6:.0f} MB ({covered/parquet_bytes:.1f}x compression)")

    print(f"\n=== resume check (re-run should skip everything) ===")
    again = ingest.run_sharded(cfg, man)
    assert again["ingested"] == 0 and again["skipped"] == len(shards), again
    print("resume: OK, no shard re-fetched")

    print(f"\n=== validate raw ===")
    assert validate.run(cfg, "raw"), "raw validation failed"

    print(f"\n=== curate ===")
    curate.run(cfg)

    print(f"\n=== validate curated ===")
    ok = validate.run(cfg, "curated")

    print(f"\n=== serve ===")
    query.build(cfg)
    con = query.connect(cfg)
    print(con.execute("SELECT count(*) docs FROM news.documents").fetchall())
    print(con.execute(
        "SELECT symbol, n_mentions, first_seen, last_seen FROM news.coverage "
        "ORDER BY n_mentions DESC LIMIT 5").fetchall())
    print("multi-ticker docs:", con.execute(
        "SELECT count(*) FROM news.documents WHERE n_symbols > 1").fetchall())
    con.close()

    print(f"\n=== RESULT: {'PASS' if ok else 'FAIL (see checks above)'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
