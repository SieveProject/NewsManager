"""Bronze layer: shard bytes -> one Parquet file per shard.

Two modes:

  sharded (default)  Parallel and resumable. Each worker range-fetches one shard,
                     writes those bytes to a temp file, has DuckDB parse it, then
                     deletes the temp file. Peak temp disk is
                     shard_bytes * workers -- ~1.5 GB at the defaults, never 23 GB.
                     A crash costs one shard, not the whole run.

  stream             One DuckDB `read_csv` over the URL. No temp files at all,
                     but no resume: a drop at 90% loses the run. Measured at
                     ~7 MB/s single-stream, so roughly 55 minutes end to end.

Transformation here is deliberately nil -- this layer stays 1:1 with the CSV so
curation bugs can be fixed without re-downloading. Typing happens in curate.
"""

from __future__ import annotations

import multiprocessing
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb

from . import source
from .config import CSV_HEADER, Config
from .shard import Manifest, Shard


def usable_cpus() -> int:
    """CPUs this process may actually use, honouring container CPU quotas.

    os.cpu_count() and DuckDB's default both report the *host*: 256 on the
    rented 4090, whose container quota is ~30. DuckDB then opened 256 threads,
    each buffering long article text, and `validate` hit OutOfMemory at 4 GB
    -- 16 MB per thread. On a laptop the two numbers agree, so it never showed.
    """
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:  # cgroup v2: "<quota> <period>" or "max <period>"
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            n = min(n, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return n


# Each scan thread holds a whole decompressed row group. One raw shard is one
# row group whose Article column alone is ~172 MB (~350 MB with the summaries),
# so a thread needs well over that. Measured on the VM: validate at 4 GB OOMed
# with 30 threads; curate at 9 GB OOMed with 18.
_MIN_BYTES_PER_THREAD = 2**30


def heavy_memory_limit() -> str:
    """Budget for curate/units, which hold a whole year of article text.

    9 GB suits the 16 GB laptop this was built on. 2023 alone is ~6.3 GB of
    text, so on a big box raise it: NM_HEAVY_MEMORY=64GB.
    """
    return os.environ.get("NM_HEAVY_MEMORY", "9GB")

_UNITS = {"KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40}


def _bytes(limit: str) -> int:
    m = re.fullmatch(r"\s*([\d.]+)\s*([KMGT]i?B)\s*", limit, re.IGNORECASE)
    if not m:
        raise ValueError(f"unrecognised memory limit {limit!r}")
    return int(float(m.group(1)) * _UNITS[m.group(2).upper().replace("IB", "B")])


def connect(cfg: Config, *, memory_limit: str | None = None, threads: int | None = None) -> duckdb.DuckDBPyConnection:
    limit = memory_limit or cfg.memory_limit
    if threads is None:
        threads = max(1, min(usable_cpus(), _bytes(limit) // _MIN_BYTES_PER_THREAD))
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{limit}'")
    con.execute(f"SET threads={threads}")
    # The source is sorted by symbol and we re-sort during curate, so holding
    # rows back to preserve arrival order only costs memory on a 23 GB scan.
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{cfg.tmp / 'duckdb_spill'}'")
    return con


def _columns_clause(cfg: Config) -> str:
    inner = ", ".join(f"'{k}':'{v}'" for k, v in cfg.ingest_columns.items())
    return "{" + inner + "}"


def shard_path(cfg: Config, index: int) -> Path:
    return cfg.raw / f"part-{index:05d}.parquet"


def _done(cfg: Config, index: int) -> bool:
    """A shard counts as done only if its Parquet footer is readable.

    Size alone is not enough: a process killed mid-write leaves a plausible-looking
    but truncated file, which would then be skipped on resume and silently drop rows.
    """
    p = shard_path(cfg, index)
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        duckdb.connect().execute(f"SELECT 1 FROM read_parquet('{p}') LIMIT 1").fetchall()
        return True
    except Exception:
        p.unlink(missing_ok=True)
        return False


def ingest_shard(cfg: Config, shard: Shard) -> tuple[int, int, float]:
    """Fetch one shard and write it as Parquet. Returns (index, rows, seconds)."""
    t0 = time.time()
    out = shard_path(cfg, shard.index)
    tmp_csv = cfg.tmp / f"shard-{shard.index:05d}.csv"
    tmp_out = out.with_suffix(".parquet.partial")

    try:
        data = source.fetch_range(cfg.url, shard.start, shard.end)
        # Each shard is a standalone CSV: header + whole records only.
        with tmp_csv.open("wb") as fh:
            fh.write(CSV_HEADER.encode())
            fh.write(data)
        del data

        con = connect(cfg, threads=2)
        cols = ", ".join(f'"{c}"' for c in cfg.ingest_columns)
        con.execute(
            f"""
            COPY (
                SELECT {cols}, {shard.index} AS _shard
                FROM read_csv(
                    '{tmp_csv}',
                    header=true,
                    columns={_columns_clause(cfg)},
                    quote='"', escape='"', delim=',',
                    -- parallel=false: the parallel scanner splits the shard into
                    -- chunks and must guess whether each starts inside a quoted
                    -- field. Articles carry raw newlines, and it guessed wrong on
                    -- 23 of 87 shards ("Expected 12 columns, found 2"). The serial
                    -- scan read the same shards with identical row counts at the
                    -- same speed -- 6 shards already run in parallel above it.
                    -- strict_mode=true so a misaligned row errors instead of
                    -- landing in the raw layer.
                    strict_mode=true, ignore_errors=false, parallel=false
                )
            ) TO '{tmp_out}' (FORMAT parquet, COMPRESSION '{cfg.compression}', ROW_GROUP_SIZE 100000)
            """
        )
        rows = con.execute(f"SELECT count(*) FROM read_parquet('{tmp_out}')").fetchone()[0]
        con.close()
        # Publish atomically so a kill never leaves a half-file that resume trusts.
        os.replace(tmp_out, out)
        return shard.index, rows, time.time() - t0
    finally:
        tmp_csv.unlink(missing_ok=True)
        Path(tmp_out).unlink(missing_ok=True)


def _worker(args: tuple[Config, Shard]) -> tuple[int, int, float]:
    return ingest_shard(*args)


def run_sharded(cfg: Config, manifest: Manifest, *, resume: bool = True) -> dict:
    cfg.ensure_dirs()
    pending = [s for s in manifest.shards if not (resume and _done(cfg, s.index))]
    skipped = len(manifest.shards) - len(pending)
    print(f"[ingest] {len(manifest.shards)} shards, {skipped} already done, {len(pending)} pending")
    if not pending:
        return {"shards": len(manifest.shards), "ingested": 0, "skipped": skipped, "rows": 0}

    total_rows = 0
    done = 0
    t0 = time.time()
    failures: list[tuple[int, str]] = []

    # spawn, never fork. Linux defaults to fork, which copies the parent with
    # whatever locks its threads (DuckDB's pool among ~30) held at that instant;
    # children that inherit a held lock block on a futex forever. On the VM 5
    # of 6 workers hung silently a minute in. macOS already defaults to spawn,
    # which is why it never showed up locally.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=cfg.workers, mp_context=ctx) as pool:
        futures = {pool.submit(_worker, (cfg, s)): s for s in pending}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                idx, rows, secs = fut.result()
            except Exception as exc:  # noqa: BLE001 - collected, reported at end
                failures.append((s.index, str(exc)))
                print(f"[ingest] shard {s.index:05d} FAILED: {exc}")
                continue
            total_rows += rows
            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed else 0
            eta = (len(pending) - done) / rate if rate else 0
            print(
                f"[ingest] {done}/{len(pending)} shard={idx:05d} rows={rows:,} "
                f"{secs:.0f}s elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m"
            )

    if failures:
        print(f"[ingest] {len(failures)} shard(s) failed; re-run `nm ingest` to retry only those")
    return {
        "shards": len(manifest.shards),
        "ingested": done,
        "skipped": skipped,
        "rows": total_rows,
        "failed": failures,
    }


def run_stream(cfg: Config) -> dict:
    """Single-pass streaming ingest. No temp CSV, no resume."""
    cfg.ensure_dirs()
    source.verify(cfg.url, cfg.expected_bytes, cfg.expected_etag)
    out = cfg.raw / "part-stream.parquet"
    tmp_out = out.with_suffix(".parquet.partial")
    con = connect(cfg, memory_limit="8GB")
    con.execute("SET http_retries=8; SET http_retry_wait_ms=2000; SET http_timeout=600000")
    cols = ", ".join(f'"{c}"' for c in cfg.ingest_columns)
    t0 = time.time()
    con.execute(
        f"""
        COPY (
            SELECT {cols}, 0 AS _shard
            FROM read_csv('{cfg.url}', header=true, columns={_columns_clause(cfg)},
                          quote='"', escape='"', delim=',', strict_mode=false)
        ) TO '{tmp_out}' (FORMAT parquet, COMPRESSION '{cfg.compression}', ROW_GROUP_SIZE 100000)
        """
    )
    rows = con.execute(f"SELECT count(*) FROM read_parquet('{tmp_out}')").fetchone()[0]
    os.replace(tmp_out, out)
    return {"shards": 1, "ingested": 1, "skipped": 0, "rows": rows, "seconds": time.time() - t0}
