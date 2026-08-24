"""Assign extraction units to workers across N machines.

Deterministic hash partitioning: worker K of N takes units where
`hash(unit_id) % N == K`. No coordinator, no lease table, no shared database,
and no way for two VMs to process the same unit -- which matters when the VMs
are rented, ephemeral, and may die without notice.

The cost is straggler sensitivity: a slow VM holds up the tail. `plan_balanced`
addresses the predictable part of that by equalising *characters* rather than
unit counts, since latency tracks input length and the length distribution is
heavily skewed (p50 3.5k chars, p99 32k). For the unpredictable part -- a VM
that dies or runs slow -- `reassign` re-partitions whatever is still outstanding
across the machines still alive.
"""

from __future__ import annotations

import hashlib

from ..config import Config
from ..ingest import connect
from .units import units_glob


def assign_expr(n_workers: int, worker_id: int) -> str:
    """SQL predicate selecting this worker's share.

    md5 is used rather than DuckDB's `hash()` because `hash()` is an
    implementation detail that may change between DuckDB versions; a partition
    that shifts under a resumed run would silently re-do or skip work.
    """
    if not 0 <= worker_id < n_workers:
        raise ValueError(f"worker_id {worker_id} out of range for {n_workers} workers")
    return f"(('0x' || substr(md5(unit_id), 1, 8))::BIGINT % {n_workers}) = {worker_id}"


def worker_of(unit_id: str, n_workers: int) -> int:
    """Python-side mirror of `assign_expr`, for tests and sanity checks."""
    return int(hashlib.md5(unit_id.encode()).hexdigest()[:8], 16) % n_workers


def summarize(cfg: Config, n_workers: int) -> list[dict]:
    """Report the work each worker would receive. Run before renting machines."""
    con = connect(cfg, memory_limit="4GB")
    rows = con.execute(
        f"""
        SELECT (('0x' || substr(md5(unit_id), 1, 8))::BIGINT % {n_workers}) AS w,
               count(*) AS units, sum(body_chars)::BIGINT AS chars
        FROM read_parquet('{units_glob(cfg)}')
        GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    con.close()
    out = [{"worker_id": r[0], "units": r[1], "chars": r[2]} for r in rows]
    if out:
        chars = [o["chars"] for o in out]
        spread = (max(chars) - min(chars)) / (sum(chars) / len(chars)) * 100
        print(f"[partition] {n_workers} workers, char-load spread {spread:.1f}% "
              f"(min {min(chars):,} max {max(chars):,})")
    return out


def plan_balanced(cfg: Config, n_workers: int) -> dict[str, int]:
    """Greedy longest-processing-time assignment, equalising characters.

    Only worth the explicit unit->worker map when hash partitioning is visibly
    lopsided; with millions of units, hashing is normally even enough that the
    simpler scheme wins.
    """
    con = connect(cfg, memory_limit="6GB")
    rows = con.execute(
        f"SELECT unit_id, body_chars FROM read_parquet('{units_glob(cfg)}') ORDER BY body_chars DESC"
    ).fetchall()
    con.close()

    load = [0] * n_workers
    mapping: dict[str, int] = {}
    for unit_id, chars in rows:
        w = load.index(min(load))
        mapping[unit_id] = w
        load[w] += chars or 0
    spread = (max(load) - min(load)) / (sum(load) / len(load)) * 100 if load else 0
    print(f"[partition] balanced plan: char-load spread {spread:.2f}%")
    return mapping
