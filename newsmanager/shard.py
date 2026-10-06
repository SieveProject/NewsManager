"""Split the remote CSV into independently-parseable byte ranges.

Why this exists
---------------
`Article` values contain raw newlines inside quoted fields (verified against the
live file), so cutting the CSV on newline boundaries splits records in half.
Determining quote parity at an arbitrary offset normally requires scanning from
byte 0.

This file avoids that scan by anchoring on a structural signature that only
appears at a true record start:

    \\n<float index>,<YYYY-MM-DD HH:MM:SS ...>,

Both leading fields are unquoted, so a match cannot occur inside a quoted field
unless the article text itself contains a newline followed by a bare float,
comma, and full ISO timestamp. Validated against three 6 MB windows sampled at
2 GB, 8 GB and 15 GB: 2,707 records parsed, zero ragged rows.

Boundaries are found by range-fetching a small probe window near each target
offset, so planning touches ~180 MB rather than 23 GB.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from . import source
from .config import Config

# Anchored on a newline so the match start is always the byte before a record.
#
# The index field is optional: from ~17.7 GB to the end (the last 21 of 87
# shards) it is empty and records start with ",<timestamp>,". The original
# validation sampled 2/8/15 GB and never saw that tail; requiring a float there
# found no boundary at all past 17.7 GB. Strict-mode parses of shards on both
# sides of the change (65, 66, 75, 86) gave zero misaligned rows.
RECORD_ANCHOR = re.compile(rb"\n(?:\d+\.\d+)?,\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[^,\n]{0,12},")


@dataclass
class Shard:
    index: int
    start: int  # inclusive, first byte of a record
    end: int  # inclusive, last byte before the next record starts

    @property
    def nbytes(self) -> int:
        return self.end - self.start + 1


@dataclass
class Manifest:
    url: str
    size: int
    etag: str
    shard_bytes: int
    shards: list[Shard]

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        d = json.loads(Path(path).read_text())
        d["shards"] = [Shard(**s) for s in d["shards"]]
        return cls(**d)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())


def find_boundary(cfg: Config, target: int) -> int:
    """Return the offset of the first record starting at or after `target`.

    Widens the probe window on miss; a single record can exceed the default
    window (long wire articles run past 100 KB).
    """
    window = cfg.boundary_probe_bytes
    for _ in range(6):
        end = min(target + window - 1, cfg.expected_bytes - 1)
        if end < target:
            return cfg.expected_bytes
        data = source.fetch_range(cfg.url, target, end)
        m = RECORD_ANCHOR.search(data)
        if m:
            # +1 skips the anchoring newline, landing on the record's first byte.
            return target + m.start() + 1
        if end == cfg.expected_bytes - 1:
            return cfg.expected_bytes
        window *= 2
    raise RuntimeError(
        f"no record boundary within {window:,} bytes of offset {target:,}; "
        "the file layout may differ from the validated format"
    )


def plan(cfg: Config, *, verify_source: bool = True) -> Manifest:
    """Build the shard manifest. One HTTP probe per shard, no full scan."""
    if verify_source:
        info = source.verify(cfg.url, cfg.expected_bytes, cfg.expected_etag)
    else:
        info = source.probe(cfg.url)

    size = cfg.expected_bytes
    # The first record starts right after the header line.
    first = source.fetch_range(cfg.url, 0, min(65535, size - 1))
    nl = first.find(b"\n")
    if nl == -1:
        raise RuntimeError("no header line found in first 64 KB")
    starts = [nl + 1]

    target = starts[0] + cfg.shard_bytes
    while target < size:
        b = find_boundary(cfg, target)
        if b >= size:
            break
        # Guard against a boundary landing at or before the previous one, which
        # would produce an empty or overlapping shard.
        if b <= starts[-1]:
            target = starts[-1] + cfg.shard_bytes
            continue
        starts.append(b)
        target = b + cfg.shard_bytes

    shards = [
        Shard(index=i, start=s, end=(starts[i + 1] - 1 if i + 1 < len(starts) else size - 1))
        for i, s in enumerate(starts)
    ]
    _validate(shards, size, starts[0])
    return Manifest(url=cfg.url, size=size, etag=info.etag, shard_bytes=cfg.shard_bytes, shards=shards)


def _validate(shards: list[Shard], size: int, body_start: int) -> None:
    """Every body byte must be covered exactly once, with no gaps or overlaps."""
    if not shards:
        raise RuntimeError("planner produced no shards")
    if shards[0].start != body_start:
        raise RuntimeError("first shard does not start at the first record")
    if shards[-1].end != size - 1:
        raise RuntimeError("last shard does not reach end of file")
    for a, b in zip(shards, shards[1:]):
        if a.end + 1 != b.start:
            raise RuntimeError(f"shard {a.index} ends at {a.end}, shard {b.index} starts at {b.start}")
    covered = sum(s.nbytes for s in shards)
    if covered != size - body_start:
        raise RuntimeError(f"coverage mismatch: {covered:,} vs {size - body_start:,}")
