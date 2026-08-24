"""HTTP access to the remote CSV: identity probing and ranged reads.

The source file is never fetched whole. Everything here works in ranges so the
23 GB CSV never exists on disk in one piece.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass

_UA = {"User-Agent": "NewsManager/1.0 (thesis data pipeline)"}
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class SourceInfo:
    size: int
    etag: str


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def probe(url: str) -> SourceInfo:
    """HEAD the URL and read the object's size/etag.

    Redirects are deliberately *not* followed. HuggingFace answers with a 302 to
    a CDN host, and the stable content identity lives on that 302 in
    `x-linked-size` / `x-linked-etag`. Following it yields the CDN's own etag,
    which is a different value and rotates independently of the content -- pinning
    against it would produce spurious "source changed" failures.
    """
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, method="HEAD", headers=_UA)
    try:
        with opener.open(req, timeout=60) as resp:
            headers = resp.headers
    except urllib.error.HTTPError as exc:
        if exc.code not in (301, 302, 303, 307, 308):
            raise
        headers = exc.headers

    size = headers.get("x-linked-size") or headers.get("content-length")
    etag = headers.get("x-linked-etag") or headers.get("etag") or ""
    if size is None:
        raise RuntimeError("could not determine source size from response headers")
    return SourceInfo(size=int(size), etag=etag.strip('"'))


def verify(url: str, expected_bytes: int, expected_etag: str) -> SourceInfo:
    """Fail loudly if upstream no longer matches the pinned identity.

    A silent upstream change would otherwise produce a dataset that is half old
    shards and half new ones, with nothing in the output indicating it.
    """
    info = probe(url)
    if info.size != expected_bytes:
        raise RuntimeError(
            f"source size changed: expected {expected_bytes:,} bytes, got {info.size:,}. "
            "Upstream was re-published. Update [source] in config.toml and rebuild "
            "from scratch (`nm reset`) rather than resuming onto stale shards."
        )
    if expected_etag and info.etag and info.etag != expected_etag:
        raise RuntimeError(
            f"source etag changed: expected {expected_etag}, got {info.etag}. "
            "Update [source] in config.toml and rebuild from scratch."
        )
    return info


def fetch_range(url: str, start: int, end: int, *, retries: int = 6, timeout: int = 300) -> bytes:
    """Fetch bytes [start, end] inclusive, retrying transient failures.

    A 23 GB ingest issues ~90 of these; over that many requests a transient 5xx
    or reset is expected, not exceptional, so retries are the default path.
    """
    if end < start:
        raise ValueError(f"empty range {start}-{end}")
    headers = dict(_UA)
    headers["Range"] = f"bytes={start}-{end}"
    want = end - start + 1
    last: Exception | None = None

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status not in (200, 206):
                    raise urllib.error.HTTPError(url, resp.status, "unexpected status", resp.headers, None)
                data = resp.read()
            # A 200 means the server ignored Range and is streaming the whole
            # object; treat as fatal rather than silently ingesting 23 GB.
            if resp.status == 200 and len(data) != want:
                raise RuntimeError("server ignored Range header; refusing full-object read")
            if len(data) != want:
                raise RuntimeError(f"short read: wanted {want} bytes, got {len(data)}")
            return data
        except Exception as exc:  # noqa: BLE001 - retried below, re-raised at end
            last = exc
            status = getattr(exc, "code", None)
            if status is not None and status not in _RETRY_STATUS:
                raise
            if attempt == retries - 1:
                break
            time.sleep(min(2**attempt, 30))

    raise RuntimeError(f"failed range {start}-{end} after {retries} attempts: {last}") from last
