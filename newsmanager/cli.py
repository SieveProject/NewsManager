"""Command-line entrypoint: `python -m newsmanager <stage>`."""

from __future__ import annotations

import argparse
import shutil
import sys

from . import config, curate, ingest, query, shard, source, validate
from .config import EXIT_CONFIG, ConfigError


def _fmt(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def cmd_probe(cfg, args) -> int:
    info = source.probe(cfg.url)
    print(f"url   {cfg.url}")
    print(f"size  {info.size:,} bytes ({_fmt(info.size)})")
    print(f"etag  {info.etag}")
    match = info.size == cfg.expected_bytes and (not cfg.expected_etag or info.etag == cfg.expected_etag)
    print(f"pinned match: {'yes' if match else 'NO - update config.toml [source]'}")

    free = shutil.disk_usage(cfg.root.parent).free
    print(f"\nfree disk at {cfg.root.parent}: {_fmt(free)}")
    # Text at ~4x zstd, minus dedup. Deliberately conservative.
    est = info.size * 0.30
    print(f"estimated curated footprint: ~{_fmt(est)}")
    if free < est * 1.6:
        print("WARNING: free space is tight relative to the estimate; see README sizing notes.")
    return 0 if match else 1


def cmd_plan(cfg, args) -> int:
    man = shard.plan(cfg, verify_source=not args.no_verify)
    man.save(cfg.manifest)
    sizes = [s.nbytes for s in man.shards]
    print(f"[plan] {len(man.shards)} shards -> {cfg.manifest}")
    print(f"[plan] shard bytes: min={_fmt(min(sizes))} max={_fmt(max(sizes))} total={_fmt(sum(sizes))}")
    print(f"[plan] peak temp disk during ingest: ~{_fmt(cfg.shard_bytes * cfg.workers)}")
    return 0


def cmd_ingest(cfg, args) -> int:
    if args.mode == "stream":
        stats = ingest.run_stream(cfg)
    else:
        if not cfg.manifest.exists():
            print(f"no manifest at {cfg.manifest}; run `nm plan` first", file=sys.stderr)
            return 1
        man = shard.Manifest.load(cfg.manifest)
        stats = ingest.run_sharded(cfg, man, resume=not args.no_resume)
    print(f"[ingest] {stats}")
    return 1 if stats.get("failed") else 0


def cmd_curate(cfg, args) -> int:
    curate.run(cfg)
    return 0


def cmd_validate(cfg, args) -> int:
    return 0 if validate.run(cfg, args.stage) else 1


def cmd_serve(cfg, args) -> int:
    db = query.build(cfg)
    print(f"[serve] views ready in {db}")
    if args.marts:
        query.build_marts(cfg)
    print(f"\nQuery it:\n  duckdb {db}\n  SELECT * FROM news.coverage ORDER BY n_mentions DESC LIMIT 10;")
    return 0


def cmd_all(cfg, args) -> int:
    for fn, a in (
        (cmd_probe, args),
        (cmd_plan, args),
        (cmd_ingest, args),
        (cmd_validate, argparse.Namespace(stage="raw")),
        (cmd_curate, args),
        (cmd_validate, argparse.Namespace(stage="curated")),
        (cmd_serve, args),
    ):
        rc = fn(cfg, a)
        if rc != 0:
            print(f"stage {fn.__name__} failed with {rc}", file=sys.stderr)
            return rc
    return 0


def cmd_reset(cfg, args) -> int:
    targets = {"raw": cfg.raw, "curated": cfg.curated, "marts": cfg.marts, "tmp": cfg.tmp}
    chosen = targets if args.what == "all" else {args.what: targets[args.what]}
    for name, path in chosen.items():
        if path.exists():
            shutil.rmtree(path)
            print(f"[reset] removed {name}: {path}")
    if args.what == "all":
        cfg.db.unlink(missing_ok=True)
        cfg.manifest.unlink(missing_ok=True)
        print("[reset] removed db and manifest")
    return 0


def _dispatch(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nm", description="FNSPID news pipeline (DuckDB, remote-first)")
    p.add_argument("-c", "--config", default=None, help="path to config.toml")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe", help="check the remote file identity and local capacity")

    sp = sub.add_parser("plan", help="compute record-safe shard boundaries")
    sp.add_argument("--no-verify", action="store_true", help="skip pinned size/etag check")

    si = sub.add_parser("ingest", help="fetch shards into the raw parquet layer")
    si.add_argument("--mode", choices=("sharded", "stream"), default="sharded")
    si.add_argument("--no-resume", action="store_true", help="re-ingest shards that already exist")

    sub.add_parser("curate", help="dedup, type and partition into the curated layer")

    sv = sub.add_parser("validate", help="run data-quality checks")
    sv.add_argument("--stage", choices=("all", "raw", "curated"), default="all")

    ss = sub.add_parser("serve", help="build the DuckDB view database")
    ss.add_argument("--marts", action="store_true", help="also materialise the gold marts")

    sa = sub.add_parser("all", help="probe -> plan -> ingest -> curate -> validate -> serve")
    sa.add_argument("--mode", choices=("sharded", "stream"), default="sharded")
    sa.add_argument("--no-resume", action="store_true")
    sa.add_argument("--no-verify", action="store_true")
    sa.add_argument("--marts", action="store_true")

    sr = sub.add_parser("reset", help="delete a layer so it can be rebuilt")
    sr.add_argument("what", choices=("all", "raw", "curated", "marts", "tmp"))

    args = p.parse_args(argv)
    cfg = config.load(args.config)
    cfg.ensure_dirs()

    return {
        "probe": cmd_probe,
        "plan": cmd_plan,
        "ingest": cmd_ingest,
        "curate": cmd_curate,
        "validate": cmd_validate,
        "serve": cmd_serve,
        "all": cmd_all,
        "reset": cmd_reset,
    }[args.cmd](cfg, args)


def main(argv: list[str] | None = None) -> int:
    """Same EXIT_CONFIG contract as the extraction CLI, so deploy scripts can
    treat "this will never work" differently from "try again"."""
    try:
        return _dispatch(argv)
    except ConfigError as e:
        print(f"\n[nm] erro de configuração: {e}", file=sys.stderr)
        return EXIT_CONFIG


if __name__ == "__main__":
    raise SystemExit(main())
