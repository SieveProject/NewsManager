"""Command-line entrypoint: `python -m newsmanager <stage>`."""

from __future__ import annotations

import argparse
import shutil
import sys

from . import config, curate, ingest, prices, query, sectors, shard, source, universe, validate
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


def cmd_prices(cfg, args) -> int:
    prices.fetch(cfg)
    if not args.fetch_only:
        prices.build(cfg)
    return 0


def cmd_universe(cfg, args) -> int:
    kw = {k: v for k, v in vars(args).items()
          if k in universe.DEFAULTS and v is not None}
    if args.asof:
        con = query.connect(cfg)
        for r in universe.select(con, args.asof, **kw):
            print(f"{r[7]:4d}  {r[1]:6} eventos={r[2]:5,}  docs={r[3]:5,}  preço={r[5]:9.2f}  "
                  f"vol.fin.mediano=US$ {r[6] / 1e6:,.0f} mi")
        return 0
    universe.build(cfg, start=args.start, end=args.end, **kw)
    return 0


def cmd_sectors(cfg, args) -> int:
    sectors.build(cfg)
    return 0


def cmd_h1(cfg, args) -> int:
    from .portfolio import h1
    h1.run(cfg, n=args.n, news_days=args.news_days, half_life_days=args.half_life)
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

    spp = sub.add_parser("prices", help="baixa o zip de preços do FNSPID e monta o painel limpo")
    spp.add_argument("--fetch-only", action="store_true", help="só baixa e confere o zip")

    su = sub.add_parser("universe", help="universo point-in-time por rebalanceamento mensal")
    su.add_argument("--asof", default=None, help="mostra o universo de uma data em vez de montar o painel")
    su.add_argument("--start", default="2011-01-01")
    su.add_argument("--end", default="2023-12-31")
    su.add_argument("--n", type=int, default=None, help="ações por data (padrão 100)")
    su.add_argument("--insample-days", type=int, default=None)
    su.add_argument("--news-days", type=int, default=None)
    su.add_argument("--min-price", type=float, default=None)
    su.add_argument("--min-dollar-volume", type=float, default=None)

    sub.add_parser("sectors", help="SIC da SEC + indústrias Fama-French dos tickers do universo "
                                   '(exige NM_SEC_USER_AGENT="Nome e-mail")')

    sh = sub.add_parser("h1", help="H1: notícias preveem a correlação futura além da histórica?")
    sh.add_argument("--n", type=int, default=100)
    sh.add_argument("--news-days", type=int, default=365)
    sh.add_argument("--half-life", type=float, default=90.0)

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
        "prices": cmd_prices,
        "universe": cmd_universe,
        "sectors": cmd_sectors,
        "h1": cmd_h1,
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
