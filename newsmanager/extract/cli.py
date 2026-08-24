"""`python -m newsmanager.extract <stage>` -- the LLM extraction pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .. import config as cfgmod
from . import benchmark, collect, partition, units, worker
from .client import OllamaConfig
from .prompt import DEFAULT_MAX_CHARS, load as load_prompt


def _repo() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=os.environ.get("NM_MODEL", "deepseek-r1:14b"))
    p.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    p.add_argument("--concurrency", type=int, default=int(os.environ.get("NM_CONCURRENCY", "8")))
    p.add_argument("--num-ctx", type=int, default=int(os.environ.get("NM_NUM_CTX", "4096")))
    p.add_argument("--num-predict", type=int, default=1024)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--prompt", default=None, help="prompt template (default prompts/extraction.txt)")
    p.add_argument("--schema", default=None, help="JSON schema (default prompts/schema.json)")
    p.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    p.add_argument("--think", action="store_true",
                   help="liga o raciocínio (multiplica tokens de saída; desligado por padrão)")


def _prompt_from(args):
    pp = Path(args.prompt) if args.prompt else _repo() / "prompts" / "extraction.txt"
    sp = Path(args.schema) if args.schema else _repo() / "prompts" / "schema.json"
    pr = load_prompt(pp, sp, args.max_chars)

    # An article capped at max_chars is ~max_chars/4 tokens, plus the template
    # and room to generate. Silently exceeding num_ctx truncates the *article*
    # inside Ollama with no error, so this is checked rather than trusted.
    need = (args.max_chars + len(pr.template)) // 4 + args.num_predict + 256
    if args.num_ctx < need:
        raise SystemExit(
            f"--num-ctx {args.num_ctx} is too small for --max-chars {args.max_chars}: "
            f"need ~{need} tokens (article + prompt + {args.num_predict} generated). "
            f"Raise --num-ctx to {need} or lower --max-chars. Ollama would otherwise "
            f"silently drop the end of each article."
        )
    return pr


def _oll_from(args) -> OllamaConfig:
    return OllamaConfig(
        host=args.host, model=args.model, num_ctx=args.num_ctx,
        num_predict=args.num_predict, concurrency=args.concurrency, timeout_s=args.timeout,
        think=getattr(args, 'think', False),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nm-extract", description="LLM relation extraction over the news corpus")
    p.add_argument("-c", "--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("units", help="build the deduplicated list of LLM calls to make")
    u.add_argument("--min-chars", type=int, default=None)
    u.add_argument("--no-near-dedup", action="store_true")

    pa = sub.add_parser("partition", help="show per-worker load for N machines")
    pa.add_argument("-n", "--workers", type=int, required=True)

    b = sub.add_parser("bench", help="measure throughput on this machine")
    _add_model_args(b)
    b.add_argument("--n", type=int, default=40)

    sw = sub.add_parser("sweep", help="find the best concurrency for this GPU")
    _add_model_args(sw)
    sw.add_argument("--levels", default="1,2,4,8,16,32")
    sw.add_argument("--n", type=int, default=32)

    pj = sub.add_parser("project", help="extrapolate cost from a measured rate")
    pj.add_argument("--units-per-s", type=float, required=True)
    pj.add_argument("--vms", type=int, default=1)
    pj.add_argument("--usd-per-gpu-hour", type=float, default=0.40)

    r = sub.add_parser("run", help="run extraction for one worker")
    _add_model_args(r)
    r.add_argument("--worker-id", type=int, default=int(os.environ.get("NM_WORKER_ID", "0")))
    r.add_argument("--workers", type=int, default=int(os.environ.get("NM_WORKERS", "1")))
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--no-resume", action="store_true")
    r.add_argument("--checkpoint-every", type=int, default=200,
               help="registros por segmento parquet (o WAL protege o que ainda não foi gravado)")
    r.add_argument("--order", choices=("long_first", "short_first", "natural"), default="long_first")

    c = sub.add_parser("collect", help="merge worker shards into parquet + views")
    c.add_argument("--prompt-version", default=None)
    c.add_argument("--prompt", default=None)
    c.add_argument("--schema", default=None)
    c.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    c.add_argument("--no-views", action="store_true")

    args = p.parse_args(argv)
    cfg = cfgmod.load(args.config)
    cfg.ensure_dirs()

    if args.cmd == "units":
        units.build(cfg, min_chars=args.min_chars, near_dedup=not args.no_near_dedup)
        return 0

    if args.cmd == "partition":
        rows = partition.summarize(cfg, args.workers)
        for r in rows:
            print(f"  worker {r['worker_id']:>3}: {r['units']:>10,} units  {r['chars']:>14,} chars")
        return 0

    if args.cmd == "bench":
        pr, oll = _prompt_from(args), _oll_from(args)
        res = benchmark.run(cfg, pr, oll, args.n)
        print("\nProject cost with:")
        print(f"  python -m newsmanager.extract project --units-per-s {res['units_per_s']:.3f} --vms 4")
        return 0

    if args.cmd == "sweep":
        pr, oll = _prompt_from(args), _oll_from(args)
        benchmark.sweep(cfg, pr, oll, [int(x) for x in args.levels.split(",")], args.n)
        return 0

    if args.cmd == "project":
        benchmark.project(cfg, args.units_per_s, n_vms=args.vms, usd_per_gpu_hour=args.usd_per_gpu_hour)
        return 0

    if args.cmd == "run":
        pr, oll = _prompt_from(args), _oll_from(args)
        worker.run(
            cfg, pr, oll, args.worker_id, args.workers,
            limit=args.limit, resume=not args.no_resume,
            checkpoint_every=args.checkpoint_every, order=args.order,
        )
        return 0

    if args.cmd == "collect":
        version = args.prompt_version
        if version is None:
            pp = Path(args.prompt) if args.prompt else _repo() / "prompts" / "extraction.txt"
            sp = Path(args.schema) if args.schema else _repo() / "prompts" / "schema.json"
            version = load_prompt(pp, sp, args.max_chars).version
        stats = collect.collect(cfg, version)
        if not args.no_views:
            collect.build_views(cfg, version)
        print(json.dumps(stats, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
