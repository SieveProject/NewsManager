"""`python -m newsmanager.extract <stage>` -- the LLM extraction pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .. import config as cfgmod
from ..config import EXIT_CONFIG, ConfigError
from . import benchmark, collect, link, noise, partition, units, worker
from .client import OllamaConfig
from .prompt import DEFAULT_MAX_CHARS, load as load_prompt

# Instruction model, not a reasoning one. On the same 200 articles and prompt,
# qwen2.5:14b-instruct beat deepseek-r1:14b on every quality measure (~2/3 vs
# ~1/4 of sampled tuples were real agent-to-agent relations; padding to the
# 8-tuple cap 5% vs 30%; metrics as agents 2.8% vs 7.2%) and ran 39% faster.
DEFAULT_MODEL = "qwen2.5:14b-instruct"
# Modelo leve da etapa link: escolhe entre candidatos fechados, não extrai.
DEFAULT_LINK_MODEL = "qwen2.5:7b-instruct"


def _repo() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=os.environ.get("NM_MODEL", DEFAULT_MODEL))
    p.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    p.add_argument("--concurrency", type=int, default=int(os.environ.get("NM_CONCURRENCY", "8")))
    p.add_argument("--num-ctx", type=int,
                   default=(int(os.environ["NM_NUM_CTX"]) if os.environ.get("NM_NUM_CTX") else None),
                   help="context window; derived from --max-chars when omitted")
    # 768, not 1024: with the schema capping tuples at 8 the longest response
    # measured was 568 tokens (48 stratified articles). The spare 256 tokens
    # are what keeps num_ctx at 4096 -- and 16 parallel slots inside 24 GB --
    # now that the prompt is ~4k characters.
    p.add_argument("--num-predict", type=int, default=768)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--prompt", default=None, help="prompt template (default prompts/extraction.txt)")
    p.add_argument("--schema", default=None, help="JSON schema (default prompts/schema.json)")
    p.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    p.add_argument("--think", action="store_true",
                   help="liga o raciocínio (multiplica tokens de saída; desligado por padrão)")


def _required_num_ctx(max_chars: int, template: str, num_predict: int) -> int:
    """Smallest context that fits a capped article, the template and the output.

    An article capped at max_chars, plus the template and room to generate,
    at worst-case token density. Exceeding num_ctx makes Ollama drop the *end of the
    article* with no error at all, so this is computed rather than trusted.
    """
    # Worst case, not average: ~4 chars/token holds for English prose, but
    # number-dense articles (tables, filings) measured 2.1 -- a 6,502-char one
    # took 3,997 prompt tokens and left 99 for the answer at num_ctx 4096.
    # 2 chars/token for the article, 3 for the template. Mirrored in
    # deploy/lib.sh (derive_num_ctx) so the server loads the same context.
    return max_chars // 2 + len(template) // 3 + num_predict + 256


def _resolve(args) -> tuple:
    """Load the prompt and settle num_ctx together -- each constrains the other.

    Returning both from one place keeps the two from drifting apart: num_ctx
    depends on the template length, which is only known after the prompt loads.
    """
    pp = Path(args.prompt) if args.prompt else _repo() / "prompts" / "extraction.txt"
    sp = Path(args.schema) if args.schema else _repo() / "prompts" / "schema.json"
    pr = load_prompt(pp, sp, args.max_chars, model=args.model)

    need = _required_num_ctx(args.max_chars, pr.template, args.num_predict)
    if args.num_ctx is None:
        # Derived, not defaulted: prompts/extraction.txt is meant to be edited,
        # and a fixed default silently becomes too small as the template grows.
        args.num_ctx = -(-need // 256) * 256
        print(f"[extract] --num-ctx derivado de --max-chars {args.max_chars}: "
              f"{args.num_ctx} (mínimo {need})")
    elif args.num_ctx < need:
        raise ConfigError(
            f"--num-ctx {args.num_ctx} is too small for --max-chars {args.max_chars}: "
            f"need ~{need} tokens (article + prompt + {args.num_predict} generated). "
            f"Raise --num-ctx to {need}, lower --max-chars, or omit --num-ctx to "
            f"have it derived. Ollama would otherwise silently drop the end of "
            f"each article."
        )
    return pr, _oll_from(args)


def _oll_from(args) -> OllamaConfig:
    return OllamaConfig(
        host=args.host, model=args.model, num_ctx=args.num_ctx,
        num_predict=args.num_predict, concurrency=args.concurrency, timeout_s=args.timeout,
        think=getattr(args, 'think', False),
    )


def _dispatch(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nm-extract", description="LLM relation extraction over the news corpus")
    p.add_argument("-c", "--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("units", help="build the deduplicated list of LLM calls to make")
    u.add_argument("--min-chars", type=int, default=None)
    u.add_argument("--no-near-dedup", action="store_true")
    u.add_argument("--sample-frac", type=float, default=None,
                   help="keep a deterministic, nested hash sample of units (e.g. 0.1)")

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

    ab = sub.add_parser("absorb", help="absorve WALs órfãos (VMs já recolhidas) em parquet")
    ab.add_argument("--prompt-version", default=None)

    c = sub.add_parser("collect", help="merge worker shards into parquet + views")
    c.add_argument("--prompt-version", default=None)
    c.add_argument("--prompt", default=None)
    c.add_argument("--schema", default=None)
    c.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    c.add_argument("--model", default=os.environ.get("NM_MODEL", DEFAULT_MODEL),
                   help="model the run used; part of prompt_version")
    c.add_argument("--no-views", action="store_true")

    fl = sub.add_parser("filter", help="marca tuplas ruidosas -> news.relations_clean")
    fl.add_argument("--prompt-version", default=None)
    fl.add_argument("--prompt", default=None)
    fl.add_argument("--schema", default=None)
    fl.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    fl.add_argument("--model", default=os.environ.get("NM_MODEL", DEFAULT_MODEL))
    fl.add_argument("--keep", default="",
                    help="motivos a NÃO excluir de relations_clean, separados por vírgula "
                         "(ex.: comparison,membership)")

    # Etapa link: nomes dos agentes -> tickers (ver newsmanager/extract/link.py).
    # --relations-version identifica as tuplas; o padrão é a versão dos prompts
    # de extração com o modelo de extração (não o modelo leve do link).
    lp = sub.add_parser("link-prep", help="normaliza nomes e monta candidatos a ticker (sem GPU)")
    lp.add_argument("--relations-version", default=None)
    lp.add_argument("--refresh", action="store_true", help="baixa de novo as listas de tickers")

    lr = sub.add_parser("link-run", help="modelo leve escolhe o ticker entre os candidatos")
    lr.add_argument("--relations-version", default=None)
    lr.add_argument("--model", default=os.environ.get("NM_LINK_MODEL", DEFAULT_LINK_MODEL))
    lr.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    lr.add_argument("--concurrency", type=int, default=int(os.environ.get("NM_CONCURRENCY", "32")))
    lr.add_argument("--num-ctx", type=int, default=int(os.environ.get("NM_NUM_CTX", "3072")))
    lr.add_argument("--num-predict", type=int, default=128)
    lr.add_argument("--timeout", type=float, default=120.0)
    lr.add_argument("--prompt", default=None, help="template (padrão prompts/link.txt)")
    lr.add_argument("--limit", type=int, default=None)
    lr.add_argument("--order", choices=("freq", "hash"), default="freq",
                    help="freq: nomes mais citados primeiro; hash: amostra uniforme (piloto)")
    lr.add_argument("--min-n", type=int, default=1, help="só nomes com ao menos N menções")

    lv = sub.add_parser("link-verify", help="confere os tickers aceitos só por evidência (sim/não)")
    lv.add_argument("--relations-version", default=None)
    lv.add_argument("--link-version", required=True)
    lv.add_argument("--model", default=os.environ.get("NM_LINK_MODEL", DEFAULT_LINK_MODEL))
    lv.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    lv.add_argument("--concurrency", type=int, default=int(os.environ.get("NM_CONCURRENCY", "32")))
    lv.add_argument("--prompt", default=None, help="template (padrão prompts/link_verify.txt)")
    lv.add_argument("--limit", type=int, default=None)

    lb = sub.add_parser("link-build", help="grava entity_map + relations_linked e as views")
    lb.add_argument("--relations-version", default=None)
    lb.add_argument("--link-version", default=None)
    lb.add_argument("--no-views", action="store_true")

    args = p.parse_args(argv)
    cfg = cfgmod.load(args.config)
    cfg.ensure_dirs()

    if args.cmd == "units":
        units.build(cfg, min_chars=args.min_chars, near_dedup=not args.no_near_dedup,
                    sample_frac=args.sample_frac)
        return 0

    if args.cmd == "partition":
        rows = partition.summarize(cfg, args.workers)
        for r in rows:
            print(f"  worker {r['worker_id']:>3}: {r['units']:>10,} units  {r['chars']:>14,} chars")
        return 0

    if args.cmd == "bench":
        pr, oll = _resolve(args)
        res = benchmark.run(cfg, pr, oll, args.n)
        print("\nProject cost with:")
        print(f"  python -m newsmanager.extract project --units-per-s {res['units_per_s']:.3f} --vms 4")
        return 0

    if args.cmd == "sweep":
        pr, oll = _resolve(args)
        benchmark.sweep(cfg, pr, oll, [int(x) for x in args.levels.split(",")], args.n)
        return 0

    if args.cmd == "project":
        benchmark.project(cfg, args.units_per_s, n_vms=args.vms, usd_per_gpu_hour=args.usd_per_gpu_hour)
        return 0

    if args.cmd == "run":
        pr, oll = _resolve(args)
        worker.run(
            cfg, pr, oll, args.worker_id, args.workers,
            limit=args.limit, resume=not args.no_resume,
            checkpoint_every=args.checkpoint_every, order=args.order,
        )
        return 0

    if args.cmd == "absorb":
        collect.absorb_orphan_wals(cfg, args.prompt_version)
        return 0

    if args.cmd == "filter":
        version = args.prompt_version
        if version is None:
            pp = Path(args.prompt) if args.prompt else _repo() / "prompts" / "extraction.txt"
            sp = Path(args.schema) if args.schema else _repo() / "prompts" / "schema.json"
            version = load_prompt(pp, sp, args.max_chars, model=args.model).version
        keep = {k.strip() for k in args.keep.split(",") if k.strip()}
        noise.build(cfg, version, exclude=tuple(r for r in noise.REASONS if r not in keep))
        return 0

    if args.cmd.startswith("link-"):
        rv = args.relations_version or load_prompt(
            _repo() / "prompts" / "extraction.txt", _repo() / "prompts" / "schema.json",
            DEFAULT_MAX_CHARS, model=DEFAULT_MODEL).version
        if args.cmd == "link-prep":
            link.prepare(cfg, rv, refresh=args.refresh)
        elif args.cmd == "link-run":
            oll = OllamaConfig(host=args.host, model=args.model, num_ctx=args.num_ctx,
                               num_predict=args.num_predict, concurrency=args.concurrency,
                               timeout_s=args.timeout)
            tp = Path(args.prompt) if args.prompt else _repo() / "prompts" / "link.txt"
            print(json.dumps(link.run(cfg, rv, oll, tp, limit=args.limit, order=args.order,
                                      min_n=args.min_n), indent=2))
        elif args.cmd == "link-verify":
            oll = OllamaConfig(host=args.host, model=args.model, num_ctx=int(os.environ.get("NM_NUM_CTX", "3072")), num_predict=16,
                               concurrency=args.concurrency, timeout_s=120.0)
            tp = Path(args.prompt) if args.prompt else _repo() / "prompts" / "link_verify.txt"
            print(json.dumps(link.verify(cfg, rv, args.link_version, oll, tp, limit=args.limit), indent=2))
        else:
            link.build(cfg, rv, args.link_version, views=not args.no_views)
        return 0

    if args.cmd == "collect":
        version = args.prompt_version
        if version is None:
            pp = Path(args.prompt) if args.prompt else _repo() / "prompts" / "extraction.txt"
            sp = Path(args.schema) if args.schema else _repo() / "prompts" / "schema.json"
            version = load_prompt(pp, sp, args.max_chars, model=args.model).version
        # Antes de unir: WALs recolhidos de VMs destruídas ainda não estão em
        # Parquet, e o collect só enxerga Parquet.
        collect.absorb_orphan_wals(cfg, version)
        stats = collect.collect(cfg, version)
        if not args.no_views:
            collect.build_views(cfg, version)
        print(json.dumps(stats, indent=2))
        return 0

    return 1


def main(argv: list[str] | None = None) -> int:
    """Translate a misconfiguration into EXIT_CONFIG.

    `run_worker.sh` restarts a worker that exits 1, because that is what a
    transient crash looks like. A misconfiguration re-fails identically every
    time, so it gets its own code and stops the loop instead of spending rented
    GPU minutes proving the same point a hundred times.
    """
    try:
        return _dispatch(argv)
    except ConfigError as e:
        print(f"\n[extract] erro de configuração: {e}", file=sys.stderr)
        print("[extract] reiniciar não resolve isso -- corrija e rode de novo.", file=sys.stderr)
        return EXIT_CONFIG


if __name__ == "__main__":
    raise SystemExit(main())
