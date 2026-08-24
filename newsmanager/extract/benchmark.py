"""Measure real throughput, then project corpus cost from it.

Every published tokens/sec figure is for a different GPU, quantisation, context
length and batch size than yours. The only number worth planning against is one
measured on the machine you are about to rent, with your prompt and your
articles -- so `benchmark` runs a real sample and `project` extrapolates from it.

Concurrency matters more than anything else here: Ollama batches concurrent
requests into a single forward pass, so throughput typically climbs steeply to a
plateau and then degrades as VRAM pressure forces eviction. `sweep` finds that
plateau instead of guessing at it.
"""

from __future__ import annotations

import asyncio
import random
import statistics
import time

from ..config import Config
from ..ingest import connect
from .client import OllamaClient, OllamaConfig
from .prompt import Prompt
from .units import units_glob


def _sample_units(cfg: Config, n: int, seed: int = 7) -> list[dict]:
    """Length-stratified sample.

    A uniform random sample of a p50=3.5k / p99=32k distribution is mostly
    median articles and under-represents the long tail that dominates runtime,
    which would make the projection optimistic.
    """
    con = connect(cfg, memory_limit="4GB")
    cur = con.execute(
        f"""
        WITH ranked AS (
            SELECT unit_id, title, published_at, body, body_chars, all_symbols AS symbols,
                   ntile(4) OVER (ORDER BY body_chars) AS q
            FROM read_parquet('{units_glob(cfg)}')
            USING SAMPLE {max(n * 8, 400)} ROWS (reservoir, {seed})
        )
        SELECT unit_id, title, published_at, body, body_chars, symbols FROM (
            SELECT *, row_number() OVER (PARTITION BY q ORDER BY unit_id) rn FROM ranked
        ) WHERE rn <= {max(1, n // 4)}
        """
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    con.close()
    random.Random(seed).shuffle(rows)
    return rows


async def _timed_run(units: list[dict], prompt: Prompt, oll: OllamaConfig) -> dict:
    async with OllamaClient(oll) as client:
        health = await client.health()
        if not health["has_model"]:
            raise RuntimeError(f"model {oll.model!r} not on {oll.host}; available: {health['models']}")
        await client.warm()  # exclude cold-load from the measurement

        async def one(u):
            text, _ = prompt.render(
                article=u["body"], title=u.get("title"),
                date=str(u.get("published_at") or ""),
                symbols=", ".join(u.get("symbols") or []),
            )
            return await client.generate(text, prompt.schema)

        t0 = time.time()
        results = await asyncio.gather(*(one(u) for u in units))
        elapsed = time.time() - t0

    ok = [r for r in results if r.ok]
    lat = sorted(r.latency_s for r in ok)
    ptok = sum(r.prompt_tokens for r in ok)
    otok = sum(r.output_tokens for r in ok)
    return {
        "n": len(units),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "elapsed_s": elapsed,
        "units_per_s": len(ok) / elapsed if elapsed else 0,
        "prompt_tokens": ptok,
        "output_tokens": otok,
        "prompt_tok_per_s": ptok / elapsed if elapsed else 0,
        "output_tok_per_s": otok / elapsed if elapsed else 0,
        "latency_p50": statistics.median(lat) if lat else 0,
        "latency_p95": lat[int(len(lat) * 0.95)] if lat else 0,
        "mean_tuples": 0,
    }


def run(cfg: Config, prompt: Prompt, oll: OllamaConfig, n: int = 40) -> dict:
    units = _sample_units(cfg, n)
    if not units:
        raise RuntimeError("no extraction units; run `nm-extract units` first")
    print(f"[bench] {len(units)} units, model={oll.model}, concurrency={oll.concurrency}, num_ctx={oll.num_ctx}")
    res = asyncio.run(_timed_run(units, prompt, oll))
    print(
        f"[bench] {res['ok']}/{res['n']} ok in {res['elapsed_s']:.1f}s -> "
        f"{res['units_per_s']:.3f} units/s ({res['units_per_s']*3600:,.0f}/h)\n"
        f"[bench] prompt {res['prompt_tok_per_s']:,.0f} tok/s, output {res['output_tok_per_s']:,.0f} tok/s, "
        f"latency p50={res['latency_p50']:.1f}s p95={res['latency_p95']:.1f}s"
    )
    return res


def sweep(cfg: Config, prompt: Prompt, oll: OllamaConfig, levels: list[int], n: int = 32) -> list[dict]:
    """Find the concurrency plateau for this GPU."""
    units = _sample_units(cfg, n)
    out = []
    for c in levels:
        trial = OllamaConfig(**{**oll.__dict__, "concurrency": c})
        res = asyncio.run(_timed_run(units, prompt, trial))
        res["concurrency"] = c
        out.append(res)
        print(f"[sweep] concurrency={c:>3}: {res['units_per_s']:.3f} units/s "
              f"({res['units_per_s']*3600:,.0f}/h) p95={res['latency_p95']:.1f}s failed={res['failed']}")
    best = max(out, key=lambda r: r["units_per_s"])
    print(f"[sweep] best concurrency = {best['concurrency']} at {best['units_per_s']*3600:,.0f} units/h")
    return out


def project(cfg: Config, units_per_s: float, *, n_vms: int, usd_per_gpu_hour: float) -> dict:
    """Extrapolate a measured rate to the whole corpus."""
    con = connect(cfg, memory_limit="4GB")
    n_units, total_chars = con.execute(
        f"SELECT count(*), sum(body_chars)::BIGINT FROM read_parquet('{units_glob(cfg)}')"
    ).fetchone()
    con.close()

    gpu_hours = n_units / units_per_s / 3600 if units_per_s else float("inf")
    wall_hours = gpu_hours / n_vms if n_vms else gpu_hours
    cost = gpu_hours * usd_per_gpu_hour

    print(
        f"\n=== corpus projection ===\n"
        f"units                {n_units:,}\n"
        f"measured rate        {units_per_s:.3f} units/s/VM ({units_per_s*3600:,.0f}/h)\n"
        f"total GPU-hours      {gpu_hours:,.1f}\n"
        f"VMs                  {n_vms}\n"
        f"wall-clock           {wall_hours:,.1f} h ({wall_hours/24:.1f} days)\n"
        f"cost @ ${usd_per_gpu_hour:.2f}/GPU-h  ${cost:,.0f}\n"
        f"cost per 1k units    ${cost/n_units*1000:.3f}\n"
    )
    print(
        "Rented VMs bill wall-clock, so total cost is roughly flat in VM count "
        "while wall-clock falls linearly. Rent for deadline, not for budget."
    )
    return {
        "units": n_units,
        "total_chars": total_chars,
        "units_per_s": units_per_s,
        "gpu_hours": gpu_hours,
        "n_vms": n_vms,
        "wall_hours": wall_hours,
        "usd_total": cost,
    }
