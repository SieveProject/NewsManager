"""Loop de extração de uma VM.

Durabilidade
------------
Cada resultado é persistido no instante em que chega: primeiro num WAL JSONL com
fsync, depois consolidado em segmentos Parquet completos a cada
`checkpoint_every` registros (ver `sink.py`). Uma VM que morre no meio não perde
nenhum registro e não deixa nenhum Parquet ilegível.

A retomada lê os unit_ids já persistidos (segmentos + WAL) e pula esses, de modo
que reiniciar um worker morto custa zero inferência repetida.

Falhas viram linhas com `status='failed'` em vez de sumirem. Uma unidade que
falhou continua distinguível de uma que nunca foi tentada -- é isso que permite
reprocessar só o que falhou em vez de refazer a run inteira.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..ingest import connect
from .client import OllamaClient, OllamaConfig
from .partition import assign_expr
from .prompt import Prompt
from .sink import ParquetSink, absorb_wal, recover_done
from .units import units_glob


@dataclass
class WorkerStats:
    attempted: int = 0
    ok: int = 0
    failed: int = 0
    tuples: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    def rate(self) -> float:
        return self.attempted / self.seconds if self.seconds else 0.0


def output_root(cfg: Config, prompt_version: str) -> Path:
    d = cfg.root / "extractions" / f"v={prompt_version}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch_units(cfg: Config, worker_id: int, n_workers: int, limit: int | None, order: str) -> list[dict]:
    con = connect(cfg, memory_limit="4GB")
    # Mais longos primeiro: mantém os itens lentos fora do fim da run, onde
    # deixariam a GPU processando um artigo gigante com o lote quase vazio.
    order_sql = {"long_first": "body_chars DESC", "short_first": "body_chars ASC", "natural": "unit_id"}[order]
    sql = f"""
        SELECT unit_id, repr_doc_id, published_at, year, title, body, body_chars,
               all_symbols AS symbols
        FROM read_parquet('{units_glob(cfg)}')
        WHERE {assign_expr(n_workers, worker_id)}
        ORDER BY {order_sql}
    """
    if limit:
        sql += f" LIMIT {limit}"
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    con.close()
    return rows


async def _run_async(
    cfg: Config,
    prompt: Prompt,
    oll: OllamaConfig,
    worker_id: int,
    n_workers: int,
    *,
    limit: int | None = None,
    resume: bool = True,
    checkpoint_every: int = 200,
    order: str = "long_first",
    progress_every: int = 50,
) -> WorkerStats:
    root = output_root(cfg, prompt.version)

    # Um WAL sobrevivente de uma queda anterior vira segmento Parquet antes de
    # qualquer coisa; senão esses registros ficariam invisíveis para o collect.
    absorbed = absorb_wal(root, worker_id, prompt.version, oll.model)
    if absorbed:
        print(f"[worker {worker_id}] {absorbed} registro(s) recuperados do WAL da execução anterior")

    units = _fetch_units(cfg, worker_id, n_workers, limit, order)
    done = recover_done(root, worker_id) if resume else set()
    todo = [u for u in units if u["unit_id"] not in done]

    print(f"[worker {worker_id}] {len(units):,} atribuídas, {len(done):,} já feitas, {len(todo):,} a fazer")
    print(f"[worker {worker_id}] model={oll.model} think={oll.think} concurrency={oll.concurrency} "
          f"num_ctx={oll.num_ctx} prompt_version={prompt.version}")
    if not todo:
        return WorkerStats()

    stats = WorkerStats()
    t0 = time.time()
    lock = asyncio.Lock()

    with ParquetSink(root, worker_id, prompt.version, oll.model, flush_every=checkpoint_every) as sink:
        async with OllamaClient(oll) as client:
            health = await client.health()
            if not health["has_model"]:
                raise RuntimeError(
                    f"modelo {oll.model!r} ausente em {oll.host}. "
                    f"Disponíveis: {health['models']}. Rode: ollama pull {oll.model}"
                )
            await client.warm()

            async def handle(unit: dict) -> None:
                syms = list(unit.get("symbols") or [])
                text, truncated = prompt.render(
                    article=unit["body"],
                    title=unit.get("title"),
                    date=str(unit.get("published_at") or ""),
                    symbols=", ".join(syms) if syms else None,
                )
                res = await client.generate(text, prompt.schema)

                tuples = []
                if res.ok and isinstance(res.parsed, dict):
                    raw = res.parsed.get("tuples")
                    if isinstance(raw, list):
                        tuples = [t for t in raw if isinstance(t, dict)]

                record = {
                    "unit_id": unit["unit_id"],
                    "repr_doc_id": unit["repr_doc_id"],
                    # A data da notícia acompanha cada tupla gravada.
                    "published_at": unit.get("published_at"),
                    "year": unit.get("year"),
                    "symbols": syms,
                    "status": "ok" if res.ok else "failed",
                    "error": res.error or None,
                    "truncated": truncated,
                    "body_chars": unit.get("body_chars"),
                    "prompt_tokens": res.prompt_tokens,
                    "output_tokens": res.output_tokens,
                    "latency_s": round(res.latency_s, 3),
                    "tuples": tuples,
                }

                async with lock:
                    sink.append(record)
                    stats.attempted += 1
                    stats.prompt_tokens += res.prompt_tokens
                    stats.output_tokens += res.output_tokens
                    if res.ok:
                        stats.ok += 1
                        stats.tuples += len(tuples)
                    else:
                        stats.failed += 1
                    if stats.attempted % progress_every == 0:
                        el = time.time() - t0
                        r = stats.attempted / el
                        eta = (len(todo) - stats.attempted) / r if r else 0
                        print(
                            f"[worker {worker_id}] {stats.attempted:,}/{len(todo):,} "
                            f"ok={stats.ok:,} falhas={stats.failed:,} tuplas={stats.tuples:,} "
                            f"{r:.2f}/s eta={eta/3600:.2f}h"
                        )

            # Limitado pelo semáforo do cliente; um gather sobre todas as
            # unidades criaria um objeto de task por unidade -- milhões deles.
            pending: set[asyncio.Task] = set()
            for unit in todo:
                pending.add(asyncio.create_task(handle(unit)))
                if len(pending) >= oll.concurrency * 4:
                    _, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            if pending:
                await asyncio.wait(pending)

    stats.seconds = time.time() - t0
    print(
        f"[worker {worker_id}] concluído: {stats.ok:,} ok, {stats.failed:,} falhas, "
        f"{stats.tuples:,} tuplas em {stats.seconds/3600:.2f}h ({stats.rate():.2f}/s) -> {root}"
    )
    return stats


def run(cfg: Config, prompt: Prompt, oll: OllamaConfig, worker_id: int, n_workers: int, **kw) -> WorkerStats:
    return asyncio.run(_run_async(cfg, prompt, oll, worker_id, n_workers, **kw))
