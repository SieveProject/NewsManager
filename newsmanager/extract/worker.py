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
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..config import Config, ConfigError
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


def _check_units(cfg: Config) -> None:
    # On a worker VM the corpus arrives by rsync, so "not there yet" is the
    # normal way this goes wrong. Caught here it names the fix; left to DuckDB
    # it surfaces as an IO error about a glob, and the restart loop retries it.
    if not (cfg.curated / "extraction_units").exists():
        raise ConfigError(
            f"nenhuma unidade de extração em {cfg.curated / 'extraction_units'}. "
            "No worker 0: python -m newsmanager.extract units. "
            "Nos demais: sincronize data/curated/ do worker 0 primeiro "
            "(./deploy/gather.sh push <host>)."
        )


def _count_units(cfg: Config, worker_id: int, n_workers: int, limit: int | None) -> int:
    con = connect(cfg, memory_limit="1GB")
    n = con.execute(
        f"SELECT count(*) FROM read_parquet('{units_glob(cfg)}') WHERE {assign_expr(n_workers, worker_id)}"
    ).fetchone()[0]
    con.close()
    return min(n, limit) if limit else n


def _iter_units(cfg: Config, worker_id: int, n_workers: int, limit: int | None, order: str,
                skip: set[str], batch_rows: int = 2000) -> Iterator[dict]:
    """Stream this worker's units in batches, skipping those already done.

    Never fetchall(): with one worker the partition is the whole corpus, and
    every body as a Python str is tens of GB -- enough to OOM the VM hours into
    a paid run. The ORDER BY spills to cfg.tmp; Python holds one batch at a time.
    """
    con = connect(cfg, memory_limit="4GB")
    # Mais longos primeiro: mantém os itens lentos fora do fim da run, onde
    # deixariam a GPU processando um artigo gigante com o lote quase vazio.
    order_sql = {"long_first": "body_chars DESC", "short_first": "body_chars ASC", "natural": "unit_id"}[order]
    sql = f"""
        SELECT unit_id, repr_doc_id, published_at, year, title, body, body_chars,
               all_symbols AS symbols
        FROM read_parquet('{units_glob(cfg)}')
        WHERE {assign_expr(n_workers, worker_id)}
        ORDER BY {order_sql}, unit_id
    """
    if limit:
        sql += f" LIMIT {limit}"
    try:
        reader = con.execute(sql).fetch_record_batch(batch_rows)
        for batch in reader:
            for row in batch.to_pylist():
                if row["unit_id"] not in skip:
                    yield row
    finally:
        con.close()


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

    _check_units(cfg)
    n_assigned = _count_units(cfg, worker_id, n_workers, limit)
    done = recover_done(root, worker_id) if resume else set()
    # Aproximado só com --limit (feitas fora das primeiras N também contam);
    # serve para progresso/ETA, não para decidir o que rodar.
    n_todo = max(n_assigned - len(done), 0)

    print(f"[worker {worker_id}] {n_assigned:,} atribuídas, {len(done):,} já feitas, {n_todo:,} a fazer")
    print(f"[worker {worker_id}] model={oll.model} think={oll.think} concurrency={oll.concurrency} "
          f"num_ctx={oll.num_ctx} prompt_version={prompt.version}")
    if not n_todo:
        return WorkerStats()

    stats = WorkerStats()
    t0 = time.time()
    lock = asyncio.Lock()

    with ParquetSink(root, worker_id, prompt.version, oll.model, flush_every=checkpoint_every) as sink:
        async with OllamaClient(oll) as client:
            health = await client.health()
            if not health["has_model"]:
                raise ConfigError(
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
                        eta = max(n_todo - stats.attempted, 0) / r if r else 0
                        print(
                            f"[worker {worker_id}] {stats.attempted:,}/{n_todo:,} "
                            f"ok={stats.ok:,} falhas={stats.failed:,} tuplas={stats.tuples:,} "
                            f"{r:.2f}/s eta={eta/3600:.2f}h"
                        )

            # Limitado pelo semáforo do cliente; um gather sobre todas as
            # unidades criaria um objeto de task por unidade -- milhões deles.
            pending: set[asyncio.Task] = set()
            for unit in _iter_units(cfg, worker_id, n_workers, limit, order, done):
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
