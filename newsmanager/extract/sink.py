"""Gravação incremental em Parquet, à prova de queda da VM.

Por que não é Parquet puro linha a linha
----------------------------------------
Parquet grava o *footer* (índice de row groups) apenas no fechamento do arquivo.
Um processo morto no meio deixa um arquivo sem footer -- ilegível por inteiro,
não apenas na última linha. Escrever "um Parquet por prompt" resolveria isso mas
geraria milhões de arquivos de poucos KB, e ler esse diretório depois custaria
mais que a própria extração.

O desenho usado aqui
--------------------
1. Cada registro vai imediatamente para um WAL em JSONL, com fsync. Nada é
   perdido, nem o último prompt antes da queda.
2. A cada `flush_every` registros o buffer vira um *segmento* Parquet completo,
   escrito em .tmp e renomeado atomicamente. Segmento publicado é segmento
   legível -- nunca existe Parquet pela metade.
3. O WAL é truncado só depois do rename. Se a VM cair entre as duas coisas, a
   recuperação lê segmento e WAL e descarta a duplicata por unit_id.

Resultado: a saída é Parquet (consultável direto pelo DuckDB, sem etapa de
conversão) e a perda máxima em uma queda é zero registro.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

# Schemas explícitos: sem inferência. Um segmento gravado numa VM onde todas as
# tuplas vieram vazias inferiria tipos diferentes de outra VM, e o glob final
# falharia ao unir os dois.
RELATIONS_SCHEMA = pa.schema([
    ("unit_id", pa.string()),
    ("repr_doc_id", pa.string()),
    ("published_at", pa.timestamp("us")),   # a data da notícia, exigida em cada tupla
    ("year", pa.int32()),
    ("symbols", pa.list_(pa.string())),
    ("agent_a", pa.string()),
    ("agent_b", pa.string()),
    ("relation_type", pa.string()),
    ("direction", pa.string()),
    ("strength", pa.float64()),
    ("model", pa.string()),
    ("prompt_version", pa.string()),
    ("extracted_at", pa.timestamp("us")),
])

RUNS_SCHEMA = pa.schema([
    ("unit_id", pa.string()),
    ("repr_doc_id", pa.string()),
    ("published_at", pa.timestamp("us")),
    ("year", pa.int32()),
    ("status", pa.string()),
    ("error", pa.string()),
    ("truncated", pa.bool_()),
    ("body_chars", pa.int64()),
    ("prompt_tokens", pa.int32()),
    ("output_tokens", pa.int32()),
    ("latency_s", pa.float64()),
    ("n_tuples", pa.int32()),
    ("model", pa.string()),
    ("prompt_version", pa.string()),
    ("extracted_at", pa.timestamp("us")),
])


def _ts(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    try:
        return datetime.fromisoformat(str(value).replace(" UTC", "").strip()).replace(tzinfo=None)
    except ValueError:
        return None


def _clamp01(v: Any) -> float | None:
    """Mantém strength em [0,1].

    Constrained decoding garante o *tipo* number, não a faixa: modelos emitem
    1.5 ou -0.2 com alguma frequência. Corrigir aqui evita que a análise a
    jusante tenha que desconfiar da coluna.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, f))


def heal_partial_line(path: Path) -> bool:
    """Remove uma última linha truncada antes de reabrir o WAL em append.

    Um processo morto no meio da escrita deixa uma linha sem \\n final. Abrir em
    modo append e escrever grudaria o próximo registro nesse fragmento: uma
    linha corrompida e um registro perdido de vez, porque a retomada não
    consegue reprocessar um registro cujo unit_id está ilegível.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb+") as fh:
        fh.seek(-1, os.SEEK_END)
        if fh.read(1) == b"\n":
            return False
        size = path.stat().st_size
        pos, step = size, 65536
        while pos > 0:
            back = min(step, pos)
            pos -= back
            fh.seek(pos)
            nl = fh.read(back).rfind(b"\n")
            if nl != -1:
                fh.truncate(pos + nl + 1)
                return True
        fh.truncate(0)  # arquivo inteiro é uma única linha parcial
        return True


class ParquetSink:
    """Escreve segmentos Parquet + WAL para um worker."""

    def __init__(self, root: Path, worker_id: int, prompt_version: str, model: str,
                 flush_every: int = 200, fsync_wal: bool = True):
        self.rel_dir = root / "relations"
        self.run_dir = root / "runs"
        self.rel_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.worker_id = worker_id
        self.prompt_version = prompt_version
        self.model = model
        self.flush_every = flush_every
        self.fsync_wal = fsync_wal
        self.wal_path = root / f"wal-{worker_id:03d}.jsonl"
        self._rel_buf: list[dict] = []
        self._run_buf: list[dict] = []
        self._pending = 0
        self._seq = self._next_seq()
        heal_partial_line(self.wal_path)
        self._wal = self.wal_path.open("a", encoding="utf-8")

    def _next_seq(self) -> int:
        existing = list(self.run_dir.glob(f"w{self.worker_id:03d}-*.parquet"))
        if not existing:
            return 0
        return max(int(p.stem.split("-")[1]) for p in existing) + 1

    # ---------------------------------------------------------------- escrita

    def append(self, record: dict) -> None:
        """Registra um resultado. Vai para o WAL agora, para Parquet no flush."""
        self._wal.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._wal.flush()
        if self.fsync_wal:
            os.fsync(self._wal.fileno())

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        pub = _ts(record.get("published_at"))
        self._run_buf.append({
            "unit_id": record["unit_id"],
            "repr_doc_id": record.get("repr_doc_id"),
            "published_at": pub,
            "year": int(record["year"]) if record.get("year") is not None else None,
            "status": record.get("status"),
            "error": record.get("error"),
            "truncated": bool(record.get("truncated", False)),
            "body_chars": record.get("body_chars"),
            "prompt_tokens": record.get("prompt_tokens") or 0,
            "output_tokens": record.get("output_tokens") or 0,
            "latency_s": record.get("latency_s"),
            "n_tuples": len(record.get("tuples") or []),
            "model": self.model,
            "prompt_version": self.prompt_version,
            "extracted_at": now,
        })
        for t in record.get("tuples") or []:
            if not isinstance(t, dict):
                continue
            self._rel_buf.append({
                "unit_id": record["unit_id"],
                "repr_doc_id": record.get("repr_doc_id"),
                "published_at": pub,
                "year": int(record["year"]) if record.get("year") is not None else None,
                "symbols": list(record.get("symbols") or []),
                "agent_a": t.get("agent_a"),
                "agent_b": t.get("agent_b"),
                "relation_type": t.get("relation_type"),
                "direction": t.get("direction"),
                "strength": _clamp01(t.get("strength")),
                "model": self.model,
                "prompt_version": self.prompt_version,
                "extracted_at": now,
            })
        self._pending += 1
        if self._pending >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Publica um segmento e só então limpa o WAL."""
        if not self._pending:
            return
        seq = self._seq
        self._write(self.run_dir / f"w{self.worker_id:03d}-{seq:06d}.parquet", self._run_buf, RUNS_SCHEMA)
        if self._rel_buf:
            self._write(self.rel_dir / f"w{self.worker_id:03d}-{seq:06d}.parquet", self._rel_buf, RELATIONS_SCHEMA)
        self._seq += 1
        self._rel_buf.clear()
        self._run_buf.clear()
        self._pending = 0
        # Só agora o WAL pode ser descartado: os dados já estão num Parquet legível.
        self._wal.close()
        self.wal_path.unlink(missing_ok=True)
        self._wal = self.wal_path.open("a", encoding="utf-8")

    @staticmethod
    def _write(path: Path, rows: list[dict], schema: pa.Schema) -> None:
        table = pa.Table.from_pylist(rows, schema=schema)
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)  # publicação atômica

    def close(self) -> None:
        self.flush()
        self._wal.close()
        self.wal_path.unlink(missing_ok=True)

    def __enter__(self) -> "ParquetSink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def recover_done(root: Path, worker_id: int) -> set[str]:
    """unit_ids já persistidos por este worker (segmentos + WAL).

    Lê o WAL também: registros gravados depois do último flush estão só lá, e
    reprocessá-los seria pagar inferência duas vezes.
    """
    done: set[str] = set()
    run_dir = root / "runs"
    if run_dir.exists():
        for p in sorted(run_dir.glob(f"w{worker_id:03d}-*.parquet")):
            try:
                done.update(pq.read_table(p, columns=["unit_id"])["unit_id"].to_pylist())
            except Exception:
                # Segmento ilegível não deveria existir (rename é atômico), mas
                # se existir é melhor refazer o trabalho que abortar a run.
                p.unlink(missing_ok=True)
    wal = root / f"wal-{worker_id:03d}.jsonl"
    if wal.exists():
        for line in wal.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["unit_id"])
            except (json.JSONDecodeError, KeyError):
                continue  # última linha truncada pela queda
    return done


def absorb_wal(root: Path, worker_id: int, prompt_version: str, model: str) -> int:
    """Converte um WAL órfão em segmento Parquet antes de retomar.

    Roda no início de cada worker: se a VM caiu entre o último flush e o
    fechamento, o WAL guarda registros que ainda não estão em Parquet. Sem isto
    eles ficariam invisíveis para o `collect`.
    """
    wal = root / f"wal-{worker_id:03d}.jsonl"
    if not wal.exists() or wal.stat().st_size == 0:
        return 0
    heal_partial_line(wal)
    records = []
    for line in wal.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    wal.unlink(missing_ok=True)  # consumido; o sink recria vazio
    if not records:
        return 0
    with ParquetSink(root, worker_id, prompt_version, model, flush_every=10**9) as sink:
        for r in records:
            sink.append(r)
    return len(records)
