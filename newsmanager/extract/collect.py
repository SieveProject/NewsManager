"""Consolida os segmentos Parquet dos workers em duas tabelas consultáveis.

Como o worker já grava Parquet, aqui não há conversão de formato -- só união,
deduplicação e registro das views. Isso significa que os resultados são
consultáveis *durante* a run, sem esperar o fim:

    SELECT count(*) FROM read_parquet('data/extractions/v=<versão>/relations/*.parquet');

Produz:
  extraction_runs  uma linha por unidade tentada -- status, tokens, latência.
                   Guardar as falhas é o que torna o reprocessamento seletivo e
                   permite auditar a cobertura em vez de presumi-la.
  relations        uma linha por tupla extraída, com a data da notícia.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..ingest import connect


def shard_dir(cfg: Config, prompt_version: str) -> Path:
    return cfg.root / "extractions" / f"v={prompt_version}"


def collect(cfg: Config, prompt_version: str) -> dict:
    src = shard_dir(cfg, prompt_version)
    run_seg = sorted((src / "runs").glob("*.parquet"))
    rel_seg = sorted((src / "relations").glob("*.parquet"))
    if not run_seg:
        raise RuntimeError(f"nenhum segmento em {src / 'runs'}; rode a extração primeiro")

    out_dir = cfg.curated / "relations" / f"v={prompt_version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    con = connect(cfg, memory_limit="8GB")

    # Uma unidade pode aparecer duas vezes se a VM caiu entre gravar o segmento
    # e truncar o WAL. Mantém-se uma linha por unidade, preferindo o sucesso.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW runs_dedup AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, row_number() OVER (
                PARTITION BY unit_id ORDER BY (status = 'ok') DESC, extracted_at DESC
            ) rn
            FROM read_parquet('{src / "runs" / "*.parquet"}', union_by_name=true)
        ) WHERE rn = 1
        """
    )

    runs_path = out_dir / "extraction_runs.parquet"
    con.execute(
        f"COPY (SELECT * FROM runs_dedup) TO '{runs_path}' "
        f"(FORMAT parquet, COMPRESSION '{cfg.compression}')"
    )

    rel_path = out_dir / "relations.parquet"
    if rel_seg:
        # A mesma unidade pode ter tuplas em dois segmentos (retomada após
        # queda); mantém-se apenas as da tentativa que sobreviveu ao dedup.
        con.execute(
            f"""
            COPY (
                SELECT r.* FROM (
                    SELECT * EXCLUDE (rn) FROM (
                        SELECT *, row_number() OVER (
                            PARTITION BY unit_id, agent_a, agent_b, relation_type
                            ORDER BY extracted_at DESC
                        ) rn
                        FROM read_parquet('{src / "relations" / "*.parquet"}', union_by_name=true)
                    ) WHERE rn = 1
                ) r
                SEMI JOIN runs_dedup u ON r.unit_id = u.unit_id
            ) TO '{rel_path}' (FORMAT parquet, COMPRESSION '{cfg.compression}')
            """
        )
    else:
        con.execute(
            f"COPY (SELECT * FROM read_parquet('{src / 'runs' / '*.parquet'}') LIMIT 0) "
            f"TO '{rel_path}' (FORMAT parquet)"
        )

    n_runs, n_ok, n_failed = con.execute(
        f"""SELECT count(*), sum((status='ok')::INT), sum((status<>'ok')::INT)
            FROM read_parquet('{runs_path}')"""
    ).fetchone()
    n_rel = con.execute(f"SELECT count(*) FROM read_parquet('{rel_path}')").fetchone()[0]
    con.close()

    print(
        f"[collect] {len(run_seg)} segmento(s) -> {n_runs:,} unidades "
        f"({n_ok:,} ok, {n_failed:,} falhas), {n_rel:,} relações\n"
        f"[collect] {runs_path}\n[collect] {rel_path}"
    )
    return {"segments": len(run_seg), "units": n_runs, "ok": n_ok, "failed": n_failed, "relations": n_rel}


def build_views(cfg: Config, prompt_version: str) -> None:
    """Registra as saídas da extração como views no banco de consulta."""
    import duckdb

    out_dir = cfg.curated / "relations" / f"v={prompt_version}"
    con = duckdb.connect(str(cfg.db))
    con.execute("CREATE SCHEMA IF NOT EXISTS news")
    con.execute(
        f"CREATE OR REPLACE VIEW news.relations AS "
        f"SELECT * FROM read_parquet('{out_dir / 'relations.parquet'}')"
    )
    con.execute(
        f"CREATE OR REPLACE VIEW news.extraction_runs AS "
        f"SELECT * FROM read_parquet('{out_dir / 'extraction_runs.parquet'}')"
    )
    # As relações voltam a se abrir para cada ticker sob o qual a matéria foi
    # arquivada -- é a chave de junção com um painel de preços.
    con.execute(
        """
        CREATE OR REPLACE VIEW news.relations_by_symbol AS
        SELECT s.symbol, r.* EXCLUDE (symbols)
        FROM news.relations r, UNNEST(r.symbols) AS s(symbol)
        """
    )
    # Painel diário agente-a-agente: a forma em que a extração encontra o preço.
    con.execute(
        """
        CREATE OR REPLACE VIEW news.agent_edges_daily AS
        SELECT CAST(published_at AS DATE) AS dt,
               agent_a, agent_b, direction,
               count(*)                                        AS n,
               avg(strength)                                   AS mean_strength,
               sum(CASE WHEN direction='positive' THEN strength
                        ELSE -strength END)                    AS signed_strength
        FROM news.relations
        WHERE published_at IS NOT NULL
        GROUP BY 1,2,3,4
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW news.agent_edges AS
        SELECT agent_a, agent_b, direction, relation_type,
               count(*) AS n, avg(strength) AS mean_strength,
               min(published_at) AS first_seen, max(published_at) AS last_seen
        FROM news.relations
        GROUP BY 1,2,3,4
        """
    )
    con.close()
    print(f"[collect] views em {cfg.db}: news.relations, news.relations_by_symbol, "
          f"news.agent_edges, news.agent_edges_daily")
