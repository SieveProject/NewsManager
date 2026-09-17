#!/usr/bin/env bash
# Move o corpus para as VMs e os resultados de volta.
#
#   ./deploy/gather.sh push  <host> [host...]   # envia data/curated/ (worker 0 -> demais)
#   ./deploy/gather.sh pull  <host> [host...]   # traz os segmentos e consolida
#
# Por que este script existe: cada VM extrai só a sua partição e grava em
# data/extractions/ local. Os segmentos são nomeados w<worker>-<seq>.parquet, ou
# seja, já são únicos entre máquinas e podem ser unidos num diretório só -- mas
# nada os copia para lá. Sem este passo você termina uma run paga com N
# conjuntos parciais espalhados por N máquinas efêmeras, e destruir as VMs
# perde (N-1)/N do que foi pago.
#
# `pull` é incremental e idempotente: rode quantas vezes quiser, inclusive com
# workers ainda rodando, para ir tirando resultado da máquina antes do fim.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-}"; shift || true
REMOTE_DIR="${NM_REMOTE_DIR:-NewsManager}"
PY="${PY:-python3}"
RSYNC_OPTS=(-az --partial --info=stats1)

usage() { sed -n '2,12p' "$0" >&2; exit 2; }
[ -n "${MODE}" ] && [ "$#" -gt 0 ] || usage

case "${MODE}" in
  push)
    # As demais VMs não precisam do raw (bronze), só do curated e das unidades.
    [ -d data/curated/extraction_units ] || {
      echo "ERRO: data/curated/extraction_units não existe. Rode primeiro:" >&2
      echo "      $PY -m newsmanager all && $PY -m newsmanager.extract units" >&2
      exit 2; }
    for host in "$@"; do
      echo "=== push -> ${host} ==="
      ssh "${host}" "mkdir -p '${REMOTE_DIR}/data/curated'"
      rsync "${RSYNC_OPTS[@]}" \
        data/curated/documents data/curated/mentions data/curated/extraction_units \
        "${host}:${REMOTE_DIR}/data/curated/"
    done
    echo "corpus enviado para $# host(s)."
    ;;

  pull)
    mkdir -p data/extractions
    for host in "$@"; do
      echo "=== pull <- ${host} ==="
      # Traz também os wal-*.jsonl: os últimos registros de uma VM que terminou
      # e nunca mais vai reiniciar estão só neles, e `collect` só lê Parquet.
      rsync "${RSYNC_OPTS[@]}" "${host}:${REMOTE_DIR}/data/extractions/" data/extractions/ \
        || { echo "AVISO: falhou o pull de ${host}; os demais continuam." >&2; continue; }
    done
    echo
    echo "=== absorvendo WALs recolhidos ==="
    $PY -m newsmanager.extract absorb
    echo
    echo "=== consolidando ==="
    $PY -m newsmanager.extract collect ${NM_MAX_CHARS:+--max-chars "${NM_MAX_CHARS}"}
    ;;

  *) usage ;;
esac
