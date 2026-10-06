#!/usr/bin/env bash
# Copia os resultados da extração para a nuvem, em loop, enquanto a run roda.
#
#   ./deploy/backup.sh                # a cada 10 min, para sempre
#   ./deploy/backup.sh --once         # uma passada só (ex.: no fim da run)
#
# Por que existe: a VM é alugada e efêmera. Sem isto, os resultados só existem
# no disco dela -- destruir a instância ou o crédito zerar perde tudo o que já
# foi pago. Os segmentos Parquet são imutáveis depois do rename, então cada
# passada só envia arquivos novos. Os wal-*.jsonl vão junto: guardam os até
# `checkpoint-every` registros que ainda não viraram segmento.
#
# `rclone copy` nunca apaga nada no destino. Perda máxima numa queda: o que foi
# gravado desde a última passada (BACKUP_EVERY segundos).
#
# Requer um remote rclone configurado (padrão `gdrive:`). Ver deploy/RUNBOOK.md.
set -uo pipefail
cd "$(dirname "$0")/.."

REMOTE="${NM_BACKUP_REMOTE:-gdrive:NewsManager}"
EVERY="${BACKUP_EVERY:-600}"
LOG="logs/backup.log"
mkdir -p logs

sync_once() {
  local t0 rc=0
  t0=$(date +%s)
  # Extrações cruas (segmentos + WALs): a fonte da verdade, sempre primeiro.
  rclone copy data/extractions "${REMOTE}/extractions" \
      --exclude "*.tmp" --transfers 8 --checkers 16 --retries 5 --low-level-retries 20 \
      >>"${LOG}" 2>&1 || rc=$?
  # Consolidado e logs: pequenos, úteis para inspecionar sem baixar tudo.
  if [ -d data/curated/relations ]; then
    rclone copy data/curated/relations "${REMOTE}/relations" --retries 5 >>"${LOG}" 2>&1 || rc=$?
  fi
  # ollama.log fica de fora: registra cada requisição e seria reenviado inteiro.
  rclone copy logs "${REMOTE}/logs" --exclude "backup.log" --exclude "ollama.log" --retries 3 >>"${LOG}" 2>&1 || true
  local n
  n=$(find data/extractions -name '*.parquet' -path '*/runs/*' 2>/dev/null | wc -l)
  if [ "${rc}" -eq 0 ]; then
    echo "$(date -u +%FT%TZ) OK  ${n} segmentos em ${REMOTE} ($(( $(date +%s) - t0 ))s)" >>"${LOG}"
  else
    # Falha não derruba o loop: a próxima passada tenta de novo, e o que não
    # subiu continua no disco. Mas fica registrado -- `grep FALHA logs/backup.log`.
    echo "$(date -u +%FT%TZ) FALHA rc=${rc} -- ver linhas acima; dados seguem no disco" >>"${LOG}"
  fi
  return "${rc}"
}

if [ "${1:-}" = "--once" ]; then
  sync_once
  exit $?
fi

echo "$(date -u +%FT%TZ) backup iniciado: ${REMOTE} a cada ${EVERY}s" >>"${LOG}"
while true; do
  sync_once || true
  sleep "${EVERY}"
done
