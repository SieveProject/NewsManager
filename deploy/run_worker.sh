#!/usr/bin/env bash
# Roda a fatia desta VM da extração, sobrevivendo a quedas e reinícios.
#
#   ./run_worker.sh
#
# Lê ~/worker.env (escrito por bootstrap.sh). Reinicia o worker quando ele cai;
# como a retomada é exata, um reinício custa no máximo `checkpoint-every`
# inferências repetidas.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=deploy/lib.sh
source "$(dirname "$0")/lib.sh"
[ -f "$HOME/worker.env" ] && source "$HOME/worker.env"

: "${NM_WORKER_ID:?set NM_WORKER_ID (run bootstrap.sh first)}"
: "${NM_WORKERS:?set NM_WORKERS}"
# O mesmo padrão do run_all.sh e da CLI. Divergir aqui faria o bootstrap baixar
# um modelo e a extração pedir outro -- que então nem está na máquina.
: "${NM_MODEL:=deepseek-r1:14b}"
: "${NM_CONCURRENCY:=8}"
: "${NM_MAX_CHARS:=8000}"
: "${MAX_RESTARTS:=100}"
: "${RESTART_SLEEP:=15}"
PY="${PY:-python3}"

LOG="logs/worker-${NM_WORKER_ID}.log"
mkdir -p logs

# num_ctx é derivado de max_chars pela CLI quando não é passado. Só encaminha um
# valor explícito, senão um default fixo aqui volta a poder ficar pequeno demais
# assim que prompts/extraction.txt crescer.
CTX_ARG=()
[ -n "${NM_NUM_CTX:-}" ] && CTX_ARG=(--num-ctx "${NM_NUM_CTX}")

echo "worker ${NM_WORKER_ID}/${NM_WORKERS} model=${NM_MODEL} concurrency=${NM_CONCURRENCY} max_chars=${NM_MAX_CHARS}" | tee -a "$LOG"

attempt=0
while true; do
  set +e
  $PY -m newsmanager.extract run \
      --worker-id "${NM_WORKER_ID}" \
      --workers "${NM_WORKERS}" \
      --model "${NM_MODEL}" \
      --concurrency "${NM_CONCURRENCY}" \
      --max-chars "${NM_MAX_CHARS}" \
      "${CTX_ARG[@]}" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e

  if [ "${rc}" -eq 0 ]; then
    echo "worker ${NM_WORKER_ID} finished" | tee -a "$LOG"
    exit 0
  fi

  # Um erro de configuração falha exatamente igual em toda tentativa. Reiniciar
  # cem vezes com 15s de pausa gastaria ~25 minutos de GPU alugada provando o
  # mesmo ponto, então ele para aqui.
  if [ "${rc}" -eq "${EXIT_CONFIG}" ]; then
    echo "ERRO de configuração (exit ${rc}); reiniciar não resolve. Corrija e rode de novo." | tee -a "$LOG" >&2
    exit "${rc}"
  fi

  attempt=$((attempt + 1))
  if [ "${attempt}" -ge "${MAX_RESTARTS}" ]; then
    echo "desistindo após ${attempt} reinícios" | tee -a "$LOG" >&2
    exit 1
  fi
  echo "worker saiu com ${rc}; reiniciando (${attempt}/${MAX_RESTARTS}) em ${RESTART_SLEEP}s" | tee -a "$LOG"
  sleep "${RESTART_SLEEP}"
done
