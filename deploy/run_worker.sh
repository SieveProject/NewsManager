#!/usr/bin/env bash
# Run this VM's share of the extraction, surviving disconnects and restarts.
#
#   ./run_worker.sh
#
# Reads ~/worker.env (written by bootstrap.sh). Restarts the worker on crash;
# resume means a restart costs at most `checkpoint-every` repeated inferences.
set -euo pipefail

cd "$(dirname "$0")/.."
[ -f "$HOME/worker.env" ] && source "$HOME/worker.env"

: "${NM_WORKER_ID:?set NM_WORKER_ID (run bootstrap.sh first)}"
: "${NM_WORKERS:?set NM_WORKERS}"
: "${NM_MODEL:=qwen2.5:7b-instruct}"
: "${NM_CONCURRENCY:=8}"
: "${NM_NUM_CTX:=4096}"
: "${MAX_RESTARTS:=100}"

LOG="logs/worker-${NM_WORKER_ID}.log"
mkdir -p logs

echo "worker ${NM_WORKER_ID}/${NM_WORKERS} model=${NM_MODEL} concurrency=${NM_CONCURRENCY}" | tee -a "$LOG"

attempt=0
until python3 -m newsmanager.extract run \
        --worker-id "${NM_WORKER_ID}" \
        --workers "${NM_WORKERS}" \
        --model "${NM_MODEL}" \
        --concurrency "${NM_CONCURRENCY}" \
        --num-ctx "${NM_NUM_CTX}" 2>&1 | tee -a "$LOG"; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge "$MAX_RESTARTS" ]; then
    echo "giving up after ${attempt} restarts" | tee -a "$LOG"
    exit 1
  fi
  echo "worker exited nonzero; restarting (${attempt}/${MAX_RESTARTS}) in 15s" | tee -a "$LOG"
  sleep 15
done

echo "worker ${NM_WORKER_ID} finished" | tee -a "$LOG"
