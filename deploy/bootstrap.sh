#!/usr/bin/env bash
# Provision a fresh rented GPU VM for extraction work.
#
#   ./bootstrap.sh <worker_id> <n_workers> [model]
#
# Installs Ollama, pulls the model, configures it for batch throughput, and
# verifies the GPU is actually being used. Idempotent -- safe to re-run.
set -euo pipefail
# shellcheck source=deploy/lib.sh
source "$(dirname "$0")/lib.sh"

WORKER_ID="${1:?usage: bootstrap.sh <worker_id> <n_workers> [model]}"
N_WORKERS="${2:?usage: bootstrap.sh <worker_id> <n_workers> [model]}"
# Mesmo padrão do run_all.sh, run_worker.sh e da CLI. Quando divergiam, o
# bootstrap baixava um modelo e a extração pedia outro, que não estava na VM.
MODEL="${3:-deepseek-r1:14b}"
MAX_CHARS="${NM_MAX_CHARS:-8000}"

# Ollama batches concurrent requests into one forward pass; this is where most
# of the throughput on a rented GPU comes from. 8 suits a 24 GB card with a 7B
# model at 4k context -- run `nm-extract sweep` to find the real plateau.
NUM_PARALLEL="${NUM_PARALLEL:-8}"

echo "=== worker ${WORKER_ID}/${N_WORKERS}, model ${MODEL} ==="

require_python python3

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found. Ollama will fall back to CPU, which is" >&2
  echo "         roughly two orders of magnitude slower. Check the GPU driver." >&2
else
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

echo "--- ollama ---"
ensure_ollama

# Keep the model resident and allow concurrent batching; MAX_LOADED_MODELS=1
# stops a second model from evicting this one mid-run. Works with and without
# systemd (see lib.sh).
cd "$(dirname "$0")/.."
echo "--- starting ollama (NUM_PARALLEL=${NUM_PARALLEL}) ---"
start_ollama "${NUM_PARALLEL}"

echo "--- pulling ${MODEL} ---"
ollama pull "${MODEL}"

echo "--- python deps ---"
ensure_venv python3

echo "--- verifying GPU offload ---"
# Warm the model, then check it actually landed on the GPU. A model silently
# running on CPU is the single most expensive failure mode here: the run still
# works, just ~100x slower, and you pay for every hour of it.
curl -sf http://127.0.0.1:11434/api/generate \
  -d "{\"model\":\"${MODEL}\",\"prompt\":\"ok\",\"stream\":false,\"keep_alive\":-1,\"options\":{\"num_ctx\":${NM_NUM_CTX:-4096}}}" >/dev/null
# Catches partial offload ("51%/49% CPU/GPU") too, which is the common case
# when the model *almost* fits and which a test for "100% CPU" let through.
require_gpu_offload

cat > "$HOME/worker.env" <<EOF
export NM_WORKER_ID=${WORKER_ID}
export NM_WORKERS=${N_WORKERS}
export NM_MODEL=${MODEL}
export NM_CONCURRENCY=${NUM_PARALLEL}
# Entra no prompt_version: o worker e o collect precisam do mesmo valor.
export NM_MAX_CHARS=${MAX_CHARS}
export OLLAMA_HOST=http://127.0.0.1:11434
# venv primeiro no PATH: "python3 -m newsmanager..." do RUNBOOK usa as deps dele.
export PATH="$(pwd)/.venv/bin:\$PATH"
export PY="$(pwd)/.venv/bin/python"
EOF

echo
echo "=== ready ==="
echo "  source ~/worker.env"
echo "  python3 -m newsmanager.extract sweep      # find best concurrency"
echo "  python3 -m newsmanager.extract bench      # measure throughput"
echo "  ./deploy/run_worker.sh                    # start extraction"
