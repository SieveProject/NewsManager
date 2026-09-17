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

if ! command -v ollama >/dev/null 2>&1; then
  echo "--- installing ollama ---"
  curl -fsSL https://ollama.com/install.sh | sh
fi

# Keep the model resident and allow concurrent batching. OLLAMA_MAX_LOADED_MODELS=1
# stops a second model from evicting this one mid-run.
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<EOF
[Service]
Environment="OLLAMA_NUM_PARALLEL=${NUM_PARALLEL}"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_FLASH_ATTENTION=1"
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now ollama
sleep 3
sudo systemctl restart ollama

echo "--- waiting for ollama ---"
for _ in $(seq 1 60); do
  curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break
  sleep 2
done
curl -sf http://127.0.0.1:11434/api/tags >/dev/null || { echo "ollama did not come up" >&2; exit 1; }

echo "--- pulling ${MODEL} ---"
ollama pull "${MODEL}"

echo "--- python deps ---"
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r "$(dirname "$0")/../requirements.txt"

echo "--- verifying GPU offload ---"
# Warm the model, then check it actually landed on the GPU. A model silently
# running on CPU is the single most expensive failure mode here: the run still
# works, just ~100x slower, and you pay for every hour of it.
curl -sf http://127.0.0.1:11434/api/generate \
  -d "{\"model\":\"${MODEL}\",\"prompt\":\"ok\",\"stream\":false,\"keep_alive\":-1}" >/dev/null
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
EOF

echo
echo "=== ready ==="
echo "  source ~/worker.env"
echo "  python3 -m newsmanager.extract sweep      # find best concurrency"
echo "  python3 -m newsmanager.extract bench      # measure throughput"
echo "  ./deploy/run_worker.sh                    # start extraction"
