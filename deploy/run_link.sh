#!/usr/bin/env bash
# Etapa link numa GPU alugada: Ollama + modelo leve + link-run.
#
#   ./deploy/run_link.sh [args do link-run]      # ex.: --limit 200 --order hash
#
# Pré-requisito: data/curated/entities/v=<versão>/{candidates.parquet,reference.json}
# enviados do Mac (link-prep não usa GPU e roda lá em ~20 s). Retomável: rodar
# de novo pula as chaves já respondidas.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=deploy/lib.sh
source "$(dirname "$0")/lib.sh"

MODEL="${NM_LINK_MODEL:-qwen2.5:7b-instruct}"
# O prompt de link tem ~900 tokens e a resposta ~40: cabem muito mais slots
# que os 16 da extração (KV de um 7B em q8_0 a 3072 tokens: ~90 MB por slot).
CONCURRENCY="${NM_CONCURRENCY:-32}"
# start_ollama lê NM_NUM_CTX; sem ele derivaria o contexto do prompt de
# EXTRAÇÃO (~6.6k) e reservaria KV à toa.
export NM_NUM_CTX="${NM_NUM_CTX:-3072}"

require_python python3
ensure_venv python3
ensure_ollama
start_ollama "${CONCURRENCY}"
ollama pull "${MODEL}"
curl -sf http://127.0.0.1:11434/api/chat \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"stream\":false,\"keep_alive\":-1,\"options\":{\"num_ctx\":${NM_NUM_CTX}}}" >/dev/null
require_gpu_offload

set +e
"${PY}" -m newsmanager.extract link-run --model "${MODEL}" --concurrency "${CONCURRENCY}" \
  --num-ctx "${NM_NUM_CTX}" "$@"
rc=$?
echo "EXIT=${rc}"
exit "${rc}"
