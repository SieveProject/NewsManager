#!/usr/bin/env bash
# Pipeline completo numa GPU alugada, do zero até as tuplas.
#
#   ./deploy/run_all.sh                    # máquina única
#   ./deploy/run_all.sh 0 4                # worker 0 de 4 máquinas
#
# Etapas: instala Ollama -> baixa o modelo -> ingere as notícias -> remove
# duplicatas -> extrai com o LLM -> consolida. Toda etapa é idempotente e
# retomável: se a VM cair, rode de novo o mesmo comando.
set -euo pipefail
cd "$(dirname "$0")/.."

WORKER_ID="${1:-0}"
N_WORKERS="${2:-1}"

# deepseek-r1:14b é o ponto de equilíbrio para uma GPU de 24 GB: 9 GB de pesos
# deixam espaço de KV cache para ~8 requisições concorrentes, que é de onde vem
# o throughput. O :32b (20 GB) cabe mas sufoca o batching e sai mais lento na
# prática. Em placas de 80 GB, use :32b ou :70b.
#
# ATENÇÃO: deepseek-v4-flash e deepseek-v4-pro NÃO servem aqui -- todas as tags
# publicadas são `:cloud`, ou seja, rodam na nuvem da Ollama e não na GPU que
# você alugou. Só a família r1 (e v3/v3.1) tem pesos locais de verdade.
MODEL="${NM_MODEL:-deepseek-r1:14b}"
CONCURRENCY="${NM_CONCURRENCY:-8}"
NUM_CTX="${NM_NUM_CTX:-4096}"
MAX_CHARS="${NM_MAX_CHARS:-8000}"
PY="${PY:-python3}"

log() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

log "0/6  ambiente"
$PY --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "AVISO: nvidia-smi ausente. Sem GPU a extração roda ~100x mais lenta," >&2
  echo "       e você paga por hora. Verifique o driver antes de continuar." >&2
fi
$PY -m pip install --quiet --upgrade pip
$PY -m pip install --quiet -r requirements.txt

log "1/6  Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
if command -v systemctl >/dev/null 2>&1; then
  sudo mkdir -p /etc/systemd/system/ollama.service.d
  sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<EOF
[Service]
Environment="OLLAMA_NUM_PARALLEL=${CONCURRENCY}"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_FLASH_ATTENTION=1"
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable --now ollama
  sudo systemctl restart ollama
else
  pgrep -x ollama >/dev/null || (OLLAMA_NUM_PARALLEL=$CONCURRENCY OLLAMA_KEEP_ALIVE=-1 ollama serve &>/tmp/ollama.log &)
fi

for _ in $(seq 1 60); do
  curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break
  sleep 2
done
curl -sf http://127.0.0.1:11434/api/tags >/dev/null || { echo "Ollama não subiu" >&2; exit 1; }

log "2/6  modelo ${MODEL}"
ollama pull "${MODEL}"
curl -sf http://127.0.0.1:11434/api/chat \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"stream\":false,\"think\":false,\"keep_alive\":-1}" >/dev/null
ollama ps
# Um modelo rodando 100% em CPU é o erro mais caro possível aqui: a run termina,
# só que ~100x mais devagar, e a hora de GPU é cobrada do mesmo jeito.
if ollama ps 2>/dev/null | grep -qi "100% cpu"; then
  echo "ERRO: o modelo caiu inteiro na CPU. Não inicie a extração." >&2
  echo "      Use uma quantização menor ou uma GPU com mais VRAM." >&2
  exit 1
fi

log "3/6  pipeline de notícias (download remoto + parquet)"
# Só o worker 0 constrói o corpus; as demais VMs esperam recebê-lo por rsync/S3.
if [ "${WORKER_ID}" = "0" ]; then
  $PY -m newsmanager all
else
  echo "worker ${WORKER_ID}: usando o corpus já sincronizado em data/curated/"
  [ -d data/curated/documents ] || { echo "data/curated ausente; sincronize do worker 0" >&2; exit 1; }
fi

log "4/6  deduplicação -> unidades de extração"
$PY -m newsmanager.extract units

log "5/6  extração  (worker ${WORKER_ID}/${N_WORKERS}, modelo ${MODEL})"
$PY -m newsmanager.extract partition -n "${N_WORKERS}"
# Sem thinking: em milhões de artigos a cadeia de raciocínio multiplica os
# tokens de saída sem melhorar o preenchimento de um schema fechado.
NM_WORKER_ID="${WORKER_ID}" NM_WORKERS="${N_WORKERS}" \
NM_MODEL="${MODEL}" NM_CONCURRENCY="${CONCURRENCY}" NM_NUM_CTX="${NUM_CTX}" \
  ./deploy/run_worker.sh

log "6/6  consolidação"
$PY -m newsmanager.extract collect --max-chars "${MAX_CHARS}"

log "pronto"
echo "duckdb data/news.duckdb"
echo "  SELECT published_at, agent_a, relation_type, direction, strength, agent_b"
echo "  FROM news.relations ORDER BY published_at DESC LIMIT 20;"
