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
# shellcheck source=deploy/lib.sh
source "$(dirname "$0")/lib.sh"

WORKER_ID="${1:-0}"
N_WORKERS="${2:-1}"

# qwen2.5:14b-instruct: modelo de instrução, sem raciocínio. Nos mesmos 200
# artigos e prompt, superou o deepseek-r1:14b em todas as medidas de qualidade
# (~2/3 contra ~1/4 de tuplas agente-agente válidas) e rodou 39% mais rápido.
# 9 GB de pesos + KV cache q8_0 de 16 slots cabem em 24 GB (20 GB medidos).
#
# ATENÇÃO: deepseek-v4-flash e deepseek-v4-pro NÃO servem aqui -- todas as tags
# publicadas são `:cloud`, ou seja, rodam na nuvem da Ollama e não na GPU que
# você alugou.
MODEL="${NM_MODEL:-qwen2.5:14b-instruct}"
CONCURRENCY="${NM_CONCURRENCY:-8}"
# num_ctx fica vazio de propósito: a CLI o deriva de MAX_CHARS + tamanho real
# do prompt. Um 4096 fixo aqui ficava só 50 tokens acima do mínimo (4046) e
# derrubava a run com exit 2 na primeira edição de prompts/extraction.txt.
NUM_CTX="${NM_NUM_CTX:-}"
MAX_CHARS="${NM_MAX_CHARS:-8000}"
PY="${PY:-python3}"

log() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

log "0/6  ambiente"
require_python "$PY"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "AVISO: nvidia-smi ausente. Sem GPU a extração roda ~100x mais lenta," >&2
  echo "       e você paga por hora. Verifique o driver antes de continuar." >&2
fi
ensure_venv "$PY"

log "1/6  Ollama"
ensure_ollama
start_ollama "${CONCURRENCY}"

log "2/6  modelo ${MODEL}"
ollama pull "${MODEL}"
curl -sf http://127.0.0.1:11434/api/chat \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"stream\":false,\"think\":false,\"keep_alive\":-1,\"options\":{\"num_ctx\":${NM_NUM_CTX:-$(derive_num_ctx)}}}" >/dev/null
# Pega também o offload *parcial* ("51%/49% CPU/GPU"), que é o caso comum
# quando o modelo quase cabe -- e que um teste por "100% CPU" deixava passar.
require_gpu_offload

log "3/6  pipeline de notícias (download remoto + parquet)"
# Esta etapa é limitada por rede, não por GPU: ~55 min em que a placa alugada
# fica ociosa sendo cobrada, e numa frota as outras N-1 VMs esperam por ela.
# Construa o corpus antes, numa máquina barata, e distribua com
# `./deploy/gather.sh push <host>...`; aí NM_SKIP_INGEST=1 pula tudo isto.
if [ -d data/curated/documents ] && [ "${NM_SKIP_INGEST:-}" != "0" ]; then
  echo "corpus já presente em data/curated/; pulando a ingestão."
elif [ "${NM_SKIP_INGEST:-}" = "1" ]; then
  echo "ERRO: NM_SKIP_INGEST=1 mas data/curated/documents não existe." >&2
  echo "      Envie o corpus do worker 0: ./deploy/gather.sh push <este-host>" >&2
  exit "${EXIT_CONFIG}"
elif [ "${WORKER_ID}" = "0" ]; then
  echo "AVISO: ingerindo 21,6 GB nesta máquina. Se ela tem GPU, você está" >&2
  echo "       pagando por ~1h de placa ociosa fazendo I/O de rede." >&2
  $PY -m newsmanager all
else
  echo "ERRO: worker ${WORKER_ID} não tem data/curated/. Sincronize do worker 0:" >&2
  echo "      ./deploy/gather.sh push <este-host>" >&2
  exit "${EXIT_CONFIG}"
fi

log "4/6  deduplicação -> unidades de extração"
# Se as unidades vieram no push, reconstruí-las só gasta tempo de placa parada.
# E é mais seguro reusá-las: todas as VMs têm de partir do MESMO conjunto, senão
# as partições discordam e unidades ficam sem dono (ou com dois).
if [ -d data/curated/extraction_units ] && [ "${NM_FORCE_UNITS:-}" != "1" ]; then
  echo "unidades já presentes; reusando (NM_FORCE_UNITS=1 força reconstrução)."
else
  $PY -m newsmanager.extract units
fi

log "5/6  extração  (worker ${WORKER_ID}/${N_WORKERS}, modelo ${MODEL})"
$PY -m newsmanager.extract partition -n "${N_WORKERS}"
# Sem thinking: em milhões de artigos a cadeia de raciocínio multiplica os
# tokens de saída sem melhorar o preenchimento de um schema fechado.
# MAX_CHARS entra no prompt_version, então worker e collect têm de receber o
# mesmo valor: com valores diferentes o collect procura num diretório que os
# workers nunca escreveram e a run parece ter produzido nada.
NM_WORKER_ID="${WORKER_ID}" NM_WORKERS="${N_WORKERS}" \
NM_MODEL="${MODEL}" NM_CONCURRENCY="${CONCURRENCY}" NM_NUM_CTX="${NUM_CTX}" \
NM_MAX_CHARS="${MAX_CHARS}" PY="${PY}" \
  ./deploy/run_worker.sh

log "6/6  consolidação"
# Cada VM só tem a própria partição. Consolidar aqui produziria N datasets
# parciais, cada um anunciando sucesso -- e destruir as VMs perderia (N-1)/N de
# uma run já paga. Com frota, a consolidação acontece uma vez, no coletor.
if [ "${N_WORKERS}" -eq 1 ]; then
  $PY -m newsmanager.extract collect --max-chars "${MAX_CHARS}" --model "${MODEL}"
else
  cat <<EOM

Worker ${WORKER_ID} terminou a SUA partição (1/${N_WORKERS} do total).
NÃO destrua esta VM antes de recolher os resultados. Na máquina coletora:

  NM_MAX_CHARS=${MAX_CHARS} ./deploy/gather.sh pull <host-0> ... <host-$((N_WORKERS - 1))>

Isso traz os segmentos de cada VM, absorve os WALs e consolida uma única vez.
EOM
  exit 0
fi

log "pronto"
echo "duckdb data/corpus.duckdb"
echo "  SELECT published_at, agent_a, relation_type, direction, strength, agent_b"
echo "  FROM news.relations ORDER BY published_at DESC LIMIT 20;"
