#!/usr/bin/env bash
# Checks shared by the deploy scripts. Sourced, never executed.
#
# These exist because every one of them guards a failure that costs money on a
# rented machine rather than merely wasting time on a laptop.

# Exit code the Python CLIs use for "this will never work" (newsmanager.config).
# Distinct from 1, which is what a transient crash looks like.
EXIT_CONFIG=2

# newsmanager imports tomllib, which is 3.11+. Ubuntu 22.04 -- the most common
# rented-GPU image -- ships 3.10. Checked before the 9 GB model pull, so the VM
# fails in seconds instead of ten minutes.
require_python() {
  local py="${1:-python3}"
  if ! "${py}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    local ver; ver="$("${py}" -V 2>&1 || echo 'não encontrado')"
    cat >&2 <<EOM
ERRO: newsmanager exige Python 3.11+ (tomllib). Encontrado: ${ver}
      Ubuntu 22.04 vem com 3.10. Instale o 3.11 e aponte PY para ele:
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt install -y python3.11 python3.11-venv
        PY=python3.11 ./deploy/run_all.sh ...
EOM
    return 1
  fi
  echo "python: $("${py}" -V 2>&1)"
}

# A model that only *partly* fits runs the overflow on CPU. Ollama reports that
# in the PROCESSOR column as e.g. "51%/49% CPU/GPU" -- which a test for
# "100% CPU" waves straight through, and the run then costs several times more
# per article for the whole rental. Any mention of CPU is a stop.
require_gpu_offload() {
  local ps_out body
  ps_out="$(ollama ps 2>/dev/null || true)"
  echo "${ps_out}"
  body="$(echo "${ps_out}" | tail -n +2)"
  if [ -z "${body//[[:space:]]/}" ]; then
    echo "AVISO: 'ollama ps' não listou modelo carregado; offload não verificado." >&2
    return 0
  fi
  if echo "${body}" | grep -qi "cpu"; then
    cat >&2 <<'EOM'
ERRO: parte do modelo está na CPU (veja a coluna PROCESSOR acima).
      Numa GPU alugada isso custa a mesma hora e entrega uma fração do
      throughput -- é o erro mais caro possível aqui. Opções:
        - modelo ou quantização menor (deepseek-r1:7b)
        - reduzir NM_CONCURRENCY: cada requisição paralela ocupa KV cache
        - uma placa com mais VRAM
EOM
    return 1
  fi
}

# Start (or restart) Ollama configured for batch throughput.
#
#   start_ollama <num_parallel>
#
# Rented "VMs" are often containers (RunPod, Vast): root, no sudo, and systemd
# not running as PID 1 even though the systemctl binary exists. Testing for the
# binary alone sends those boxes into `sudo systemctl`, which fails under
# `set -e` before the model is ever pulled. /run/systemd/system is the check
# systemd itself documents for "booted with systemd".
#
# OLLAMA_CONTEXT_LENGTH is pinned because recent Ollama picks the default
# context from VRAM -- 32768 on a 24 GB card -- and reserves KV cache for
# NUM_PARALLEL x that. 8 x 32k needed 62 GB on a 4090 and spilled 62% to CPU.
# Any request that omits num_ctx (warm-ups, health checks) loads the model at
# this default, so it must match what the extraction actually uses.
start_ollama() {
  local parallel="${1:?start_ollama <num_parallel>}"
  local ctx="${NM_NUM_CTX:-4096}"
  local sudo=""
  [ "$(id -u)" -ne 0 ] && sudo="sudo"

  if [ -d /run/systemd/system ]; then
    ${sudo} mkdir -p /etc/systemd/system/ollama.service.d
    ${sudo} tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<EOC
[Service]
Environment="OLLAMA_NUM_PARALLEL=${parallel}"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_CONTEXT_LENGTH=${ctx}"
EOC
    ${sudo} systemctl daemon-reload
    ${sudo} systemctl enable ollama >/dev/null 2>&1 || true
    ${sudo} systemctl restart ollama
  else
    # Restart rather than reuse: a server left over from a previous call may
    # hold a different NUM_PARALLEL, and the env is only read at startup.
    pkill -x ollama 2>/dev/null && sleep 2 || true
    mkdir -p logs
    OLLAMA_NUM_PARALLEL="${parallel}" OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=-1 \
    OLLAMA_HOST=127.0.0.1:11434 OLLAMA_FLASH_ATTENTION=1 OLLAMA_CONTEXT_LENGTH="${ctx}" \
      nohup ollama serve >>logs/ollama.log 2>&1 &
  fi

  for _ in $(seq 1 60); do
    curl -sf http://127.0.0.1:11434/api/tags >/dev/null && return 0
    sleep 2
  done
  echo "ERRO: Ollama não subiu (veja logs/ollama.log ou journalctl -u ollama)" >&2
  return 1
}

# Python deps in a repo-local venv; sets PY to its interpreter.
#
# Ubuntu 24.04 (PEP 668) refuses a system-wide `pip install` with
# "externally-managed-environment" -- the scripts died there before ever
# reaching the GPU. A venv also keeps the rented image's Python untouched.
ensure_venv() {
  local base="${1:-python3}"
  if [ ! -x .venv/bin/python ]; then
    if ! "${base}" -m venv .venv 2>/dev/null; then
      rm -rf .venv
      local sudo=""; [ "$(id -u)" -ne 0 ] && sudo="sudo"
      local minor; minor="$("${base}" -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
      echo "--- instalando python${minor}-venv ---"
      ${sudo} apt-get update -qq && ${sudo} apt-get install -y -qq "python${minor}-venv"
      "${base}" -m venv .venv
    fi
  fi
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet -r requirements.txt
  PY="$(pwd)/.venv/bin/python"
  echo "python deps: ${PY}"
}

# Install Ollama if absent. Its installer ships .tar.zst archives and needs
# zstd, which minimal GPU images lack; it fails half-way without it.
ensure_ollama() {
  command -v ollama >/dev/null 2>&1 && return 0
  if ! command -v zstd >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
    local sudo=""; [ "$(id -u)" -ne 0 ] && sudo="sudo"
    ${sudo} apt-get update -qq && ${sudo} apt-get install -y -qq zstd pciutils
  fi
  curl -fsSL https://ollama.com/install.sh | sh
}
