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
