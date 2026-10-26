#!/usr/bin/env bash
# Seção 18, fase 2: holdout (2019-2023, uma vez), placebos, robustez e avaliação.
# Só depois de registrar as escolhas do dev na seção 18.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
G="$PY -m newsmanager.grafo"
$G holdout
$G placebos
$G robustez
$G avaliar
echo FIM fase 2
