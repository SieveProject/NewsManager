#!/usr/bin/env bash
# Seção 16 de docs/metodo_covariancia.md (Black-Litterman na covariância).
# Roda as 14 configurações (n = 100) e a confirmação da GMV em n = 50 e
# n = 200 (setor, BL, placebo). Depois: python backtests/secao16/avaliar.py
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
$PY -m newsmanager.backtester run backtests/secao16/*.toml
for c in setor bl placebo; do
  for n in 50 200; do
    $PY -m newsmanager.backtester run "backtests/secao16/s16_gmv_${c}.toml" \
      --set "universe.n=${n}" --name "s16_gmv_${c}_n${n}"
  done
done
echo FIM
