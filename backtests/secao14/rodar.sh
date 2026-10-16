#!/usr/bin/env bash
# Seção 14 de docs/metodo_covariancia.md. Roda as 20 configurações (n = 100),
# a confirmação em n = 50 e n = 200 (setor, notícias, placebo) e a robustez
# do Markowitz com γ = 2 e 10. Depois: python backtests/secao14/avaliar.py
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
$PY -m newsmanager.backtester run backtests/secao14/*.toml
for m in mv maxsharpe maxdiv hrp minte; do
  for c in setor noticias placebo; do
    for n in 50 200; do
      $PY -m newsmanager.backtester run "backtests/secao14/s14_${m}_${c}.toml" \
        --set "universe.n=${n}" --name "s14_${m}_${c}_n${n}"
    done
  done
done
for g in 2 10; do
  for c in setor noticias placebo; do
    $PY -m newsmanager.backtester run "backtests/secao14/s14_mv_${c}.toml" \
      --set "model.params.gamma=${g}" --name "s14_mv_${c}_g${g}"
  done
done
echo FIM
