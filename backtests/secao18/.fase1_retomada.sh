#!/usr/bin/env bash
# Seção 18 de docs/metodo_covariancia.md, fase 1: tudo até as escolhas do dev.
# Pré-requisito: setores dos membros do S&P 500
#   NM_SEC_USER_AGENT="Nome e-mail" .venv/bin/python -m newsmanager sectors --sp500
# Depois desta fase: as escolhas (logs/secao18_escolhas.json) vão para uma nota
# na seção 18 ANTES de rodar a fase 2 (holdout).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
G="$PY -m newsmanager.grafo"


$G anonimizacao
$G diadico-painel
$G descritivas
$G diadico
# robustez diádica (sem critério): meia-vida e normalização, sem QAP
for tau in 30 180 365; do
  $G diadico-painel --tau $tau && $G diadico --tag "sp500_tau=${tau}_norm=idf" --qap 0
done
for nm in bruta grau pmi z; do
  $G diadico-painel --norm $nm && $G diadico --tag "sp500_tau=90_norm=${nm}" --qap 0
done
$G fomc
$G g4
$G dev
echo FIM fase 1
