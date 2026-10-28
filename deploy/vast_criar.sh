#!/usr/bin/env bash
# Aluga UMA RTX 3090/4090 na Vast tentando as ofertas mais baratas em ordem.
# --cancel-unavail: se o host não agenda a GPU, a Vast devolve erro em vez de criar
# uma instância parada (o "success: False" que deixava instâncias presas em loading).
#   ./deploy/vast_criar.sh [disco_GB]
set -uo pipefail
DISK="${1:-40}"
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"   # a imagem das runs que funcionaram
Q='gpu_name in [RTX_3090,RTX_4090,RTX_3090_Ti] num_gpus=1 reliability>0.98 inet_down>300 disk_space>=40 cuda_vers>=12.0 rentable=true verified=true'

ids=$(vastai search offers "$Q" -o dph_total --limit 15 --raw | python3 -c '
import json, sys
for o in json.load(sys.stdin):
    if "CN" not in (o.get("geolocation") or ""):          # China: rede cortada numa run anterior
        print(o["id"], round(o["dph_total"], 3), (o.get("geolocation") or "?").replace(" ", "_"))')

while read -r id price geo; do
  [ -z "$id" ] && continue
  echo "--- oferta $id ($geo, US\$ $price/h)"
  out=$(vastai create instance "$id" --image "$IMAGE" --disk "$DISK" --ssh --direct --cancel-unavail \
        --label newsmanager --raw 2>&1)
  echo "$out" | tail -2
  if echo "$out" | grep -q '"success": *true'; then
    echo "OK: instância $(echo "$out" | python3 -c 'import json,sys; print(json.load(sys.stdin)["new_contract"])')"
    exit 0
  fi
done <<< "$ids"
echo "nenhuma oferta agendou" >&2
exit 1
