#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash sysu.sh all
#   bash sysu.sh indoor
#   DEVICE=2 bash sysu.sh indoor

mode="${1:-all}"
device="${DEVICE:-1}"

common_args=(
  --dataset sysu
  --debug wsl
  --save-path sysu_agw
  --arch resnet
  --stage1-epoch 20
  --milestones 30 70
  --lr 0.0003
  --device "${device}"
  --cre-sample-rate 1.0
  --enable-trrm 0
  --enable-remain-gate 0
  --enable-rdl 0
  --enable-rgmfd 1
#   --rgmfd-start-epoch 20
  --rgmfd-rel-weight 0.1
  --rgmfd-orth-weight 0.05
  --rgmfd-gate-reg-weight 0.01
  --model-path /root/saved_pretrain_sysu_resnet/model_20.pth
)

case "${mode}" in
  all)
    python3 main.py \
      "${common_args[@]}" \
      --search-mode all
    ;;

  indoor)
    python3 main.py \
      "${common_args[@]}" \
      --search-mode indoor
    ;;

  both)
    bash "$0" all
    bash "$0" indoor
    ;;

  *)
    echo "Usage: bash sysu.sh {all|indoor|both}"
    exit 2
    ;;
esac
