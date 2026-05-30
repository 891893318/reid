#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash regdb.sh          # run t2v and v2t for trials 1..10
#   bash regdb.sh t2v      # run t2v for trials 1..10
#   bash regdb.sh v2t      # run v2t for trials 1..10
#   DEVICE=2 bash regdb.sh t2v
#   START_TRIAL=3 END_TRIAL=5 bash regdb.sh both

mode="${1:-both}"
device="${DEVICE:-3}"
start_trial="${START_TRIAL:-1}"
end_trial="${END_TRIAL:-10}"

common_args=(
  --dataset regdb
  --debug wsl
  --save-path regdb
  --arch resnet
  --stage1-epoch 50
  --milestones 50 70
  --lr 0.00055
  --device "${device}"
  --num-workers 32
  --cre-sample-rate 1.0
  --enable-rgmfd 1
  --rgmfd-rel-weight 0.2
  --rgmfd-orth-weight 0.05
  --rgmfd-gate-reg-weight 0.01
)

run_one() {
  local test_mode="$1"
  local trial="$2"

  echo "Running RegDB ${test_mode}, trial ${trial}"
  python3 main.py \
    "${common_args[@]}" \
    --test-mode "${test_mode}" \
    --trial "${trial}"
}

run_mode() {
  local test_mode="$1"
  local trial

  for trial in $(seq "${start_trial}" "${end_trial}"); do
    run_one "${test_mode}" "${trial}"
  done
}

case "${mode}" in
  t2v|v2t)
    run_mode "${mode}"
    ;;

  both)
    run_mode t2v
    run_mode v2t
    ;;

  *)
    echo "Usage: bash regdb.sh {t2v|v2t|both}"
    exit 2
    ;;
esac
