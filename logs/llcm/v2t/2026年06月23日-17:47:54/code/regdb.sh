#!/bin/bash

# 循环 trial 1~10
for trial in {1..10}
do
    echo "====================================="
    echo " Running trial $trial (IR -> VIS, t2v)"
    echo "====================================="

    python3 main.py \
    --dataset regdb \
    --debug wsl \
    --save-path regdb \
    --arch resnet \
	    --trial $trial \
	    --stage1-epoch 50 \
	    --stage2-epoch 120 \
	    --milestones 50 70 \
	    --lr 0.00055 \
	    --device 3 \
	    --test-mode t2v \
	    --num-workers 32 \
	    --cre-sample-rate 1.0 \
	    --enable-uprt 1 \
	    --uprt-weight 0.1 \
	    --uprt-cls-weight 0.05 \
	    --uprt-cls-temperature 1.0 \
	    --uprt-proto-cls-weight 0.02 \
	    --uprt-proto-cls-temperature 1.0 \
	    --uprt-common-tri-weight 1.0 \
	    --uprt-common-tri-start-epoch 35 \
	    --uprt-common-tri-warmup-epochs 10 \
	    --uprt-cmo-weight 0.03 \
	    --uprt-cmo-start-epoch 50 \
	    --uprt-cmo-warmup-epochs 10 \
	    --uprt-cmo-temperature 2.0 \
	    --uprt-cmo-min-target-prob 0.05 \
	    --uprt-specific-ce-weight 0.015 \
	    --uprt-remain-ce-weight 0.0 \
	    --uprt-specific-ce-start-epoch 35 \
	    --uprt-remain-ce-start-epoch 70 \
	    --uprt-relation-ce-warmup-epochs 10 \
	    --uprt-specific-ce-strength 0.35 \
	    --uprt-remain-ce-strength 0.15 \
	    --uprt-specific-ce-min-target-prob 0.70 \
	    --uprt-remain-ce-min-target-prob 0.45 \
	    --uprt-relation-ce-temperature 1.0 \
	    --uprt-hard-weight 0.03 \
	    --uprt-hard-start-epoch 70 \
	    --uprt-hard-warmup-epochs 10 \
	    --uprt-hard-topk 20 \
	    --uprt-hard-temperature 0.07 \
	    --uprt-hard-min-confidence 0.90 \
	    --uprt-temperature 0.07 \
	    --uprt-shared-temperature 0.07 \
	    --uprt-topk 10 \
	    --uprt-epsilon 0.10 \
	    --uprt-tau 0.5 \
	    --uprt-min-mass 0.05 \
	    --uprt-expert-weight 0.10 \
	    --uprt-prior-weight 0.05 \
	    --uprt-specific-prior 0.50 \
	    --uprt-remain-prior 0.20 \
	    --uprt-recovery-weight 0.08 \
	    --uprt-recovery-start-epoch 40 \
	    --uprt-recovery-warmup-epochs 20 \
	    --uprt-recovery-target-coverage 0.95 \
	    --uprt-recovery-min-coverage 0.80 \
	    --uprt-recovery-min-entropy 0.08 \
	    --uprt-recovery-temperature 0.20 \
	    --uprt-recovery-topk 20 \
	    --uprt-start-epoch 5 \
	    --uprt-warmup-epochs 10 \
	    --enable-rgmfd 1 \
	    --rgmfd-orth-weight 0.05 \
	    --rgmfd-gate-reg-weight 0.01
done

# 循环 trial 1~10
for trial in {1..10}
do
    echo "====================================="
    echo " Running trial $trial (VIS -> IR, v2t)"
    echo "====================================="

    python3 main.py \
    --dataset regdb \
    --debug wsl \
    --save-path regdb \
    --arch resnet \
	    --trial $trial \
	    --stage1-epoch 50 \
	    --stage2-epoch 120 \
	    --milestones 50 70 \
	    --lr 0.00055 \
	    --device 3 \
	    --test-mode v2t \
	    --num-workers 32 \
	    --cre-sample-rate 1.0 \
	    --enable-uprt 1 \
	    --uprt-weight 0.1 \
	    --uprt-cls-weight 0.05 \
	    --uprt-cls-temperature 1.0 \
	    --uprt-proto-cls-weight 0.02 \
	    --uprt-proto-cls-temperature 1.0 \
	    --uprt-common-tri-weight 1.0 \
	    --uprt-common-tri-start-epoch 35 \
	    --uprt-common-tri-warmup-epochs 10 \
	    --uprt-cmo-weight 0.03 \
	    --uprt-cmo-start-epoch 50 \
	    --uprt-cmo-warmup-epochs 10 \
	    --uprt-cmo-temperature 2.0 \
	    --uprt-cmo-min-target-prob 0.05 \
	    --uprt-specific-ce-weight 0.015 \
	    --uprt-remain-ce-weight 0.0 \
	    --uprt-specific-ce-start-epoch 35 \
	    --uprt-remain-ce-start-epoch 70 \
	    --uprt-relation-ce-warmup-epochs 10 \
	    --uprt-specific-ce-strength 0.35 \
	    --uprt-remain-ce-strength 0.15 \
	    --uprt-specific-ce-min-target-prob 0.70 \
	    --uprt-remain-ce-min-target-prob 0.45 \
	    --uprt-relation-ce-temperature 1.0 \
	    --uprt-hard-weight 0.03 \
	    --uprt-hard-start-epoch 70 \
	    --uprt-hard-warmup-epochs 10 \
	    --uprt-hard-topk 20 \
	    --uprt-hard-temperature 0.07 \
	    --uprt-hard-min-confidence 0.90 \
	    --uprt-temperature 0.07 \
	    --uprt-shared-temperature 0.07 \
	    --uprt-topk 10 \
	    --uprt-epsilon 0.10 \
	    --uprt-tau 0.5 \
	    --uprt-min-mass 0.05 \
	    --uprt-expert-weight 0.10 \
	    --uprt-prior-weight 0.05 \
	    --uprt-specific-prior 0.50 \
	    --uprt-remain-prior 0.20 \
	    --uprt-recovery-weight 0.08 \
	    --uprt-recovery-start-epoch 40 \
	    --uprt-recovery-warmup-epochs 20 \
	    --uprt-recovery-target-coverage 0.95 \
	    --uprt-recovery-min-coverage 0.80 \
	    --uprt-recovery-min-entropy 0.08 \
	    --uprt-recovery-temperature 0.20 \
	    --uprt-recovery-topk 20 \
	    --uprt-start-epoch 5 \
	    --uprt-warmup-epochs 10 \
	    --enable-rgmfd 1 \
	    --rgmfd-orth-weight 0.05 \
	    --rgmfd-gate-reg-weight 0.01
done
