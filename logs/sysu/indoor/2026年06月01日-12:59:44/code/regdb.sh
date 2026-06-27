#!/bin/bash

# 循环 trial 1~10
for trial in {2..10}
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
    --milestone 50 70 \
    --lr 0.00055 \
    --device 3 \
    --test-mode t2v \
    --num-workers 32 \
    --cre-sample-rate 1.0 \
    --enable-rgmfd 1 \
    --rgmfd-rel-weight 0.3 \
    --rgmfd-orth-weight 0.05 \
    --rgmfd-gate-reg-weight 0.01
done


