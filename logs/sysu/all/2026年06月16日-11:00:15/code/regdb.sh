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
    --milestone 50 70 \
    --lr 0.00055 \
    --device 3 \
    --test-mode t2v \
    --num-workers 32 \
    --cre-sample-rate 1.0 \
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
    --milestone 50 70 \
    --lr 0.00055 \
    --device 3 \
    --test-mode v2t \
    --num-workers 32 \
    --cre-sample-rate 1.0 \
    --enable-rgmfd 1 \
    --rgmfd-orth-weight 0.05 \
    --rgmfd-gate-reg-weight 0.01
done