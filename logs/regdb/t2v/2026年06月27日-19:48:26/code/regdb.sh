#!/bin/bash

# Shared SYSU/LLCM/RegDB innovation defaults live in main.py.
# Keep only RegDB-specific experiment knobs here.
for mode in t2v v2t
do
    for trial in {1..10}
    do
        echo "====================================="
        echo " Running RegDB ${mode}, trial ${trial}"
        echo "====================================="
        python3 main.py \
            --dataset regdb \
            --device 3 \
            --test-mode "${mode}" \
            --trial "${trial}" \
            --stage1-epoch 50 \
            --stage2-epoch 120 \
            --milestones 50 70 \
            --lr 0.00055 \
            --num-workers 32 \
            --uprt-specific-ce-weight 0.015 \
            --rgmfd-start-epoch 0
    done
done
