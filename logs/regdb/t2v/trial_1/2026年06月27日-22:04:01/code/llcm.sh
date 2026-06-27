#!/bin/bash

# Shared SYSU/LLCM innovation defaults live in main.py.
# Keep only LLCM-specific experiment knobs here.
python3 main.py \
    --dataset llcm \
    --device 0 \
    --test-mode v2t \
    --stage1-epoch 80 \
    --milestones 30 70 \
    --rgmfd-start-epoch 0 \
    --uprt-specific-ce-weight 0.0

# python3 main.py \
#     --dataset llcm \
#     --device 1 \
#     --test-mode v2t \
#     --stage1-epoch 80 \
#     --milestones 30 70 \
#     --rgmfd-start-epoch 0 \
#     --uprt-specific-ce-weight 0.0
