#!/bin/bash

# Shared SYSU/LLCM innovation defaults live in main.py.
# Keep only SYSU-specific experiment knobs here.
python3 main.py \
    --dataset sysu \
    --device 1 \
    --search-mode all \
    --stage1-epoch 20 \
    --milestones 30 90 \
    --rgmfd-start-epoch 0 \
    --uprt-specific-ce-weight 0.015

# --model-path /root/WSL_ReID/logs/sysu/all/<run_time>/models/stage1/model_20.pth
