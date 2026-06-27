python3 main.py \
--dataset sysu \
--debug wsl \
--save-path sysu_agw \
--arch resnet \
--stage1-epoch 20 \
--milestones 30 90 \
--lr 0.0003 \
--device 0 \
--search-mode all \
--cre-sample-rate 1.0 \
--enable-uprt 1 \
--uprt-weight 0.1 \
--uprt-cls-weight 0.05 \
--uprt-cls-temperature 1.0 \
--uprt-proto-cls-weight 0.02 \
--uprt-proto-cls-temperature 1.0 \
--uprt-hard-weight 0.0 \
--uprt-hard-start-epoch 40 \
--uprt-hard-warmup-epochs 10 \
--uprt-hard-topk 20 \
--uprt-hard-temperature 0.07 \
--uprt-hard-min-confidence 0.85 \
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
--rgmfd-start-epoch 0 \
--enable-rgmfd 1 \
--rgmfd-orth-weight 0.05 \
--rgmfd-gate-reg-weight 0.01
# --model-path /root/WSL_ReID/logs/sysu/all/<run_time>/models/stage1/model_20.pth

# python3 main.py \
# --dataset sysu \
# --debug wsl \
# --save-path sysu_agw \
# --arch resnet \
# --stage1-epoch 20 \
# --milestone 30 70 \
# --lr 0.0003 \
# --device 2 \
# --search-mode indoor 
