python3 main.py \
--dataset sysu \
--debug wsl \
--save-path sysu_agw \
--arch resnet \
--stage1-epoch 20 \
--milestones 30 70 \
--lr 0.0003 \
--device 0 \
--search-mode all \
--cre-sample-rate 1.0 \
--enable-uprt 1 \
--uprt-weight 0.1 \
--uprt-cls-weight 0.05 \
--uprt-cls-temperature 1.0 \
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
