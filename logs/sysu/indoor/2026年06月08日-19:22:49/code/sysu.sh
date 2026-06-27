python3 main.py \
--dataset sysu \
--debug wsl \
--save-path sysu_agw \
--arch resnet \
--stage1-epoch 20 \
--milestones 30 70 \
--lr 0.0003 \
--device 3 \
--search-mode indoor \
--cre-sample-rate 1.0 \
--enable-rgmfd 1 \
--rgmfd-start-epoch 0 \
--rgmfd-rel-weight 0.1 \
--rgmfd-orth-weight 0.05 \
--rgmfd-gate-reg-weight 0.01 \
--rank-hard-start 30 \
--rank-hard-weight 0.05 \
--rank-hard-margin 0.10 \
--rank-hard-topk 5 \
--rank-pos-weight 0.2
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
