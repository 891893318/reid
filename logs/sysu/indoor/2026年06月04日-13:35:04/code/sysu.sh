python3 main.py \
--dataset sysu \
--debug wsl \
--save-path sysu_agw \
--arch resnet \
--stage1-epoch 20 \
--milestone 30 70 \
--lr 0.0003 \
--device 1 \
--search-mode indoor \
--cre-sample-rate 1.0 \
--enable-rgmfd 1 \
--rgmfd-rel-weight 0.1 \
--rgmfd-orth-weight 0.05 \
--rgmfd-gate-reg-weight 0.01 \
--enable-rgspc 1 \
--rgspc-start-epoch 40 \
--rgspc-weight 0.03 \
--rgspc-temperature 0.07 \
--rgspc-hard-k 16 \
--rgspc-hard-weight 0.05 \
# --model-path /root/WSL_ReID/logs/sysu/indoor/2026年06月03日-15:16:30/models/stage1/model_20.pth

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
