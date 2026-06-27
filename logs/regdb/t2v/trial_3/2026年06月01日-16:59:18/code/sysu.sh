python3 main.py \
--dataset sysu \
--debug wsl \
--save-path sysu_agw \
--arch resnet \
--stage1-epoch 20 \
--milestone 30 70 \
--lr 0.0003 \
--device 0 \
--search-mode indoor \
--cre-sample-rate 1.0 \
--enable-remain-gate 0 \
--enable-rgmfd 1 \
--rgmfd-rel-weight 0.1 \
--rgmfd-orth-weight 0.05 \
--rgmfd-gate-reg-weight 0.01 \
--enable-crccl 1 \
--crccl-start-epoch 20 \
--crccl-cm-weight 0.03 \
--crccl-shared-weight 0.02 \
--crccl-cf-weight 0.003 \
--crccl-min-weight 0.2 \
--crccl-replace-cmo 1 \
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
