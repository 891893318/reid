python3 main.py \
--dataset llcm \
--debug wsl \
--save-path llcm \
--arch resnet \
--stage1-epoch 80 \
--milestone 30 70 \
--lr 0.0003 \
--device 0 \
--test-mode t2v \
--cre-sample-rate 1.0 \
--enable-trrm 0 \
--enable-remain-gate 0 \
--enable-rdl 1 \
--enable-rgmfd 1 \
--rgmfd-rel-weight 0.3 \
--rgmfd-orth-weight 0.05 \
--rgmfd-gate-reg-weight 0.01 \
--rgmfd-start-epoch 20 
# --model-path /root/WSL_ReID/logs/llcm/t2v/2026年05月27日-11:45:00/models/stage1/model_80.pth

# python3 main.py \
# --dataset llcm \
# --debug wsl \
# --save-path llcm \
# --arch resnet \
# --stage1-epoch 80 \
# --milestone 30 70 \
# --lr 0.0003 \
# --device 3 \
# --test-mode v2t \
# --cre-sample-rate 1.0 \
# --enable-trrm 0 \
# --enable-remain-gate 0 \
# --enable-rdl 1 \
# --rgmfd-start-epoch 20 \
# --enable-rgmfd 1 \
# --rgmfd-rel-weight 0.1 \
# --rgmfd-orth-weight 0.05 \
# --rgmfd-gate-reg-weight 0.01 \
# # --model-path /root/saved_pretrain_sysu_resnet/model_20.pth
