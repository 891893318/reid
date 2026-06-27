python3 main.py \
--dataset llcm \
--debug wsl \
--save-path llcm \
--arch resnet \
--stage1-epoch 80 \
--milestone 30 70 \
--lr 0.0003 \
--device 1 \
--test-mode t2v \
--cre-sample-rate 1.0 \
--enable-rgmfd 1 \
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
# --rgmfd-start-epoch 20 \
# --enable-rgmfd 1 \
# --rgmfd-orth-weight 0.05 \
# --rgmfd-gate-reg-weight 0.01 \
# # --model-path /root/WSL_ReID/logs/llcm/v2t/<run_time>/models/stage1/model_80.pth
