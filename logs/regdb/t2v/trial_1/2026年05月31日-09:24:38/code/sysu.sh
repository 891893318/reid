python3 main.py \
--dataset sysu \
--debug wsl \
--save-path sysu_agw \
--arch resnet \
--stage1-epoch 20 \
--milestone 30 70 \
--lr 0.0003 \
--device 2 \
--search-mode indoor \
--cre-sample-rate 1.0 \
--enable-remain-gate 0 \
--enable-rgmfd 1 \
--rgmfd-rel-weight 0.1 \
--rgmfd-orth-weight 0.05 \
--rgmfd-gate-reg-weight 0.01 \
--enable-p2r 1 \
--p2r-start-epoch 60 \
--p2r-refine-start-epoch 70 \
--p2r-anchor-weight 0.01 \
--p2r-soft-weight 0.003 \
--p2r-refine-weight 0.005 \
--p2r-unc-weight 0.001 \
--p2r-anchor-threshold 0.85 \
--p2r-min-anchors 32 \
--p2r-candidate-topk 5 \
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
