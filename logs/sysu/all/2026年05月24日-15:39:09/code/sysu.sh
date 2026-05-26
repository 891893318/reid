python3 main.py \
--dataset sysu \
--debug wsl \
--save-path sysu_agw \
--arch resnet \
--stage1-epoch 20 \
--milestone 30 70 \
--lr 0.0003 \
--device 0 \
--search-mode all \
--cre-confidence 0 \
--enable-trrm 0 \
--enable-remain-gate 0 
# --model-path /root/WSL_ReID/logs/sysu/all/2026年05月22日-15:24:24/models/model_20.pth

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
