python3 main.py \
--dataset sysu \
--debug wsl \
--save-path sysu_agw \
--arch resnet \
--stage1-epoch 20 \
--milestone 30 70 \
--lr 0.0003 \
--device 2 \
--search-mode all \
--cre-sample-rate 1.0 \
--enable-trrm 0 \
--enable-remain-gate 0 \
--enable-rdl 0 \
--rgmfd-start-epoch 0 \
--enable-rgmfd 1 \
--rgmfd-rel-weight 0.1 \
--rgmfd-orth-weight 0.05 \
--rgmfd-gate-reg-weight 0.01 \
--uclae-weight 1.0 \
--uclae-temperature 1.0 \
--uclae-conflict-temperature 0.25 \
--uclae-min-teacher-confidence 0.2 \
--uclae-require-native-correct 1 \
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
