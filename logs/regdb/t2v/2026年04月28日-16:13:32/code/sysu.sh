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
--enable-fimro 1 \
--fimro-alpha 0.02 \
--fimro-beta 0.02 \
--fimro-low-ratio 0.25 \
--fimro-low-noise 0.15 \
--fimro-fuse-scale 0.0 \
--fimro-mask-mode square \
--enable-soft-relation 0 \
--soft-relation-temp 0.07 \
--soft-lambda-v2r 0.4 \
--soft-lambda-r2v 0.4 \
--soft-lambda-proto 0.2 \
--soft-cm-weight 0.05 \
--cmo-weight 0.5 \
--scrc-cm-weight 0.0 \
--scrc-bi-weight 0.0 \
--scrc-proto-weight 0.0 \
--stage1-epoch 30 \
--phase2-pseudo-warmup 40 \
--phase2-pseudo-start-weight 0.2 \
--phase2-pseudo-max-weight 0.6 \
--num-workers 32

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