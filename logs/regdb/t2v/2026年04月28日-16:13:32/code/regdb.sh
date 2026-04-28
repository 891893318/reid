python3 main.py \
--dataset regdb \
--debug wsl \
--save-path regdb \
--arch resnet \
--trial 1 \
--stage1-epoch 50 \
--phase2-pseudo-warmup 40 \
--phase2-pseudo-start-weight 0.2 \
--phase2-pseudo-max-weight 0.5 \
--phase2-adaptive-pseudo 1 \
--phase2-pseudo-quality-floor 0.5 \
--milestones 50 70 \
--lr 0.00055 \
--device 0 \
--test-mode t2v \
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
--cmo-weight 0.3 \
--scrc-cm-weight 0.0 \
--scrc-bi-weight 0.0 \
--scrc-proto-weight 0.0 \
--num-workers 32

# python3 main.py \
# --dataset regdb \
# --debug wsl \
# --save-path regdb \
# --arch resnet \
# --trial 1 \
# --stage1-epoch 50 \
# --milestones 50 70 \
# --lr 0.00055 \
# --device 0 \
# --test-mode v2t \
# --enable-fimro 1 \
# --fimro-alpha 0.02 \
# --fimro-beta 0.02 \
# --fimro-low-ratio 0.25 \
# --fimro-low-noise 0.15 \
# --fimro-fuse-scale 0.0 \
# --fimro-mask-mode square \
# --enable-soft-relation 1 \
# --soft-relation-temp 0.07 \
# --soft-lambda-v2r 0.4 \
# --soft-lambda-r2v 0.4 \
# --soft-lambda-proto 0.2 \
# --soft-cm-weight 0.05 \
# --num-workers 32
