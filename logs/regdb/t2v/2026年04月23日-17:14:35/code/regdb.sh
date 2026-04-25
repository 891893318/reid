python3 main.py \
  --dataset regdb \
  --debug wsl \
  --arch resnet \
  --trial 1 \
  --stage1-epoch 50 \
  --stage2-epoch 120 \
  --milestones 50 70 \
  --lr 0.001 \
  --device 0 \
  --test-mode t2v \
  --num-workers 32 \
  --tri-weight 0.25 \
  --weak-weight 0.25 \
  --specific-warmup 12 \
  --remain-start 30 \
  --remain-warmup 10 \
  --bmpr-high-th 0.78 \
  --bmpr-mid-th 0.60 \
  --bmpr-margin-weight 0.35 \
  --bmpr-proto-weight 0.65 \
  --bmpr-topk 3


# python3 main.py \
# --dataset regdb \
# --debug wsl \
# --save-path regdb \
# --arch resnet \
# --trial 1 \
# --stage1-epoch 50 \
# --milestone 50 70 \
# --lr 0.00055 \
# --device 0 \
# --test-mode v2t \
# --num-workers 32
