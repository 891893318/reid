python3 main.py \
  --dataset regdb \
  --debug wsl \
  --arch resnet \
  --trial 1 \
  --stage1-epoch 50 \
  --stage2-epoch 120 \
  --milestones 50 70 \
  --lr 0.00055 \
  --device 0 \
  --test-mode t2v \
  --num-workers 32 \
  --tri-weight 0.25 \
  --weak-weight 0.25 \
  --bcpt-pred-weight 0.45 \
  --bcpt-feat-weight 0.40 \
  --bcpt-conf-weight 0.15 \
  --bcpt-ot-reg 0.05 \
  --bcpt-ot-iters 40 \
  --bcpt-high-th 0.45 \
  --bcpt-mid-th 0.20 \
  --bcpt-topk 3


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

