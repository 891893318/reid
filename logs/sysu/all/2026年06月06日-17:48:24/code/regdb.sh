python3 main.py \
--dataset regdb \
--debug wsl \
--save-path regdb \
--arch resnet \
--trial 1 \
--stage1-epoch 50 \
--milestone 50 70 \
--lr 0.00055 \
--device 3 \
--test-mode t2v \
--num-workers 32 \
--cre-sample-rate 1.0 \
--enable-rgmfd 1 \
--rgmfd-rel-weight 0.2 \
--rgmfd-orth-weight 0.05 \
--rgmfd-gate-reg-weight 0.01

python3 main.py \
--dataset regdb \
--debug wsl \
--save-path regdb \
--arch resnet \
--trial 1 \
--stage1-epoch 50 \
--milestone 50 70 \
--lr 0.00055 \
--device 3 \
--test-mode v2t \
--num-workers 32 \
--cre-sample-rate 1.0 \
--enable-rgmfd 1 \
--rgmfd-rel-weight 0.2 \
--rgmfd-orth-weight 0.05 \
--rgmfd-gate-reg-weight 0.01

