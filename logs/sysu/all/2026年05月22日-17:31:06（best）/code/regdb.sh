python3 main.py \
--dataset regdb \
--debug wsl \
--save-path regdb \
--arch resnet \
--trial 1 \
--stage1-epoch 50 \
--milestone 50 70 \
--lr 0.00055 \
--device 0 \
--test-mode t2v \
--num-workers 32

python3 main.py \
--dataset regdb \
--debug wsl \
--save-path regdb \
--arch resnet \
--trial 1 \
--stage1-epoch 50 \
--milestone 50 70 \
--lr 0.00055 \
--device 0 \
--test-mode v2t \
--num-workers 32


