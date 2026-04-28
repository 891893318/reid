python3 main.py \
--dataset llcm \
--debug wsl \
--save-path llcm \
--arch resnet \
--stage1-epoch 80 \
--milestone 30 70 \
--lr 0.0003 \
--device 1 \
--test-mode t2v 

python3 main.py \
--dataset llcm \
--debug wsl \
--save-path llcm \
--arch resnet \
--stage1-epoch 80 \
--milestone 30 70 \
--lr 0.0003 \
--device 1 \
--test-mode v2t 

