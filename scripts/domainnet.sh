#!/bin/bash

python main.py \
  -a resnet50 \
  --batch-size 32 \
  --mlp --aug-plus --cos \
  --data-A '/mnt/data1/ljy/data/domainnet/clipart' \
  --data-B '/mnt/data1/ljy/domainnet/sketch' \
  --num_cluster '7' \
  --warmup-epoch 20 \
  --temperature 0.2 \
  --exp-dir 'domainnet_clipart-sketch' \
  --lr 0.0001 \
  --clean-model '/mnt/data1/ljy/pre-trained/moco/moco_v2_800ep_pretrain.pth.tar' \
  --instcon-weight 1.0 \
  --cwcon-startepoch 20 \
  --cwcon-satureepoch 100 \
  --cwcon-weightstart 0.0 \
  --cwcon-weightsature 1.0 \
  --cwcon_filterthresh 0.2 \
  --epochs 200 \
  --selfentro-temp 0.1 \
  --selfentro-weight 0.5 \
  --selfentro-startepoch 100 \
  --distofdist-weight 0.1 \
  --distofdist-startepoch 100 \
  --prec_nums '50,100,200' \

