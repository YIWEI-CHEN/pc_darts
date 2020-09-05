#!/usr/bin/env bash

GPU=3
BATCH_SIZE=224
SEED=2
EPOCH=50
SEARCH_SPACE="darts"
EXP_PATH="exp/single/batch_size${BATCH_SIZE}_S${SEED}_space_${SEARCH_SPACE}_gpu${GPU}"


python train_search.py --data /home/yiwei/cifar10 --batch_size ${BATCH_SIZE} --gpu ${GPU} \
    --save ${EXP_PATH} --seed ${SEED}  --epochs ${EPOCH} --report_freq 25 \
    --search_space ${SEARCH_SPACE}