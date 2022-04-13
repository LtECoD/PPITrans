#! /bin/bash

DATASET=dscript
ARCH=baseline

fairseq-train \
    --user-dir module \
    --save-dir ./save/${DATASET}/${ARCH} \
    --seed 100 \
    \
    --optimizer adam \
    --lr 3e-5 \
    --batch-size 32 \
    --max-epoch 40 \
    \
    --data-dir ./data/${DATASET}/processed \
    --train-subset human_train \
    --valid-subset human_test \
    --max-len 800 \
    \
    --task ppi \
    --arch ${ARCH} \
    --criterion ppi_criterion \
    \
    --dropout 0.2 \
    --emb-dim 1024 \
    --hid-dim 256 \
    --cnn-layers 3 \
    --kernel-size 3 \
    --trans-layers 6

