#! /bin/bash

DATASET=dscript
ARCH=baseline
CRITEION=ppi_contrastive

fairseq-train \
    --user-dir module \
    --save-dir ./save/${DATASET}/${ARCH}-${CRITEION} \
    --seed 100 \
    \
    --optimizer adam \
    --lr 3e-5 \
    --batch-size 32 \
    --max-epoch 3 \
    \
    --data-dir ./data/${DATASET}/processed \
    --train-subset human_train \
    --valid-subset human_test \
    --max-len 800 \
    \
    --task ppi \
    --arch ${ARCH} \
    --criterion ${CRITEION} \
    --gamma 0.2 \
    \
    --dropout 0.2 \
    --emb-dim 1024 \
    --hid-dim 256 \
    --cnn-layers 3 \
    --kernel-size 3 \
    --trans-layers 6

# batch-size 32时精度和召回率还可以，64时精度不太行, batch-size 对性能的影响有点严重
