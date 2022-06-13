#! /bin/bash

DATASET=dscript
ARCH=naive_ppi
CRITEION=ppi_crossentropy

# without ppm
fairseq-train \
    --user-dir module \
    --save-dir ./save/${DATASET}/${ARCH}-woppm \
    --seed 100 \
    \
    --optimizer adam \
    --lr 3e-5 \
    --batch-size 32 \
    --max-epoch 10 \
    \
    --data-dir ./data/${DATASET}/processed \
    --train-subset human_train \
    --valid-subset human_test \
    --max-len 800 \
    \
    --task ppi \
    --arch ${ARCH} \
    --criterion ${CRITEION} \
    \
    --dropout 0.2 \
    --emb-dim 1024 \
    --hid-dim 256   \
    --wo-ppm

# with ppm
fairseq-train \
    --user-dir module \
    --save-dir ./save/${DATASET}/${ARCH} \
    --seed 100 \
    \
    --optimizer adam \
    --lr 3e-5 \
    --batch-size 32 \
    --max-epoch 5 \
    \
    --data-dir ./data/${DATASET}/processed \
    --train-subset human_train \
    --valid-subset human_test \
    --max-len 800 \
    \
    --task ppi \
    --arch ${ARCH} \
    --criterion ${CRITEION} \
    \
    --dropout 0.2 \
    --emb-dim 1024 \
    --hid-dim 256
