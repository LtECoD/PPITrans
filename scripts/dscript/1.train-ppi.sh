#! /bin/bash

DATASET=dscript
ARCH=ppi
CRITEION=ppi_crossentropy

# without ppm, without pooling
fairseq-train \
    --user-dir module \
    --save-dir ./save/${DATASET}/${ARCH}-woppm-wopool \
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
    --hid-dim 256 \
    --cnn-layers 3 \
    --kernel-size 3 \
    --trans-layers 6 \
    --wo-ppm \
    --wo-pool


# without ppm, with pooling
# fairseq-train \
#     --user-dir module \
#     --save-dir ./save/${DATASET}/${ARCH}-woppm \
#     --seed 100 \
#     \
#     --optimizer adam \
#     --lr 3e-5 \
#     --batch-size 32 \
#     --max-epoch 10 \
#     \
#     --data-dir ./data/${DATASET}/processed \
#     --train-subset human_train \
#     --valid-subset human_test \
#     --max-len 800 \
#     \
#     --task ppi \
#     --arch ${ARCH} \
#     --criterion ${CRITEION} \
#     \
#     --dropout 0.2 \
#     --emb-dim 1024 \
#     --hid-dim 256 \
#     --cnn-layers 3 \
#     --kernel-size 3 \
#     --trans-layers 6    \
#     --wo-ppm


# with ppm, without pooling
fairseq-train \
    --user-dir module \
    --save-dir ./save/${DATASET}/${ARCH}-wopool \
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
    \
    --dropout 0.2 \
    --emb-dim 1024 \
    --hid-dim 256 \
    --cnn-layers 3 \
    --kernel-size 3 \
    --trans-layers 6    \
    --wo-pool


# with ppm, with pooling
# fairseq-train \
#     --user-dir module \
#     --save-dir ./save/${DATASET}/${ARCH} \
#     --seed 100 \
#     \
#     --optimizer adam \
#     --lr 3e-5 \
#     --batch-size 32 \
#     --max-epoch 3 \
#     \
#     --data-dir ./data/${DATASET}/processed \
#     --train-subset human_train \
#     --valid-subset human_test \
#     --max-len 800 \
#     \
#     --task ppi \
#     --arch ${ARCH} \
#     --criterion ${CRITEION} \
#     \
#     --dropout 0.2 \
#     --emb-dim 1024 \
#     --hid-dim 256 \
#     --cnn-layers 3 \
#     --kernel-size 3 \
#     --trans-layers 6
