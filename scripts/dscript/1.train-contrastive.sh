#! /bin/bash

DATASET=dscript
ARCH=contrastive
CRITEION=contrastive

# fairseq-train \
#     --user-dir module \
#     --save-dir ./save/${DATASET}/${ARCH}-${CRITEION} \
#     --seed 100 \
#     \
#     --optimizer adam \
#     --lr 3e-5 \
#     --batch-size 32 \
#     --max-epoch 5 \
#     \
#     --data-dir ./data/${DATASET}/processed \
#     --train-subset human_train \
#     --valid-subset human_test \
#     --max-len 800 \
#     \
#     --task ppi \
#     --arch ${ARCH} \
#     --criterion ${CRITEION} \
#     --temp 0.1  \
#     \
#     --dropout 0.2 \
#     --emb-dim 1024 \
#     --hid-dim 256 \
#     --cnn-layers 3 \
#     --kernel-size 3 \
#     --trans-layers 6

ENCODER_SAVE_DIR=./save/${DATASET}/${ARCH}-${CRITEION}
ARCH=contrastiveppi
CRITEION=ppi_crossentropy

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
    \
    --cmodel-path ${ENCODER_SAVE_DIR}/checkpoint_last.pt \
    --dropout 0.2 \
    --emb-dim 1024 \
    --hid-dim 256 \
    --cnn-layers 3 \
    --kernel-size 3 \
    --trans-layers 6

