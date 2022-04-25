#! /bin/bash

DATASET=dscript
ARCH=naive_ppi


TEST_SET=('human_test' 'ecoli_test' 'fly_test' 'mouse_test' 'worm_test' 'yeast_test')

for set in ${TEST_SET[*]}; do
        # without pm, full naive model
        python module/predict.py \
        --user-dir module \
        --task ppi \
        \
        --arch ${ARCH} \
        --path  ./save/${DATASET}/${ARCH}-woppm/checkpoint_best.pt \
        --emb-dim 1024 \
        --hid-dim 256 \
        --wo-ppm  \
        \
        --results-path  ./results/${DATASET}/${ARCH}-woppm/prediction  \
        --rep-path ./results/${DATASET}/${ARCH}-woppm/rep \
        \
        --data-dir ./data/${DATASET}/processed \
        --max-len 800 \
        --gen-subset  ${set}  \
        --batch-size 32 \

        # with ppm
        python module/predict.py \
        --user-dir module \
        --task ppi \
        \
        --arch ${ARCH} \
        --path  ./save/${DATASET}/${ARCH}/checkpoint_best.pt \
        --emb-dim 1024 \
        --hid-dim 256 \
        \
        --results-path  ./results/${DATASET}/${ARCH}/prediction  \
        --rep-path ./results/${DATASET}/${ARCH}/rep \
        \
        --data-dir ./data/${DATASET}/processed \
        --max-len 800 \
        --gen-subset  ${set}  \
        --batch-size 32
done

