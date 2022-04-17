#! /bin/bash

DATASET=dscript
ARCH=baseline
CRITEION=ppi_crossentropy

TEST_SET=('human_test' 'ecoli_test' 'fly_test' 'mouse_test' 'worm_test' 'yeast_test')

for set in ${TEST_SET[*]}; do
	python module/predict.py \
        --user-dir module \
        --task ppi \
        \
        --arch ${ARCH} \
        --path  ./save/${DATASET}/${ARCH}-${CRITEION}/checkpoint_best.pt \
        --emb-dim 1024 \
        --hid-dim 256 \
        \
        --results-path  ./results/${DATASET}/${ARCH}-${CRITEION}/prediction  \
        --rep-path ./results/${DATASET}/${ARCH}-${CRITEION}/rep \
        \
        --data-dir ./data/${DATASET}/processed \
        --max-len 800 \
        --gen-subset  ${set}  \
        --batch-size 32
done

