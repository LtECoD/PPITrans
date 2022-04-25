#! /bin/bash

DATASET=dscript
ARCH=naive_ppi
CRITEION=ppi_crossentropy

TEST_SET=('ecoli_test' 'mouse_test' 'fly_test' 'worm_test' 'yeast_test' 'human_test')

for split in ${TEST_SET[*]}; do
    python module/evaluate.py \
        --result_dir ./results/${DATASET}/${ARCH}/prediction  \
        --metric_dir ./results/${DATASET}/${ARCH}/metric \
        --split ${split}

    python module/evaluate.py \
        --result_dir ./results/${DATASET}/${ARCH}-woppm/prediction  \
        --metric_dir ./results/${DATASET}/${ARCH}-woppm/metric \
        --split ${split}
done