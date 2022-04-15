#! /bin/bash

DATASET=dscript
ARCH=baseline

TEST_SET=('human_test' 'ecoli_test' 'fly_test' 'mouse_test' 'worm_test' 'yeast_test')

for split in ${TEST_SET[*]}; do
    python module/evaluate.py \
        --result_dir ./results/${DATASET}/${ARCH}/prediction  \
        --metric_dir ./results/${DATASET}/${ARCH}/metric \
        --split ${split}
done