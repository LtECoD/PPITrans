#! /bin/bash

DATASET=dscript
ARCH=baseline
CRITEION=ppi_contrastive

# 'mouse_test' 'fly_test' 'worm_test' 'yeast_test' 
TEST_SET=('ecoli_test' 'human_test')

for split in ${TEST_SET[*]}; do
    python module/evaluate.py \
        --result_dir ./results/${DATASET}/${ARCH}-${CRITEION}/prediction  \
        --metric_dir ./results/${DATASET}/${ARCH}-${CRITEION}/metric \
        --split ${split}
done