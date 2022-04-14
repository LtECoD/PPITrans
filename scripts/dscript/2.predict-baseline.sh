#! /bin/bash

DATASET=dscript
ARCH=baseline


python module/predict.py \
    --path              # 模型存储路径
    --results_path
    --gen_subset