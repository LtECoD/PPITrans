#! /bin/bash

python experiments/2.type_classify/type_classsify.py \
    --model_dir ./save/dscript/ppi

python experiments/2.type_classify/type_classsify.py \
    --model_dir ./save/dscript/ppi-woppm

python experiments/2.type_classify/type_classsify.py \
    --model_dir ./save/dscript/rnn

python experiments/2.type_classify/type_classsify.py \
    --model_dir ./save/dscript/rnn-woppm