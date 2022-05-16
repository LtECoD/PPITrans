#! /bin/bash

python experiments/1.acid_classify/acid_classsify.py \
    --model_dir ./save/dscript/ppi

python experiments/1.acid_classify/acid_classsify.py \
    --model_dir ./save/dscript/ppi-woppm

python experiments/1.acid_classify/acid_classsify.py \
    --model_dir ./save/dscript/rnn

python experiments/1.acid_classify/acid_classsify.py \
    --model_dir ./save/dscript/rnn-woppm