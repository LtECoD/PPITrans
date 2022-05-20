#! /bin/bash

python experiments/8.context_prediction/cp.py \
    --model_dir ./save/dscript/ppi

python experiments/8.context_prediction/cp.py \
    --model_dir ./save/dscript/ppi-woppm

python experiments/8.context_prediction/cp.py \
    --model_dir ./save/dscript/rnn

python experiments/8.context_prediction/cp.py \
    --model_dir ./save/dscript/rnn-woppm