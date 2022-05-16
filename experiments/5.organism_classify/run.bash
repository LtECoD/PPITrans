#!/bin/bash

python experiments/5.organism_classify/orga_classify.py \
    --model_dir ./save/dscript/ppi-woppm

python experiments/5.organism_classify/orga_classify.py \
    --model_dir ./save/dscript/ppi

python experiments/5.organism_classify/orga_classify.py \
    --model_dir ./save/dscript/rnn-woppm

python experiments/5.organism_classify/orga_classify.py \
    --model_dir ./save/dscript/rnn