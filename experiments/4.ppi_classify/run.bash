#! /bin/bash

python experiments/4.ppi_classify/ppi_classify.py   \
    --model_dir ./save/dscript/ppi-woppm

python experiments/4.ppi_classify/ppi_classify.py   \
    --model_dir ./save/dscript/ppi

python experiments/4.ppi_classify/ppi_classify.py   \
    --model_dir ./save/dscript/rnn-woppm

python experiments/4.ppi_classify/ppi_classify.py   \
    --model_dir ./save/dscript/rnn