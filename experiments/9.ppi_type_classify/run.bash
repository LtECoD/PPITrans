#! /bin/bash

python experiments/9.ppi_type_classify/ppi_type_classify.py   \
    --model_dir ./save/dscript/ppi

python experiments/9.ppi_type_classify/ppi_type_classify.py   \
    --model_dir ./save/dscript/ppi-woppm

python experiments/9.ppi_type_classify/ppi_type_classify.py   \
    --model_dir ./save/dscript/rnn

python experiments/9.ppi_type_classify/ppi_type_classify.py   \
    --model_dir ./save/dscript/rnn-woppm
