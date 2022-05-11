#! /bin/bash

python experiments/3.secondary_structure/ss.py  \
    --model_dir ./save/dscript/ppi

python experiments/3.secondary_structure/ss.py  \
    --model_dir ./save/dscript/ppi \
    --is_eight_class

python experiments/3.secondary_structure/ss.py  \
    --model_dir ./save/dscript/ppi-woppm

python experiments/3.secondary_structure/ss.py  \
    --model_dir ./save/dscript/ppi-woppm \
    --is_eight_class