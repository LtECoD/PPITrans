#! /bin/bash
# python data/dscript/builddataset.py \
#     --pair_dir  ./data/dscript/data/pairs   \
#     --seq_dir   ./data/dscript/data/seqs    \
#     --processed_dir ./data/dscript/processed    \
#     --max_length    800 \
#     --min_length    50  

python data/dscript/embed.py  \
    --pretrained_model Rostlab/prot_t5_xl_uniref50 \
    --processed_dir data/dscript/processed \
    --device 0 1 \
    --batch_size 32