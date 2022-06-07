import os
import random
import argparse
import pickle
from turtle import forward
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict


import sys

from transformers import PrefixConstrainedLogitsProcessor
sys.path.append(".")
from experiments.utils import Protein, load_model
from experiments.utils import forward_kth_translayer
from experiments.utils import forward_decoder_proj
from experiments.utils import lookup_embed


def softmax(x):    
    x = x - np.max(x, axis = -1, keepdims = True)    
    x = np.exp(x) / np.sum(np.exp(x), axis = -1, keepdims = True)
    return x


def load_sample(pair_fp, seq_fp, k=5):
    with open(seq_fp, "r") as f:
        seqs_dict = dict([line.strip().split("\t") for line in f.readlines()])
    ppi_pairs = defaultdict(list)
    unppi_pairs = defaultdict(list)

    for line in open(pair_fp, "r").readlines():
        fid, sid, l = line.strip().split()
        if int(l) == 1:
            ppi_pairs[fid].append(sid)
            ppi_pairs[sid].append(fid)
        elif int(l) == 0:
            unppi_pairs[fid].append(sid)
            unppi_pairs[sid].append(fid)

    while True:
        chosen_son = random.choice(list(seqs_dict.keys()))
        if len(ppi_pairs[chosen_son]) > k and len(unppi_pairs[chosen_son]) > k:
            break
    chosen_ppi_sons = random.sample(ppi_pairs[chosen_son], k=k)
    chosen_unppi_sons = random.sample(unppi_pairs[chosen_son], k=k)

    chosen_pro = Protein(name=chosen_son, seq=seqs_dict[chosen_son])
    pos_pros = [Protein(name=son, seq=seqs_dict[son]) for son in chosen_ppi_sons]
    neg_pros = [Protein(name=son, seq=seqs_dict[son]) for son in chosen_unppi_sons]
    return chosen_pro, pos_pros, neg_pros


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--self_dir", type=str, default="./experiments/8.1.important_residuals")
    parser.add_argument("--model_dir", type=str, help="saved ppi model")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--m", type=int, default=1000)
    args = parser.parse_args()
    random.seed(args.seed)

    emb_dir= os.path.join("./data/dscript/processed/embs/human_test")
    seq_fp = os.path.join("./data/dscript/processed/seqs/human_test.fasta")
    pair_fp = os.path.join("./data/dscript/processed/pairs/human_test.tsv")

    model_name = os.path.basename(args.model_dir)
    model = load_model(args.model_dir)

    pro, pos_pros, neg_pros = load_sample(pair_fp, seq_fp, args.k)
    print(f"Protein: {pro}")
    for p in pos_pros:
        print(f"Interated Protein: {p}")
    for p in neg_pros:
        print(f"Uninteracted Protein: {p}")

    for p in [pro] + pos_pros + neg_pros:
        # embedder
        if not hasattr(model.encoder, "embeder"):
            emb = np.load(os.path.join(emb_dir, p.name+".npy"))
        else:
            emb = lookup_embed(p, model.encoder.embeder)
        # projector
        with torch.no_grad():
            emb = model.encoder.forward_projecter(torch.Tensor(emb).unsqueeze(0))
            emb = emb.squeeze(0).numpy()
        # layers
        for k in range(model.encoder.num_layers):    
            emb = forward_kth_translayer(model, emb, k)
        p.set_emb(emb)

    pro_rep = np.mean(pro.emb, axis=0)
    pos_logits = []
    neg_logits = []
    for p in pos_pros:
        rep = pro_rep * np.mean(p.emb, axis=0)
        logits = forward_decoder_proj(model, rep)
        pos_logits.append(logits)
    for p in neg_pros:
        rep = pro_rep * np.mean(p.emb, axis=0)
        logits = forward_decoder_proj(model, rep)
        neg_logits.append(logits)
    pos_logits = np.vstack(pos_logits)
    neg_logits = np.vstack(neg_logits)
    pos_probs = softmax(pos_logits)
    neg_probs = softmax(neg_logits)

    pos_cms = []
    for idx, p in enumerate(pos_pros):
        element_rep = np.expand_dims(pro.emb, axis=1) * np.expand_dims(p.emb, axis=0)   # L1 x L2 x D
        element_logits = forward_decoder_proj(model, element_rep)
        element_logits_gap = element_logits[:, :, 1] - element_logits[:, :, 0]
        gap_threshold = np.partition(element_logits_gap.reshape(-1), -args.m)[-args.m]
        element_cm = element_logits_gap >= gap_threshold
        pos_cms.append(element_cm)

    neg_cms = []
    for idx, p in enumerate(neg_pros):
        element_rep = np.expand_dims(pro.emb, axis=1) * np.expand_dims(p.emb, axis=0)   # L1 x L2 x D
        element_logits = forward_decoder_proj(model, element_rep)
        element_logits_gap = element_logits[:, :, 1] - element_logits[:, :, 0]
        gap_threshold = np.partition(element_logits_gap.reshape(-1), -args.m)[-args.m]
        element_cm = element_logits_gap >= gap_threshold
        neg_cms.append(element_cm)

    with open(os.path.join(args.self_dir, "tmpdata", f"{pro.name}.pkl"), "wb") as f:
            pickle.dump([pro.name, pos_cms, neg_cms, pos_probs, neg_probs], f)


    # fig, subs = plt.subplots(nrows=2, ncols=args.k, figsize=(15, 5))
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.7)
    # for idx, p in enumerate(pos_pros):
    #     subs[0][idx].imshow(pos_cms[idx], cmap="Blues")
    #     subs[0][idx].set_xlabel(p.name)
    #     subs[0][idx].set_title(f"P={round(float(pos_probs[idx][1]), 3)}")
    # for idx, p in enumerate(neg_pros):
    #     subs[1][idx].imshow(neg_cms[idx], cmap="BuGn")
    #     subs[1][idx].set_xlabel(p.name)
    #     subs[1][idx].set_title(f"P={round(float(neg_probs[idx][1]), 3)}")

    # plt.savefig(os.path.join(args.self_dir, f"{pro.name}-importance.pdf"))
