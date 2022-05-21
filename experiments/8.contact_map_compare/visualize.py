import os
import random
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
sys.path.append(".")
from experiments.utils import Protein, load_model
from experiments.utils import forward_kth_translayer
from experiments.utils import forward_decoder_proj
from experiments.utils import lookup_embed


def softmax(x):    
    x = x - np.max(x, axis = -1, keepdims = True)    
    x = np.exp(x) / np.sum(np.exp(x), axis = -1, keepdims = True)
    return x


def load_sample(_dir, chosen_son):
    seqs = open(os.path.join(_dir, "seqs.txt"), "r").readlines()
    seqs = dict([l.strip().split() for l in seqs])
    sample = pickle.load(open(os.path.join(_dir, "samples", f"{chosen_son}.pkl"), "rb"))

    fid, sid = sample[0], sample[1]
    sites = sample[2]
    cm = sample[3]

    fpro = Protein(name=fid, seq=seqs[fid])
    spro = Protein(name=sid, seq=seqs[sid])

    return fpro, spro, sites, cm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--self_dir", type=str, default="./experiments/8.contact_map_compare")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--model_dir", type=str, help="saved ppi model")
    parser.add_argument("--threshold", type=int, default=16)
    args = parser.parse_args()
    random.seed(args.seed)

    data_dir = os.path.join(args.self_dir, "data")
    samples_dir = os.path.join(data_dir, "samples")
    emb_dir = os.path.join("./experiments/7.ppi_contact_map/data/embs")
    sample_num = len(\
        [l for l in os.listdir(samples_dir) if l.endswith("pkl")])

    model_name = os.path.basename(args.model_dir)
    model = load_model(args.model_dir)

    fig, subs = plt.subplots(nrows=2, ncols=args.k, figsize=(12, 5))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0.7)
    chosen_sons = list(range(sample_num))
    random.shuffle(chosen_sons)
    idx = 0
    for cs in chosen_sons:

        fpro, spro, sites, cm = load_sample(data_dir, chosen_son=cs)
        fpro_reps = []
        spro_reps = []
        # embedder
        if not hasattr(model.encoder, "embeder"):
            fpro_reps.append(np.load(os.path.join(emb_dir, fpro.name+".npy")))
            spro_reps.append(np.load(os.path.join(emb_dir, spro.name+".npy")))
        else:
            fpro_reps.append(lookup_embed(fpro, model.encoder.embeder))
            spro_reps.append(lookup_embed(spro, model.encoder.embeder))

        # projector
        with torch.no_grad():
            femb = model.encoder.forward_projecter(torch.Tensor(fpro_reps[-1]).unsqueeze(0))
            semb = model.encoder.forward_projecter(torch.Tensor(spro_reps[-1]).unsqueeze(0))
            fpro_reps.append(femb.detach().squeeze(0).numpy())
            spro_reps.append(semb.detach().squeeze(0).numpy())
        # layers
        for k in range(model.encoder.num_layers):    
            femb = forward_kth_translayer(model, fpro_reps[-1], k)
            semb = forward_kth_translayer(model, spro_reps[-1], k)
            fpro_reps.append(femb)
            spro_reps.append(semb)
    
        # classifier
        frep = fpro_reps[-1]
        srep = spro_reps[-1]
        rep = np.mean(frep, axis=0) * np.mean(srep, axis=0)
        logits = forward_decoder_proj(model, rep)
        probs = softmax(logits)
        if probs[1] < probs[0]:
            continue

        element_rep = np.expand_dims(frep, axis=1) * np.expand_dims(srep, axis=0)       # L1 x L2 x D
        element_logits = forward_decoder_proj(model, element_rep)
        element_logits_gap = element_logits[:, :, 1] - element_logits[:, :, 0]
        gap_threshold = np.partition(element_logits_gap.reshape(-1), -len(sites))[-len(sites)]
        element_cm = element_logits_gap >= gap_threshold

        subs[0][idx].imshow(element_cm, cmap="Blues")
        subs[0][idx].set_title(f"P={round(float(probs[1]), 3)}")
        subs[0][idx].set_ylabel(fpro.name)
        subs[0][idx].set_xticks([])
        subs[0][idx].set_yticks([])

        subs[1][idx].imshow(cm<args.threshold, cmap="Blues")
        subs[1][idx].set_ylabel(fpro.name)
        subs[1][idx].set_xlabel(spro.name)
        subs[1][idx].set_xticks([])
        subs[1][idx].set_yticks([])

        idx += 1
        if idx >= args.k:
            break

    plt.savefig(os.path.join(args.self_dir, f"cm_compare.pdf"))

