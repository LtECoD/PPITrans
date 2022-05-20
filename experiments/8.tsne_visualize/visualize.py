import os
import random
import argparse
import pickle
import numpy as np
import torch

import sys
sys.path.append(".")
from experiments.utils import Protein, load_model
from experiments.utils import forward_kth_translayer
from experiments.utils import lookup_embed

def load_sample(_dir, chosen_son):
    seqs = open(os.path.join(_dir, "seqs.txt"), "r").readlines()
    sample = pickle.load(open(os.path.join(_dir, f"{chosen_son}.pkl"), "rb"))

    proteins = [Protein(name=pro, seq=seq) for pro, seq in seqs.items()]
    fid, sid = sample[0], sample[1]
    sites = sample[2]
    cm = sample[4]

    fpro = Protein(name=fid, seq=seqs[fid])
    spro = Protein(name=fid, seq=seqs[fid])

    return fpro, spro, sites, cm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--self_dir", type=str, default="./experiments/7.ppi_contact_map")
    parser.add_argument("--model_dir", type=str, help="saved ppi model")
    parser.add_argument("--chosen_son", type=str)
    args = parser.parse_args()
    random.seed(args.seed)

    model_name = os.path.basename(args.model_dir)
    model = load_model(args.model_dir)

    protein_dir = os.path.join(args.self_dir, "data")
    emb_dir = os.path.join(protein_dir, "embs")
    fpro, spro, sites, cm = load_sample(protein_dir)
    print(f"Protein: {fpro}")
    print(f"Protein: {spro}")

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