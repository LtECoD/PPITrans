"""
观察互作蛋白质和非互作蛋白，氨基酸频率的差异
"""
from collections import defaultdict
import os
import sys
import random
import argparse
import scipy.stats
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from typing import Counter

sys.path.append(".")
from experiments.utils import organisms, acids, Protein


def load_proteins(_dir, split):
    lines = open(os.path.join(_dir, split+".fasta")).readlines()
    proteins = []
    for l in lines:
        name, seq = l.strip().split()
        pro = Protein(name=name, seq=seq)
        pro.count_aa()
        proteins.append(pro)
    return proteins

def load_pairs(_dir, split, pro_idx_dict):
    pairs = []
    #读取pair
    orga_pair_fp = os.path.join(_dir, split+".tsv")
    pairs = [line.strip().split() for line in open(orga_pair_fp, "r").readlines()]
    pairs = [[pro_idx_dict[fp], pro_idx_dict[sp], int(l)]  for fp, sp, l in pairs]
    return pairs


def JS_divergence(p,q):
    m=(p+q)/2
    return 0.5*scipy.stats.entropy(p, m, base=2)+0.5*scipy.stats.entropy(q, m, base=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument('--self_dir', type=str, default="./experiments/11.acid_cooccurence")
    args = parser.parse_args()
    random.seed(args.seed)

    protein_pair_dir = os.path.join(args.self_dir, "data", "data")
    os.makedirs(protein_pair_dir, exist_ok=True)
    if len(os.listdir(protein_pair_dir)) <= 0:
        protein_dir = os.path.join("./data/dscript/processed/seqs")
        pair_dir = os.path.join("./data/dscript/processed/pairs")
        proteins = {}
        pairs = {}
        for orga in organisms:
            proteins[orga] = load_proteins(protein_dir, orga+"_test")

            pro_idx_dict = {}
            for idx, pro in enumerate(proteins[orga]):
                pro_idx_dict[pro.name] = idx

            pairs[orga] = load_pairs(pair_dir, orga+"_test", pro_idx_dict)
            print(f"{orga}: pair num: {len(pairs[orga])}, protein num: {len(proteins[orga])}")
        pickle.dump(proteins, open(os.path.join(protein_pair_dir, "proteins.pl"), "wb"))
        pickle.dump(pairs, open(os.path.join(protein_pair_dir, "pairs.pl"), "wb"))
    else:
        proteins = pickle.load(open(os.path.join(protein_pair_dir, "proteins.pl"), "rb"))
        pairs = pickle.load(open(os.path.join(protein_pair_dir, "pairs.pl"), "rb"))

    # 统计互作蛋白的熵和非互作蛋白的熵
    ppi_ent_dir = os.path.join(args.self_dir, "data", "ppi_ent")
    os.makedirs(ppi_ent_dir, exist_ok=True)
    if len(os.listdir(ppi_ent_dir)) <= 0:
        ppi_ent = defaultdict(list)
        unppi_ent = defaultdict(list)
        for orga in organisms:
            for (fid, sid, l) in pairs[orga]:
                fpro = proteins[orga][fid]
                spro = proteins[orga][sid]

                counter = Counter(list(fpro.seq+spro.seq))
                counts = [0] * len(acids)
                for aa, value in counter.items():
                    counts[acids.index(aa)] = value
                assert sum(counts) == len(fpro.seq) + len(spro.seq)
                length = sum(counts)
                freq = np.array([v/length for v in counts])
                ent = scipy.stats.entropy(freq, base=2)
                if l == 1:
                    ppi_ent[orga].append(ent)
                elif l == 0:
                    unppi_ent[orga].append(ent)
                else:
                    raise ValueError
        pickle.dump(ppi_ent, open(os.path.join(ppi_ent_dir, "ppi.pl"), "wb"))
        pickle.dump(unppi_ent, open(os.path.join(ppi_ent_dir, "unppi.pl"), "wb"))
    else:
        ppi_ent = pickle.load(open(os.path.join(ppi_ent_dir, "ppi.pl"), "rb"))
        unppi_ent = pickle.load(open(os.path.join(ppi_ent_dir, "unppi.pl"), "rb"))
    fig, subs = plt.subplots(len(organisms), 1, figsize=(12, 5))
    for idx, orga in enumerate(organisms):
        ppi_density = gaussian_kde(ppi_ent[orga])
        ppi_density.covariance_factor = lambda : .1
        ppi_density._compute_covariance()
        ppi_xs = np.linspace(0, np.max(ppi_ent[orga]), 200)
        ppi_ys = ppi_density(ppi_xs)

        unppi_density = gaussian_kde(unppi_ent[orga])
        unppi_density.covariance_factor = lambda : .1
        unppi_density._compute_covariance()
        unppi_xs = np.linspace(0, np.max(unppi_ent[orga]), 200)
        unppi_ys = unppi_density(unppi_xs)

        subs[idx].plot(ppi_xs, ppi_ys)
        subs[idx].plot(unppi_xs, unppi_ys)
    
    plt.savefig(os.path.join(args.self_dir, "ppi_ent_dist.pdf"))




    # 统计互作蛋白的js和非互作蛋白的js分布
    # ppi_js_dir = os.path.join(args.self_dir, "data", "ppi_js")
    # os.makedirs(ppi_js_dir, exist_ok=True)
    # if len(os.listdir(ppi_js_dir)) <= 0:
    #     ppi_js = defaultdict(list)
    #     unppi_js = defaultdict(list)
    #     for orga in organisms:
    #         for (fid, sid, l) in pairs[orga]:
    #             fpro = proteins[orga][fid]
    #             spro = proteins[orga][sid]
    #             js = JS_divergence(fpro.aa_freq, spro.aa_freq)
    #             if l == 1:
    #                 ppi_js[orga].append(js)
    #             elif l == 0:
    #                 unppi_js[orga].append(js)
    #             else:
    #                 raise ValueError
    #     pickle.dump(ppi_js, open(os.path.join(ppi_js_dir, "ppi.pl"), "wb"))
    #     pickle.dump(unppi_js, open(os.path.join(ppi_js_dir, "unppi.pl"), "wb"))
    # else:
    #     ppi_js = pickle.load(open(os.path.join(ppi_js_dir, "ppi.pl"), "rb"))
    #     unppi_js = pickle.load(open(os.path.join(ppi_js_dir, "unppi.pl"), "rb"))
    # fig, subs = plt.subplots(len(organisms), 1, figsize=(12, 5))
    # for idx, orga in enumerate(organisms):
    #     ppi_density = gaussian_kde(ppi_js[orga])
    #     ppi_density.covariance_factor = lambda : .1
    #     ppi_density._compute_covariance()
    #     ppi_xs = np.linspace(0, 0.2, 200)
    #     ppi_ys = ppi_density(ppi_xs)

    #     unppi_density = gaussian_kde(unppi_js[orga])
    #     unppi_density.covariance_factor = lambda : .1
    #     unppi_density._compute_covariance()
    #     unppi_xs = np.linspace(0, 0.2, 200)
    #     unppi_ys = unppi_density(unppi_xs)

    #     subs[idx].plot(ppi_xs, ppi_ys)
    #     subs[idx].plot(unppi_xs, unppi_ys)
    
    # plt.savefig(os.path.join(args.self_dir, "ppi_js_dist.pdf"))

    

