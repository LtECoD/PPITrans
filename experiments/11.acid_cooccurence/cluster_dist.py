"""
观察互作蛋白质和非互作蛋白，氨基酸频率的差异
"""
import os
import sys
import random
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import gaussian_kde
from tqdm import tqdm
from itertools import combinations
sys.path.append(".")
from experiments.utils import organisms, Protein

from Bio import Align
from Bio.Align import substitution_matrices

matrix = substitution_matrices.load("PAM250")
aligner = Align.PairwiseAligner()
aligner.substitution_matrix = matrix


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


def score(seq1, seq2):
    score = aligner.score(seq1, seq2) / min(len(seq1), len(seq2))
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument('--self_dir', type=str, default="./experiments/11.acid_cooccurence")
    parser.add_argument("--cluster_sample_k", type=int, default=10)
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

    ppi_cluster_score = defaultdict(list)
    unppi_cluster_score = defaultdict(list)

    for orga in organisms:
        adj_list = defaultdict(list)
        unadj_list = defaultdict(list)
        for (fidx, sidx, l) in pairs[orga]:
            if l == 0:
                unadj_list[fidx].append(sidx)
                unadj_list[sidx].append(fidx)
            elif l == 1:
                adj_list[fidx].append(sidx)
                adj_list[sidx].append(fidx)
            else:
                raise ValueError

        for key in tqdm(adj_list):
            neighbors = adj_list[key]
            un_neighs = unadj_list[key]

            if len(neighbors) < 3 or len(un_neighs) < 3:
                continue

            nei_comb = list(combinations(neighbors, 2))
            unn_comb = list(combinations(un_neighs, 2))

            nei_comb = random.sample(nei_comb, k=min(len(nei_comb), args.cluster_sample_k))
            unn_comb = random.sample(unn_comb, k=min(len(unn_comb), args.cluster_sample_k))

            for (fidx, sidx) in nei_comb:
                ppi_cluster_score[orga].append(score(
                    proteins[orga][fidx].seq, proteins[orga][sidx].seq))
            for (fidx, sidx) in unn_comb:
                unppi_cluster_score[orga].append(score(
                    proteins[orga][fidx].seq, proteins[orga][sidx].seq))

    fig, subs = plt.subplots(len(organisms), 1, figsize=(12, 5))
    for idx, orga in enumerate(organisms):
        print(random.sample(ppi_cluster_score[orga], k=10))
        print(random.sample(unppi_cluster_score[orga][:10], k=10))
        print()

        ppi_density = gaussian_kde(ppi_cluster_score[orga])
        ppi_xs = np.linspace(0, max(ppi_cluster_score[orga]), 200)
        ppi_ys = ppi_density(ppi_xs)

        unppi_density = gaussian_kde(unppi_cluster_score[orga])
        unppi_xs = np.linspace(0, max(unppi_cluster_score[orga]), 200)
        unppi_ys = unppi_density(unppi_xs)

        subs[idx].plot(ppi_xs, ppi_ys)
        subs[idx].plot(unppi_xs, unppi_ys)
    
    plt.savefig(os.path.join(args.self_dir, "cluster_dist.pdf"))


    # # 统计互作蛋白的熵和非互作蛋白的比对得分
    # ppi_score = defaultdict(list)
    # unppi_score = defaultdict(list)
    # for orga in organisms[:1]:
    #     for (fid, sid, l) in tqdm(pairs[orga]):
    #         fpro = proteins[orga][fid]
    #         spro = proteins[orga][sid]

    #         _score = score(fpro.seq, spro.seq)

    #         if l == 1:
    #             ppi_score[orga].append(_score)
    #         elif l == 0:
    #             unppi_score[orga].append(_score)
    #         else:
    #             raise ValueError
    
    # fig, subs = plt.subplots(len(organisms), 1, figsize=(12, 5))
    # for idx, orga in enumerate(organisms[:1]):
    #     ppi_density = gaussian_kde(ppi_score[orga])
    #     ppi_xs = np.linspace(0, np.max(unppi_score[orga]), 200)
    #     ppi_ys = ppi_density(ppi_xs)

    #     unppi_density = gaussian_kde(ppi_score[orga])
    #     unppi_xs = np.linspace(0, np.max(unppi_score[orga]), 200)
    #     unppi_ys = unppi_density(unppi_xs)

    #     subs[idx].plot(ppi_xs, ppi_ys)
    #     subs[idx].plot(unppi_xs, unppi_ys)
    
    # plt.savefig(os.path.join(args.self_dir, "ppi_dist.pdf"))


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

    

