import os
import sys
import random
import argparse
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Counter

sys.path.append(".")
from experiments.utils import organisms, acids, Protein
acids = acids[:-1]


def load_proteins(_dir, split, k):
    lines = open(os.path.join(_dir, split+".fasta")).readlines()
    proteins = []
    if k != -1:
        lines = random.sample(lines, k=k)
    for l in lines:
        name, seq = l.strip().split()
        pro = Protein(name=name, seq=seq)
        if "X" in pro.seq:
            continue
        proteins.append(pro)
    return proteins


def build_seq_co_matrix(seq, window):
    matrix = np.zeros((len(acids), len(acids)))
    for idx in range(window, len(seq)-window-1):
        context = seq[idx-window: idx] + seq[idx+1:idx+window+1]
        aa_num_dict = dict(Counter(list(context)))
        for _aa in aa_num_dict:
            matrix[acids.index(seq[idx]), acids.index(_aa)] += aa_num_dict[_aa]
    return matrix

def normalize(matrix):
    row_sum = np.sum(matrix, axis=-1)
    return matrix / (np.expand_dims(row_sum, axis=-1)+1e-6)


def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p, M, base=2)+0.5*scipy.stats.entropy(q, M, base=2)


def co_matrix_divergence(co_matrix1, co_matrix2):
    js_sum = 0
    for idx in range(len(co_matrix1)):
        js_sum += JS_divergence(co_matrix1[idx], co_matrix2[idx])
    return js_sum


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--self_dir', type=str, default="./experiments/11.acid_cooccurence")
    parser.add_argument('--protein_num_per_orga', type=int, default=5000)

    args = parser.parse_args()
    random.seed(args.seed)

    protein_dir = os.path.join("./data/dscript/processed/seqs")
    proteins = {}
    for orga in organisms:
        proteins[orga] = load_proteins(protein_dir, orga+"_test", args.protein_num_per_orga)
        print(f"{orga} protein num: {len(proteins[orga])}")

    # 直接统计频率
    orga_aa_freq_dict = {}
    for orga in organisms:
        orga_acid_freq = np.zeros(len(acids))
        for pro in proteins[orga]:
            pro_freq = Counter(list(pro.seq))
            for aa, value in pro_freq.items():
                orga_acid_freq[acids.index(aa)] += value
        orga_acid_freq = orga_acid_freq / np.sum(orga_acid_freq)
        orga_aa_freq_dict[orga] = orga_acid_freq
    aa_freq_js_pd = pd.DataFrame(index=organisms, columns=organisms)
    for forga in organisms:
        for sorga in organisms:
            aa_freq_js_pd[forga][sorga] = JS_divergence(
                orga_aa_freq_dict[forga], orga_aa_freq_dict[sorga])
    fig = plt.subplot()
    fig.set_xticks(range(len(organisms)), organisms)
    fig.set_yticks(range(len(organisms)), organisms)
    im = fig.imshow(np.array(aa_freq_js_pd, dtype=np.float32), cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.savefig(f"./experiments/9.acid_cooccurence/aa_freq_js.pdf", bbox_inches='tight')
    plt.close()


    # 计算共现矩阵差异
    orga_matrix_dict = {}
    for orga in organisms:
        orga_matrix = np.zeros((len(acids), len(acids)))
        for pro in proteins[orga]:
            seq_co_matrix = build_seq_co_matrix(pro.seq, window=args.window)
            orga_matrix = orga_matrix + seq_co_matrix
        orga_matrix = normalize(orga_matrix)
        orga_matrix_dict[orga] = orga_matrix    

    # 计算共现矩阵差异
    window_co_dis = np.zeros((len(organisms), len(organisms)))
    for forga in organisms:
        for sorga in organisms:
            if forga == sorga:
                continue
            window_co_dis[organisms.index(forga), organisms.index(sorga)] = \
                co_matrix_divergence(orga_matrix_dict[forga], orga_matrix_dict[sorga])
    
    fig = plt.subplot()
    fig.set_xticks(range(len(organisms)), organisms)
    fig.set_yticks(range(len(organisms)), organisms)
    im = fig.imshow(np.array(window_co_dis, dtype=np.float32), cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.savefig(f"./experiments/9.acid_cooccurence/window-{args.window}-co-dis.pdf", bbox_inches='tight')
    plt.close()
