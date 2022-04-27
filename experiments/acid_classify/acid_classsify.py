import os
import numpy as np
import argparse
import random
from collections import defaultdict

organisms = ['ecoli', 'mouse', 'fly', 'worm', 'yeast', 'human']
standard_acids = [
        ('A', 1), ('C', 6), ('D', 5), ('E', 7), ('F', 2), 
        ('G', 1), ('H', 4), ('I', 2), ('K', 5), ('L', 2),
        ('M', 3), ('N', 4), ('P', 2), ('Q', 7), ('R', 4),
        ('S', 3), ('T', 3), ('V', 1), ('W', 4), ('Y', 3), ('X', 0)]


class Protein:
    def __init__(self, name, seq):
        self.name = name
        self.seq = seq

    def set_emb(self, emb):
        self.emb = emb
    
    def discretize(self):
        """打散序列"""
        return list(self.seq), self.emb

    def __str__(self):
        return self.name + "\t" + self.seq


def get_proteins(args):
    """读取氨基酸及其序列，并按照物种划分好"""
    proteins = defaultdict(list)
    for orga in organisms:
        orga_fasta_fp = os.path.join(args.seq_dir, orga+"_test.fasta")
        lines = open(orga_fasta_fp, "r").readlines()
        selected_proteins = random.choices(lines, k=args.num_protein_per_organism)
        for sp in selected_proteins:
            pro, seq = sp.strip().split("\t")
            proteins[orga].append(Protein(name=pro, seq=seq))
    return proteins


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_protein_per_organism", type=int, default=100)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--seq_dir", type=str, default="./data/dscript/processed/seqs")
    parser.add_argument("--pretrained_emb_dir", type=str, default='./data/dscript/processed/embs')
    parser.add_argument("--self_dir", type=str, default="./experiments/acid_classify")
    parser.add_argument("--test_ppm", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)

    # selct proteins
    train_proteins = get_proteins(args)
    test_proteins = get_proteins(args)


    ##### 测试pretrained-embedding
    #读取embedding
    if args.test_ppm:
        for org in organisms:
            for pro in train_proteins[org]:
                emb = np.load(os.path.join(args.pretrained_emb_dir, org+"_test", pro.name+".npy"))
                pro.set_emb(emb)

            for pro in test_proteins[org]:
                emb = np.load(os.path.join(args.pretrained_emb_dir, org+"_test", pro.name+".npy"))
                pro.set_emb(emb)
        print(">>>>> complete loading pretrained embeddings")

        for org in organisms:
            pass


    

