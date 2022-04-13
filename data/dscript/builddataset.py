import random
import os
import argparse
from typing import DefaultDict

# from paper "Learning Unknown from Correlations: 
# Graph Neural Network for Inter-novel-protein Interaction Prediction"
standard_acids = [
        ('A', 1), ('C', 6), ('D', 5), ('E', 7), ('F', 2), 
        ('G', 1), ('H', 4), ('I', 2), ('K', 5), ('L', 2),
        ('M', 3), ('N', 4), ('P', 2), ('Q', 7), ('R', 4),
        ('S', 3), ('T', 3), ('V', 1), ('W', 4), ('Y', 3)]

class PPI:
    def __init__(self, fst_pro, sec_pro, label=1):
        self.fst_pro = fst_pro
        self.sec_pro = sec_pro
        self.label = label
        assert label == 1 or label == 0

    def __eq__(self, __o: object) -> bool:
        if (self.fst_pro == __o.fst_pro and self.sec_pro == __o.sec_pro) or \
            (self.fst_pro == __o.sec_pro and self.sec_pro == __o.fst_pro):
            return True
        else:
            return False
    
    def __str__(self) -> str:
        return f"{self.fst_pro}\t{self.sec_pro}\t{str(self.label)}\n"


def handle(pairfp, fastafp):
    with open(pairfp, "r") as f:
        ppi_lines = f.readlines()
    with open(fastafp, "r") as f:
        seq_lines = f.readlines()
    
    # 存储正样本邻居，tsv文件中负样本总在最后
    pneighbors = DefaultDict(set)

    ppis = []
    proteins = set()
    for ppiline in ppi_lines:
        fpro, spro, label = ppiline.split()
        proteins.update((fpro, spro))
        label = int(float(label))
        if label == 0:
            if spro in pneighbors[fpro] or fpro in pneighbors[spro]:
                # 标签冲突，作为正样本
                continue
        elif label == 1:
            pneighbors[fpro].add(spro)
            pneighbors[spro].add(fpro)
        else:
            raise ValueError
        ppis.append(PPI(fpro, spro, label))
    
    acid_seqs = {}
    for idx in range(0, len(seq_lines), 2):
        key = seq_lines[idx].strip()[1:].strip()
        value = seq_lines[idx+1].strip()
        if key in proteins:
            acid_seqs[key] = value
    assert len(acid_seqs) == len(proteins)
    return ppis, acid_seqs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_dir", type=str, default="./data/dscript/data/pairs")
    parser.add_argument("--seq_dir", type=str, default="./data/dscript/data/seqs")
    parser.add_argument('--processed_dir', type=str, default="./data/dscript/processed")

    # 过滤样本的参数
    parser.add_argument('--max_length', type=int, default=800)
    parser.add_argument('--min_length', type=int, default=50)
    args = parser.parse_args()

    if not os.path.exists(args.processed_dir):
        os.mkdir(args.processed_dir)
        os.mkdir(os.path.join(args.processed_dir, "pairs"))
        os.mkdir(os.path.join(args.processed_dir, "seqs"))

    pair_fns = os.listdir(args.pair_dir)
    for pairfn in pair_fns:
        organism = pairfn.split("_")[0].strip()
        pairfp = os.path.join(args.pair_dir, pairfn)
        fastafp = os.path.join(args.seq_dir, organism + ".fasta")        
        ppis, acid_seqs = handle(pairfp, fastafp)

        # 被丢弃的蛋白质
        dropout_proteins = {"Too short": [], "Too long": [],} # "With non-standard acids": []}
        # 蛋白质长度
        protein_lengths = []
        for pro, acid_seq in acid_seqs.items():
            # 筛选蛋白质
            qualified = False
            if len(acid_seq) < args.min_length:
                dropout_proteins['Too short'].append(pro)
            elif len(acid_seq) > args.max_length:
                dropout_proteins['Too long'].append(pro)
            # elif len(set(list(acid_seq)) - set([acid[0] for acid in standard_acids])) > 0:
            #     dropout_proteins['With non-standard acids'].append(pro)
            else:
                qualified = True
            if qualified:
                protein_lengths.append((pro, len(acid_seq)))
        
        # 输出蛋白质的数量等信息
        if sum([len(value) for value in dropout_proteins.values()]) > 0:
            print(f"============{pairfn.split('.')[0]} Dataset Filter============")
            print(f"Total {len(acid_seqs)} proteins.")
            print(f"\tFilter {len(dropout_proteins['Too short'])} because they are too short.")
            print(f"\tFilter {len(dropout_proteins['Too long'])} because they are too long.")
            # print(f"\tFilter {len(dropout_proteins['With non-standard acids'])} because they have non-standard acids.")
            # 集合过滤掉的蛋白质
            dropout_proteins = set(dropout_proteins['Too long'] + dropout_proteins['Too short'])
            # + dropout_proteins['With non-standard acids'])

            # 删除不合格的蛋白质和ppi
            ppis = [ppi for ppi in ppis if ppi.fst_pro not in dropout_proteins and ppi.sec_pro not in dropout_proteins]
            acid_seqs = {key: value for key, value in acid_seqs.items() if key not in dropout_proteins}

        with open(os.path.join(args.processed_dir, "pairs", pairfn), "w") as f:
            f.writelines([str(ppi) for ppi in ppis])
        with open(os.path.join(args.processed_dir, "seqs", pairfn.split(".")[0] + ".fasta"), "w") as f:
            f.writelines([f"{pro}\t{sequence}\n" for pro, sequence in acid_seqs.items()])

        # print statistics
        print(f"============{pairfn.split('.')[0]} Dataset Statistics============")
        print(f'Total {len(ppis)} positive samples:')
        print(f'\t Positive: {len([ppi for ppi in ppis if ppi.label == 1])}')
        print(f'\t Negative: {len([ppi for ppi in ppis if ppi.label == 0])}')

        print(f"Total {len(acid_seqs)}  Proteins:" )
        print(f"\tMax length of protein: {max([pro[1] for pro in protein_lengths])}")
        print(f"\tMin length of protein: {min([pro[1] for pro in protein_lengths])}")
        print(f"\tAvg length of protein: {round(sum([pro[1] for pro in protein_lengths])/len(protein_lengths), 3)}")
