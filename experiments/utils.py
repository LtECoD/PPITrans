import os
import torch
import re
import itertools
import numpy as np
from typing import Counter

from module.model import PPIModel
from module.model import RNNModel

organisms = ['ecoli', 'mouse', 'fly', 'worm', 'yeast', 'human']
standard_acids = [
        ('A', 1), ('C', 6), ('D', 5), ('E', 7), ('F', 2), 
        ('G', 1), ('H', 4), ('I', 2), ('K', 5), ('L', 2),
        ('M', 3), ('N', 4), ('P', 2), ('Q', 7), ('R', 4),
        ('S', 3), ('T', 3), ('V', 1), ('W', 4), ('Y', 3), ('X', 0)]
acids = [acid_type[0] for acid_type in standard_acids]
types = [acid_type[1] for acid_type in standard_acids]

acids_vocab = {k[0]: idx+1 for idx, k in enumerate(standard_acids)}


def build_window_vocab(keep_order=True, including_center=False):
    """windows size is 3"""
    iter_num = 3 if including_center else 2
    vocab = ["".join(w) for w in itertools.product(acids, repeat=iter_num)]
    start_vocab = ["^"+"".join(w) for w in list(itertools.product(acids, repeat=iter_num-1))]
    end_vocab = ["".join(w)+"&" for w in list(itertools.product(acids, repeat=iter_num-1))]
    vocab = start_vocab + vocab + end_vocab

    if not keep_order:
        vocab = set(["".join(sorted(list(w))) for w in vocab])
        vocab = list(vocab)

    vocab = dict(zip(vocab, list(range(len(vocab)))))
    return vocab


class Protein:
    def __init__(self, name, seq):
        self.name = name
        if seq is not None:
            self.seq = re.sub(r"[UZOB]", "X", seq)
        else:
            self.seq = None
        self.length = len(seq) if seq is not None else None
        self.ss = None      # secondary structure
        self.emb = None 
        self.cm = None      # contact map

    def count_aa(self):
        counter = Counter(list(self.seq))
        counts = [0] * len(acids)
        for aa, value in counter.items():
            counts[acids.index(aa)] = value
        assert sum(counts) == self.length
        self.aa_freq = np.array([v/self.length for v in counts])

    def set_ss(self, ss):
        assert len(ss) == len(self.seq)
        self.ss = ss

    def set_emb(self, emb):
        self.emb = emb
        if self.length is not None:
            assert self.length == len(emb)
        else:
            self.length = len(emb)

    def set_cm(self, cm):
        assert self.length == cm.shape[0]
        self.cm = cm

    def discretize(self):
        """打散序列"""
        return list(self.seq), self.emb

    def __str__(self):
        _str = f"{self.name}\t{self.seq}\n"
        return _str
    

def load_model(model_dir):
    model_name = os.path.basename(model_dir)
    # 加载模型
    if "ppi" in model_name:
        model_class = PPIModel
    elif "rnn" in model_name:
        model_class = RNNModel
    else:
        raise NotImplementedError 

    state_dict = torch.load(os.path.join(model_dir, 'checkpoint_best.pt'))
    args = state_dict["args"]
    model = model_class.build_model(args, None)
    model.load_state_dict(state_dict["model"])
    model.eval()
    print(model)
    return model


def forward_kth_translayer(model, emb, k):
    length = torch.LongTensor([len(emb)])
    with torch.no_grad():
        emb = torch.Tensor(emb).unsqueeze(0)           # 1 x L x D
        enc, _ = model.encoder.forward_kth_translayer(emb, length, k)
    return enc.squeeze(0).detach().numpy()


def lookup_embed(pro, embeder):
    aa_list = list(pro.seq)
    ids = torch.LongTensor([acids_vocab[a] for a in aa_list]).unsqueeze(0)
    ids = ids.to(embeder.aa_embeder.weight.device)
    embed = embeder(ids).squeeze(0).detach().numpy()
    return embed 