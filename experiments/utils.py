import os
import torch

organisms = ['ecoli', 'mouse', 'fly', 'worm', 'yeast', 'human']
standard_acids = [
        ('A', 1), ('C', 6), ('D', 5), ('E', 7), ('F', 2), 
        ('G', 1), ('H', 4), ('I', 2), ('K', 5), ('L', 2),
        ('M', 3), ('N', 4), ('P', 2), ('Q', 7), ('R', 4),
        ('S', 3), ('T', 3), ('V', 1), ('W', 4), ('Y', 3), ('X', 0)]
acids = [acid_type[0] for acid_type in standard_acids]
types = [acid_type[1] for acid_type in standard_acids]


class Protein:
    def __init__(self, name, seq):
        self.name = name
        self.seq = seq
        self.pos_neighbors = []
        self.neg_neighbors = []
        self.length = len(seq) if seq is not None else None
        self.ss = None      # secondary structure
        self.emb = None 
        self.cm = None      # contact map

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

    def add_pos_neighbor(self, pro):
        assert isinstance(pro, Protein)
        self.pos_neighbors.append(pro)

    def add_neg_neighbor(self, pro):
        assert isinstance(pro, Protein)
        self.neg_neighbors.append(pro)

    def __str__(self):
        _str = f"{self.name}\t{self.seq}\n"
        _str = _str + f"{len(self.pos_neighbors)} interacted proteins:\n"
        for nei in self.pos_neighbors:
            _str = _str + f"\t{nei.name}\n"
        _str = _str + f"{len(self.neg_neighbors)} uninteracted proteins:\n"
        for nei in self.neg_neighbors:
            _str = _str + f"\t{nei.name}\n"
        return _str
    
def load_proteins(orga, _dir):
    pairs = open(os.path.join(_dir, orga+".pair"), "r").readlines()
    pairs = [p.strip().split() for p in pairs]

    seqs = open(os.path.join(_dir, orga+".seq")).readlines()
    seqs = dict([seq.strip().split() for seq in seqs])

    proteins = {}

    for (fst, sec, label) in pairs:
        if fst in proteins:
            pro = proteins[fst]
        else:
            pro = Protein(name=fst, seq=seqs[fst])
        
        nei_pro = Protein(name=sec, seq=seqs[sec])
        if int(label) == 1:
            pro.add_pos_neighbor(nei_pro)
        elif int(label) == 0:
            pro.add_neg_neighbor(nei_pro)
        else:
            raise NotImplementedError
        proteins[fst] = pro
    
    return list(proteins.values())


def load_model(model_class, ckpt_path):
    state_dict = torch.load(ckpt_path)
    args = state_dict["args"]
    model = model_class.build_model(args, None)
    model.load_state_dict(state_dict["model"])
    return model


def forward_kth_translayer(model, emb, k):
    length = torch.LongTensor([len(emb)])
    emb = torch.Tensor(emb).unsqueeze(0)           # 1 x L x D
    enc, _ = model.encoder.forward_kth_translayer(emb, length, k)
    return enc.squeeze(0).detach().numpy()

