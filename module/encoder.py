import torch.nn as nn
from fairseq.models import BaseFairseqModel

from module.utils import get_padding_mask


class NaiveEncoder(BaseFairseqModel):
    """不使用cnn和transformer，只进行维度的转换"""
    def __init__(self, args):
        super().__init__()
        self.linear = nn.Linear(args.emb_dim, args.hid_dim)
    
    def forward(self, fst_embs, fst_lens, sec_embs, sec_lens):
        fst_encs = self.linear(fst_embs)
        sec_encs = self.linear(sec_embs)
        return fst_encs, fst_lens, sec_encs, sec_lens


class NaiveFullEncoder(NaiveEncoder):
    def __init__(self, args):
        super().__init__(args)
        self.embeder = nn.Embedding(num_embeddings=22, \
            embedding_dim=args.emb_dim, padding_idx=0)
    
    def forward(self, fst_seqs, fst_lens, sec_seqs, sec_lens):
        fst_embs = self.embeder(fst_seqs)
        sec_embs = self.embeder(sec_seqs)
        return super().forward(fst_embs, fst_lens, sec_embs, sec_lens)


class Encoder(BaseFairseqModel):
    def __init__(self, args):
        super().__init__()
        self.max_len = args.max_len
  
        self.projector = nn.Sequential(
            nn.Linear(args.emb_dim, args.hid_dim, bias=False),
            nn.Dropout(args.dropout),
            nn.LayerNorm(args.hid_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=args.hid_dim, nhead=4, \
                dim_feedforward=args.hid_dim*4, dropout=args.dropout, batch_first=True), 
            num_layers=args.trans_layers)

    def forward_projecter(self, embs):
        # embs: B x L x D
        embs = self.projector(embs)
        return embs

    def forward_kth_translayer(self, encs, lens, k):
        padding_mask = get_padding_mask(lens, encs.size(1))
        encs = self.transformer.layers[k](encs, src_key_padding_mask=padding_mask)
        return encs, lens

    def forward_transformer(self, encs, lens):
        padding_mask = get_padding_mask(lens, encs.size(1))
        encs = self.transformer(encs, src_key_padding_mask=padding_mask)
        return encs, lens

    def forward(self, fst_embs, fst_lens, sec_embs, sec_lens):
        """
            fst_embs: bsz x max_len x emb_dim
            sec_embs: bsz x max_len x emb_dim
        """
        fst_encs = self.forward_projecter(fst_embs)
        sec_encs = self.forward_projecter(sec_embs)
        fst_encs_list = [fst_embs, fst_encs]
        sec_encs_list = [sec_embs, sec_encs]
        for k in range(self.transformer.num_layers):
            fst_encs, _ = self.forward_kth_translayer(fst_encs, fst_lens, k)
            sec_encs, _ = self.forward_kth_translayer(sec_encs, sec_lens, k)
            fst_encs_list.append(fst_encs)
            sec_encs_list.append(sec_encs)

        # fst_encs, fst_lens = self.forward_transformer(fst_embs, fst_lens)
        # sec_encs, sec_lens = self.forward_transformer(sec_embs, sec_lens)

        return fst_encs_list, fst_lens, sec_encs_list, sec_lens


class FullEncoder(Encoder):
    def __init__(self, args):
        super().__init__(args)
        self.embeder = nn.Embedding(num_embeddings=22, \
            embedding_dim=args.emb_dim, padding_idx=0)

    def forward(self, fst_seqs, fst_lens, sec_seqs, sec_lens):
        fst_embs = self.embeder(fst_seqs)
        sec_embs = self.embeder(sec_seqs)
        return super().forward(fst_embs, fst_lens, sec_embs, sec_lens)