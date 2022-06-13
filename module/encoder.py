import torch
import torch.nn as nn
from fairseq.models import BaseFairseqModel

from module.utils import get_padding_mask


class Embedder(nn.Module):
    def __init__(self, args):
        super(Embedder, self).__init__()
        self.aa_embeder = nn.Embedding(num_embeddings=22, \
            embedding_dim=args.emb_dim, padding_idx=0)
        self.pos_embeder = nn.Embedding(num_embeddings=args.max_len+10,
            embedding_dim=args.emb_dim, padding_idx=0)

    def forward(self, seqs, lens=None):
        bsz, maxlen = seqs.size()
        pos = torch.arange(maxlen) + 1
        pos = pos.unsqueeze(0).repeat(bsz, 1).to(seqs.device)
        if lens is not None:
            pos[pos > lens.unsqueeze(-1)] = 0
        return self.aa_embeder(seqs) + self.pos_embeder(pos)


class NaiveEncoder(BaseFairseqModel):
    """transformer，只进行维度的转换"""
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
        self.embeder = Embedder(args)

    def forward(self, fst_seqs, fst_lens, sec_seqs, sec_lens):
        fst_embs = self.embeder(fst_seqs, fst_lens)
        sec_embs = self.embeder(sec_seqs, sec_lens)
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
        self.num_layers = self.transformer.num_layers

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

        fst_encs, fst_lens = self.forward_transformer(fst_encs, fst_lens)
        sec_encs, sec_lens = self.forward_transformer(sec_encs, sec_lens)

        return fst_encs, fst_lens, sec_encs, sec_lens


class FullEncoder(Encoder):
    def __init__(self, args):
        super().__init__(args)
        self.embeder = Embedder(args)

    def forward(self, fst_seqs, fst_lens, sec_seqs, sec_lens):
        fst_embs = self.embeder(fst_seqs, fst_lens)
        sec_embs = self.embeder(sec_seqs, sec_lens)
        return super().forward(fst_embs, fst_lens, sec_embs, sec_lens)


class RNN(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super(RNN, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hid_dim, batch_first=True, bidirectional=True)
        self.out_proj = nn.Linear(hid_dim*3, hid_dim)

    def forward(self, encs, lens):
        # encs:  B x L x D
        out = torch.nn.utils.rnn.pack_padded_sequence(encs, lens.cpu(), True, False)
        out, _ = self.gru(out)     # b x l x 2d
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, True, total_length=encs.size(1))
        out = torch.cat((out, encs), dim=-1)
        out = self.out_proj(out)
        return out, lens


class RNNEncoder(BaseFairseqModel):
    def __init__(self, args):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(args.emb_dim, args.hid_dim, bias=False),
            nn.Dropout(args.dropout),
            nn.LayerNorm(args.hid_dim))
        self.layers = nn.Sequential(
            RNN(args.hid_dim, args.hid_dim), 
            RNN(args.hid_dim, args.hid_dim),
            RNN(args.hid_dim, args.hid_dim),
            RNN(args.hid_dim, args.hid_dim),
            RNN(args.hid_dim, args.hid_dim),
            RNN(args.hid_dim, args.hid_dim))
        self.num_layers = len(self.layers)

    def forward_projecter(self, embs):
        # embs: B x L x D
        embs = self.projector(embs)
        return embs
    
    def forward_kth_translayer(self, encs, lens, k):
        encs, lens = self.layers[k](encs, lens)
        return encs, lens
        
    def forward(self, fst_embs, fst_lens, sec_embs, sec_lens):
        """
            fst_embs: bsz x max_len x emb_dim
            sec_embs: bsz x max_len x emb_dim
        """
        fst_encs = self.forward_projecter(fst_embs)
        sec_encs = self.forward_projecter(sec_embs)

        # fst_encs, fst_lens = self.layers(fst_encs, fst_lens)
        # sec_encs, sec_lens = self.layers(sec_encs, sec_lens)

        for k in range(len(self.layers)):
            fst_encs, fst_lens = self.forward_kth_translayer(fst_encs, fst_lens, k)
            sec_encs, sec_lens = self.forward_kth_translayer(sec_encs, sec_lens, k)

        return fst_encs, fst_lens, sec_encs, sec_lens


class FullRNNEncoder(RNNEncoder):
    def __init__(self, args):
        super().__init__(args)
        self.embeder = Embedder(args)

    def forward(self, fst_seqs, fst_lens, sec_seqs, sec_lens):
        fst_embs = self.embeder(fst_seqs, fst_lens)
        sec_embs = self.embeder(sec_seqs, sec_lens)
        return super().forward(fst_embs, fst_lens, sec_embs, sec_lens)