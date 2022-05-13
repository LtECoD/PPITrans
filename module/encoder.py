from functools import total_ordering
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


class RCNN(nn.Module):
    def __init__(self, input_dim, hid_dim, pool_kernel_size=2, only_conv=False):
        super(RCNN, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=input_dim, out_channels=hid_dim, kernel_size=3, stride=1, padding="same")
        self.only_conv = only_conv
        if not only_conv:
            self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size)
            self.gru = nn.GRU(
                input_size=hid_dim, hidden_size=hid_dim, batch_first=True, bidirectional=True)

    def forward(self, encs, lens):
        # encs:  B x L x D
        encs = encs.permute(0, 2, 1)
        encs = self.conv(encs)
        if not self.only_conv:
            encs = self.pool(encs)
            lens = torch.floor(lens/self.pool.kernel_size)
        else:
            lens= lens
        encs = encs.permute(0, 2, 1)

        if not self.only_conv:
            # padded
            out = torch.nn.utils.rnn.pack_padded_sequence(encs, lens.cpu(), True, False)
            out, _ = self.gru(out)     # b x l x 2d
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, True, total_length=encs.size(1))
            out = torch.cat((out, encs), dim=-1)
        else:
            out = encs
        return out, lens


class PIPREncoder(BaseFairseqModel):
    def __init__(self, args):
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(args.emb_dim, args.hid_dim, bias=False),
            nn.Dropout(args.dropout),
            nn.LayerNorm(args.hid_dim))
        self.layers = nn.Sequential(
            RCNN(args.hid_dim, args.hid_dim, 2), 
            # RCNN(3*args.hid_dim, args.hid_dim, 2),
            RCNN(3*args.hid_dim, args.hid_dim, 2),
            RCNN(3*args.hid_dim, args.hid_dim, 2),
            RCNN(3*args.hid_dim, args.hid_dim, 2),
            RCNN(3*args.hid_dim, args.hid_dim, None, True))
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

        for k in range(len(self.layers)):
            fst_encs, fst_lens = self.forward_kth_translayer(fst_encs, fst_lens, k)
            sec_encs, sec_lens = self.forward_kth_translayer(sec_encs, sec_lens, k)

        return fst_encs, fst_lens, sec_encs, sec_lens


class FullPIPREncoder(PIPREncoder):
    def __init__(self, args):
        super().__init__(args)
        self.embeder = Embedder(args)

    def forward(self, fst_seqs, fst_lens, sec_seqs, sec_lens):
        fst_embs = self.embeder(fst_seqs, fst_lens)
        sec_embs = self.embeder(sec_seqs, sec_lens)
        return super().forward(fst_embs, fst_lens, sec_embs, sec_lens)