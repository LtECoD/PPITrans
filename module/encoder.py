import torch
import torch.nn as nn
from fairseq.models import BaseFairseqModel

from module.utils import get_padding_mask


class SimpleEncoder(BaseFairseqModel):
    """不使用cnn和transformer，只进行维度的转换"""
    def __init__(self, args):
        super().__init__()
        self.linear = nn.Linear(args.emb_dim, args.hid_dim)
    
    def forward(self, fst_embs, fst_lens, sec_embs, sec_lens):
        fst_encs = self.linear(fst_embs)
        sec_encs = self.linear(sec_embs)
        return fst_encs, fst_lens, sec_encs, sec_lens
        

class BaselineEncoder(BaseFairseqModel):

    @staticmethod
    def add_args(parser):
        parser.add_argument("--cnn-layers", type=int)
        parser.add_argument("--kernel-size", type=int)
        parser.add_argument("--trans-layers", type=int)

    def __init__(self, args):
        super().__init__()
        self.kernel_size = args.kernel_size
        self.max_len = args.max_len
        init_cnn_block = nn.Sequential(
            nn.Conv1d(in_channels=args.emb_dim, out_channels=args.hid_dim, \
                kernel_size=self.kernel_size),
            nn.ReLU(),
            nn.Conv1d(in_channels=args.hid_dim, out_channels=args.hid_dim, \
                    kernel_size=self.kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(num_features=args.hid_dim))
        cnn_blocks = [
            nn.Sequential(
                nn.Conv1d(in_channels=args.hid_dim, out_channels=args.hid_dim, \
                    kernel_size=self.kernel_size),
                nn.ReLU(),
                nn.Conv1d(in_channels=args.hid_dim, out_channels=args.hid_dim, \
                    kernel_size=self.kernel_size),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.BatchNorm1d(num_features=args.hid_dim)) \
            for idx in range(args.cnn_layers-1)]
        self.cnn_blocks = nn.Sequential(init_cnn_block, *cnn_blocks)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=args.hid_dim, nhead=4, \
                dim_feedforward=args.hid_dim*4, dropout=args.dropout, batch_first=True), 
            num_layers=args.trans_layers)

    def forward(self, fst_embs, fst_lens, sec_embs, sec_lens):
        """
            fst_embs: bsz x max_len x emb_dim
            sec_embs: bsz x max_len x emb_dim
        """
        fst_embs = fst_embs.permute(0, 2, 1)
        sec_embs = sec_embs.permute(0, 2, 1)

        assert fst_embs.size(2) == sec_embs.size(2) == self.max_len
        fst_cnn_encs = self.cnn_blocks(fst_embs)
        sec_cnn_encs = self.cnn_blocks(sec_embs)
        for idx in range(len(self.cnn_blocks)):
            fst_lens = fst_lens - 2 * (self.kernel_size - 1)
            fst_lens = torch.floor(fst_lens / 2).long()

            sec_lens = sec_lens - 2 * (self.kernel_size - 1)
            sec_lens = torch.floor(sec_lens / 2).long()
        assert fst_cnn_encs.size(2) == sec_cnn_encs.size(2)

        # 转置回来
        fst_cnn_encs = fst_cnn_encs.permute(0, 2, 1)
        sec_cnn_encs = sec_cnn_encs.permute(0, 2, 1)

        fst_padding_mask = get_padding_mask(fst_lens, fst_cnn_encs.size(1))
        sec_padding_mask = get_padding_mask(sec_lens, fst_cnn_encs.size(1))

        fst_encs = self.transformer(fst_cnn_encs, src_key_padding_mask=fst_padding_mask)
        sec_encs = self.transformer(sec_cnn_encs, src_key_padding_mask=sec_padding_mask)
        return fst_encs, fst_lens, sec_encs, sec_lens
