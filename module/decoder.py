import torch
import torch
import torch.nn as nn
from fairseq.models import BaseFairseqModel

from module.utils import get_padding_mask


class ContactMapInsector:
    def __init__(self):
        pass

    def __call__(self, fst_encs, fst_lens, sec_encs, sec_lens):
        """encs: B x L x D"""
        fst_padding_mask = get_padding_mask(fst_lens, max_len=fst_encs.size(1))
        sec_padding_mask = get_padding_mask(sec_lens, max_len=sec_encs.size(1))
        #! todo
        raise NotImplementedError



class BaselineDecoder(BaseFairseqModel):
    def __init__(self, args):
        super().__init__()
        self.projector = nn.Linear(args.hid_dim, 2)

    def get_pro_rep(self, encs, lens):
        """获取蛋白质序列的表示，使用AVGPool，将编码压缩成1"""
        padding_mask = get_padding_mask(lens, max_len=encs.size(1))
        rep = encs * (1.-padding_mask.float()).unsqueeze(-1)
        rep = torch.sum(rep, dim=1)
        rep = torch.div(rep, lens.unsqueeze(-1))
        return rep

    def forward(self, fst_encs, fst_lens, sec_encs, sec_lens):
        fst_rep = self.get_pro_rep(fst_encs, fst_lens)      # B x D
        sec_rep = self.get_pro_rep(sec_encs, sec_lens)      # B x D
        rep = fst_rep * sec_rep
        logits = self.projector(rep)
        return {"logits": logits}
