import torch
import torch.nn as nn
from fairseq.models import BaseFairseqModel

from module.utils import get_padding_mask
from module.utils import get_pro_rep


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

    def forward(self, fst_encs, fst_lens, sec_encs, sec_lens):
        fst_reps = get_pro_rep(fst_encs, fst_lens)      # B x D
        sec_reps = get_pro_rep(sec_encs, sec_lens)      # B x D
        #! 由原来的乘改为了加，初步的实验效果貌似不如相乘，先加上对比损失，如果效果还不行
        #! 则改回相加
        reps = fst_reps * sec_reps
        logits = self.projector(reps)
        return {"logits": logits, "fst_reps": fst_reps, 'sec_reps': sec_reps, "reps": reps}
