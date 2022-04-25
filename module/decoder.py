import torch.nn as nn
from fairseq.models import BaseFairseqModel

from module.utils import get_pro_rep


class Decoder(BaseFairseqModel):
    def __init__(self, args):
        super().__init__()
        self.projector = nn.Linear(args.hid_dim, 2)

    def forward(self, fst_encs, fst_lens, sec_encs, sec_lens):
        fst_reps = get_pro_rep(fst_encs, fst_lens)      # B x D
        sec_reps = get_pro_rep(sec_encs, sec_lens)      # B x D
        
        reps = fst_reps * sec_reps
        logits = self.projector(reps)
        return {"logits": logits, "fst_reps": fst_reps, 'sec_reps': sec_reps, "reps": reps}
