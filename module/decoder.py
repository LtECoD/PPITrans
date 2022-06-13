import torch.nn as nn
from fairseq.models import BaseFairseqModel

from module.utils import get_pro_rep


class Decoder(BaseFairseqModel):
    def __init__(self, args):
        super().__init__()
        self.projector = nn.Linear(args.hid_dim, 2)
        if hasattr(args, "fuse_out"):
            self.fuse_out = args.fuse_out
        else:
            self.fuse_out = False
        if self.fuse_out:
            self.rep_maker = nn.Sequential(
                nn.Linear(args.emb_dim+args.hid_dim, args.hid_dim),
                nn.ReLU())

    def forward(self, fst_encs, fst_lens, sec_encs, sec_lens):

        fst_reps = get_pro_rep(fst_encs, fst_lens)      # B x D
        sec_reps = get_pro_rep(sec_encs, sec_lens)
        reps = fst_reps * sec_reps

        logits = self.projector(reps)
        return {"logits": logits, "fst_reps": fst_reps, 'sec_reps': sec_reps, "reps": reps}
