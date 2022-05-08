from sklearn.feature_selection import SelectPercentile
import torch
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
    
        if self.fuse_out:
            fst_reps = self.rep_maker(torch.cat((fst_encs[0], fst_encs[-1]), dim=-1))
            sec_reps = self.rep_maker(torch.cat((sec_encs[0], sec_encs[-1]), dim=-1))
            fst_reps = get_pro_rep(fst_reps, fst_lens)
            sec_reps = get_pro_rep(sec_reps, sec_lens)
            reps = fst_reps * sec_reps

            # fst_embs = get_pro_rep(fst_encs[0], fst_lens)
            # sec_embs = get_pro_rep(sec_encs[0], sec_lens)
            # emb_reps = fst_embs * sec_embs
            # reps = self.rep_maker(torch.cat((reps, emb_reps), dim=-1))
        else:
            fst_reps = get_pro_rep(fst_encs[-1], fst_lens)      # B x D
            sec_reps = get_pro_rep(sec_encs[-1], sec_lens)
            reps = fst_reps * sec_reps

        logits = self.projector(reps)
        return {"logits": logits, "fst_reps": fst_reps, 'sec_reps': sec_reps, "reps": reps}
