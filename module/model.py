from fairseq.models import register_model
from fairseq.models import register_model_architecture
from fairseq.models import BaseFairseqModel

from module.encoder import SimpleEncoder
from module.encoder import BaselineEncoder
from module.decoder import BaselineDecoder


class BaseModel(BaseFairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def add_args(parser):
        # model arguments
        parser.add_argument("--dropout", type=float)
        parser.add_argument("--emb-dim", type=int)
        parser.add_argument("--hid-dim", type=int)

    def forward(self, fst_embs, fst_lens, sec_embs, sec_lens):
        fst_encs, fst_lens, sec_encs, sec_lens = self.encoder(
            fst_embs, fst_lens, sec_embs, sec_lens)

        out = self.decoder(fst_encs=fst_encs, fst_lens=fst_lens, \
            sec_encs=sec_encs, sec_lens=sec_lens)
        return out


@register_model('simplebaseline')
class SimpleBaseline(BaseModel):
    @classmethod
    def build_model(cls, args, task):
        encoder = SimpleEncoder(args)
        decoder = BaselineDecoder(args)
        return cls(encoder, decoder)


@register_model('baseline')
class Baseline(BaseModel):
    @classmethod
    def build_model(cls, args, task):
        encoder = BaselineEncoder(args)
        decoder = BaselineDecoder(args)
        return cls(encoder, decoder)


@register_model_architecture("baseline", "baseline")
def baseline_architecture(args):
    pass

@register_model_architecture("simplebaseline", "simplebaseline")
def simplebaseline_architecture(args):
    pass