from fairseq.models import register_model
from fairseq.models import register_model_architecture
from fairseq.models import BaseFairseqModel

from module.encoder import NaiveEncoder, NaiveFullEncoder
from module.encoder import Encoder, FullEncoder
from module.decoder import Decoder


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
        parser.add_argument("--wo-ppm", action="store_true", \
            help="whether without pretrained models")
        parser.add_argument("--trans-layers", type=int)

    def forward(self, inputs):
        fst_encs, fst_lens, sec_encs, sec_lens = self.encoder(**inputs)
        out = self.decoder(fst_encs=fst_encs, fst_lens=fst_lens, \
            sec_encs=sec_encs, sec_lens=sec_lens)
        return out


@register_model('naive_ppi_model')
class NaivePPIModel(BaseModel):
    """解码器只包括一个简单的线性层"""
    @classmethod
    def build_model(cls, args, task):
        encoder = NaiveFullEncoder(args) if args.wo_ppm else NaiveEncoder(args)
        decoder = Decoder(args)
        return cls(encoder, decoder)


@register_model('ppi_model')
class PPIModel(BaseModel):
    @classmethod
    def build_model(cls, args, task):
        encoder = FullEncoder(args) if args.wo_ppm else Encoder(args)
        decoder = Decoder(args)
        return cls(encoder, decoder)


from module.encoder import RNNEncoder, FullRNNEncoder
@register_model("rnn_model")
class RNNModel(BaseModel):
    @classmethod
    def build_model(cls, args, task):
        encoder = FullRNNEncoder(args) if args.wo_ppm else RNNEncoder(args)
        decoder = Decoder(args)
        return cls(encoder, decoder)


@register_model_architecture("naive_ppi_model", "naive_ppi")
def naive_ppi_architecture(args):
    args.wo_ppm = getattr(args, 'wo_ppm', False)


@register_model_architecture("ppi_model", "ppi")
def ppi_architecture(args):
    args.wo_ppm = getattr(args, 'wo_ppm', False)


@register_model_architecture("rnn_model", "rnn")
def pipr_architecture(args):
    args.wo_ppm = getattr(args, 'wo_ppm', False)
