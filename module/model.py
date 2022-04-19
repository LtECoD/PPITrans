import torch
from fairseq.models import register_model
from fairseq.models import register_model_architecture
from fairseq.models import BaseFairseqModel

from module.encoder import SimpleEncoder
from module.encoder import BaselineEncoder
from module.decoder import BaselineDecoder
from module.utils import get_pro_rep


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
        parser.add_argument("--cnn-layers", type=int)
        parser.add_argument("--kernel-size", type=int)
        parser.add_argument("--trans-layers", type=int)

    def forward(self, fst_embs, fst_lens, sec_embs, sec_lens):
        fst_encs, fst_lens, sec_encs, sec_lens = self.encoder(
            fst_embs, fst_lens, sec_embs, sec_lens)

        out = self.decoder(fst_encs=fst_encs, fst_lens=fst_lens, \
            sec_encs=sec_encs, sec_lens=sec_lens)
        return out


@register_model('simplebaseline')
class SimpleBaseline(BaseModel):
    """解码器只包括一个简单的线性层"""
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


# 解码器为空，只返回蛋白质对的表示
@register_model('contrastive')
class Contrastive(BaseModel):
    @classmethod
    def build_model(cls, args, task):
        encoder = BaselineEncoder(args)
        return cls(encoder, None)
    
    def forward(self, fst_embs, fst_lens, sec_embs, sec_lens):
        fst_encs, fst_lens, sec_encs, sec_lens = self.encoder(
            fst_embs, fst_lens, sec_embs, sec_lens)
        fst_reps = get_pro_rep(fst_encs, fst_lens)      # B x D
        sec_reps = get_pro_rep(sec_encs, sec_lens)      # B x D
        reps = fst_reps * sec_reps
        return {"reps": reps, "fst_reps": fst_reps, 'sec_reps': sec_reps}


@register_model("contrastiveppi")
class ContrastivePPI(BaseModel):
    @staticmethod
    def add_args(parser):
        BaseModel.add_args(parser)
        parser.add_argument("--cmodel-path", type=str, \
            help="encoder pretrained with supervised contrastive loss")

    @classmethod
    def build_model(cls, args, task):
        encoder = BaselineEncoder(args)
        #! 加载encoder
        if hasattr(args, "cmodel_path"):
            state = torch.load(args.cmodel_path, map_location="cpu")['model']
            encoder.load_state_dict({k.replace("encoder.", ""):v for k, v in state.items()})
            #! 取消禁止所有参数更新
            # for param in encoder.parameters():
            #     param.requires_grad = False

        decoder = BaselineDecoder(args)
        return cls(encoder, decoder)


@register_model_architecture('contrastive', 'contrastive')
def contrastive_architecture(args):
    pass


@register_model_architecture('contrastiveppi', 'contrastiveppi')
def contrastive_ppi_architecture(args):
    pass


@register_model_architecture("baseline", "baseline")
def baseline_architecture(args):
    pass

@register_model_architecture("simplebaseline", "simplebaseline")
def simplebaseline_architecture(args):
    pass