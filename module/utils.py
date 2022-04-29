import random
import torch
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_padding_mask(seq_len, max_len):
    """根据长度获取transformer中的mask"""
    padding_mask = torch.arange(max_len).view(1, -1).repeat(seq_len.size(0), 1) # B x L
    padding_mask = padding_mask.to(seq_len.device)
    padding_mask = padding_mask >= seq_len.view(-1, 1)
    padding_mask.requres_grad = False
    return padding_mask     # B x L


def get_pro_rep(encs, lens):
    """获取蛋白质序列的表示，使用AVGPool，将编码压缩成1"""
    padding_mask = get_padding_mask(lens, max_len=encs.size(1))
    rep = encs * (1.-padding_mask.type_as(encs)).unsqueeze(-1)
    rep = torch.sum(rep, dim=1)
    rep = torch.div(rep, lens.unsqueeze(-1))
    return rep
