from cProfile import label
import math
import torch
import torch.nn.functional as F

from fairseq import metrics
from fairseq.criterions import FairseqCriterion
from fairseq.criterions import register_criterion


@register_criterion("ppi_crossentropy")
class PPICrossEntropy(FairseqCriterion):
    def __init__(self):
        super().__init__(None)
    
    def forward(self, model, sample, reduce=True):
        inputs = sample["inputs"]
        labels = sample["labels"]

        output = model(**inputs)
        loss, metrics = self.compute_loss(output, labels, reduce=reduce)
        sample_size = len(labels)

        logging_output = {
            "sample_size": sample_size,
            "loss": loss.data}
        logging_output.update(metrics)
        return loss, sample_size, logging_output

    def compute_loss(self, output, labels, reduce=True):
        logits = output['logits']
        lprobs = F.log_softmax(logits, dim=-1)

        loss = F.nll_loss(
            lprobs, labels, reduction="sum" if reduce else "none")

        # 计算metrics
        preds = torch.argmax(lprobs, dim=-1)
        acc = (preds == labels).float().mean()

        if torch.sum(preds) > 0:
            pre = torch.sum(preds[labels.bool()]) / torch.sum(preds)
        else:
            pre = 1.
        if torch.sum(labels) > 0:
            rec = torch.sum(preds[labels.bool()]) / torch.sum(labels)
        else:
            rec = 1.
        metrics = {"acc": acc, "pre": pre, "rec": rec}

        return loss, metrics
    
    @staticmethod
    def reduce_metrics(logging_outputs):
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        acc_sum = sum(log.get("acc", 0) for log in logging_outputs)
        pre_sum = sum(log.get("pre", 0) for log in logging_outputs)
        rec_sum = sum(log.get("rec", 0) for log in logging_outputs)

        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("acc", acc_sum/len(logging_outputs), len(logging_outputs), round=3)
        metrics.log_scalar("pre", pre_sum/len(logging_outputs), len(logging_outputs), round=3)
        metrics.log_scalar("rec", rec_sum/len(logging_outputs), len(logging_outputs), round=3)


@register_criterion("contrastive")
class Contrastive(FairseqCriterion):
    """
    ! 对比损失设计两部分，蛋白质表示层面，蛋白质联合表示层面
    ! 蛋白质表示层面：
        ! 互作蛋白表示夹角小，非互作蛋白表示夹角大
    ! 蛋白质联合表示层面：
        ! 同类联合表示的夹角小，不同类联合表示夹角大，
    """
    def __init__(self, temp):
        super().__init__(None)
        self.temp = temp

    @classmethod
    def add_args(cls, parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--temp', type=float, default=1.)

    @classmethod
    def build_criterion(cls, args, task):        
        return cls(args.temp)

    def forward(self, model, sample, reduce=True):
        inputs = sample["inputs"]
        labels = sample["labels"]

        output = model(**inputs)
        sloss, intra_loss, inter_loss = self.compute_contrastive_loss(output, labels, reduce=reduce)
        loss = sloss + intra_loss + inter_loss
        sample_size = len(labels)

        logging_output = {
            "sample_size": sample_size,
            "loss": float(loss),
            'sloss': float(sloss),
            "intraloss": float(intra_loss), 
            "interloss": float(inter_loss),}
        return loss, sample_size, logging_output

    def compute_contrastive_loss(self, output, labels, reduce=True):
        fst_reps = output['fst_reps']       # B x D
        sec_reps = output['sec_reps']       # B x D
        cosine_similarity = F.cosine_similarity(fst_reps, sec_reps, dim=-1)     # B
        scaled_similarity = (cosine_similarity + 1.) / 2.
        sim_loss = torch.where(labels.bool(), 1.-scaled_similarity, scaled_similarity)
        sim_loss = torch.sum(sim_loss)

        #! 新方案，测试中
        reps = output['reps']
        cosine_similarity_matrix = F.cosine_similarity(reps.unsqueeze(1), reps.unsqueeze(0), dim=-1)    # B x B
        scaled_similarity_matrix = (cosine_similarity_matrix + 1.) / 2.
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
        # 同类样本之间的夹角尽量小
        mask.fill_diagonal_(0)
        intra_loss = torch.sum(scaled_similarity_matrix * mask, dim=-1)
        intra_loss = torch.div(intra_loss, torch.sum(mask, dim=-1)+1e-6)
        intra_loss = 1. - intra_loss
        intra_loss = torch.sum(intra_loss)
        # 不同类样本之间的夹角尽量大
        mask = 1. - mask
        mask.fill_diagonal_(0)
        inter_loss = torch.sum(scaled_similarity_matrix * mask, dim=-1)
        inter_loss = torch.div(inter_loss, torch.sum(mask, dim=-1)+1e-6)
        inter_loss = torch.sum(inter_loss)
        return sim_loss, intra_loss, inter_loss

        #! 旧方案，weight有错误，改为同类softmax尽量大
        # reps = output['reps']          # B x D
        # score = torch.div(torch.matmul(reps, reps.T), self.temp)    # B x B
        # max_value, _ = torch.max(score, dim=-1)
        # score = score - max_value.view(-1, 1).detach()
        # score.fill_diagonal_(float("-inf"))
        # score = torch.log_softmax(score, dim=-1)            # B x B

        # mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
        # score = torch.where(mask, score, torch.zeros_like(score))
        # score = torch.sum(score, dim=-1)
        # score = torch.div(score, torch.sum(mask, dim=-1)+1e-6)
        # score = -1. * torch.sum(score)
        # return sim_loss, score
    
    @staticmethod
    def reduce_metrics(logging_outputs):
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sloss_sum = sum(log.get("sloss", 0) for log in logging_outputs)
        intra_loss_sum = sum(log.get("intraloss", 0) for log in logging_outputs)
        inter_loss_sum = sum(log.get("interloss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("sloss", sloss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("intraloss", intra_loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("interloss", inter_loss_sum / sample_size, sample_size, round=3)
