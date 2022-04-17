from atexit import register
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


@register_criterion("ppi_contrastive")
class PPI_Contrastive(PPICrossEntropy):
    """
    ! 对比损失设计两部分，蛋白质表示层面，蛋白质联合表示层面
    ! 蛋白质表示层面：尚未有具体方案
    ! 蛋白质联合表示层面：互作蛋白联合表示的夹角小，非互作蛋白夹角尽量大（用常规对比学习可实现），
    """
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    @classmethod
    def add_args(cls, parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--gamma', type=float)

    @classmethod
    def build_criterion(cls, args, task):        
        return cls(args.gamma)

    def forward(self, model, sample, reduce=True):
        inputs = sample["inputs"]
        labels = sample["labels"]

        output = model(**inputs)
        cross_entropy_loss, metrics = super().compute_loss(output, labels, reduce=reduce)
        superivsed_contrastive_loss = self.compute_contrastive_loss(output, labels, reduce=reduce)
        loss = cross_entropy_loss + self.gamma * superivsed_contrastive_loss
        sample_size = len(labels)

        logging_output = {
            "sample_size": sample_size,
            "loss": float(loss),
            "ce_loss": float(cross_entropy_loss),
            "sc_loss": float(superivsed_contrastive_loss)}
        logging_output.update(metrics)
        return loss, sample_size, logging_output

    def compute_contrastive_loss(self, output, labels, reduce=True):
        reps = output['reps']          # B x D
        score = torch.matmul(reps, reps.T)    # B x B
        score.fill_diagonal_(float("-inf"))
        score = torch.log_softmax(score, dim=-1)            # B x B

        #! 尝试只对负样本使用对比学习
        # if torch.sum(labels) <= 1:
        #     pos_score = 0.
        # else:
        #     weight = labels.view(1, -1).repeat(labels.size(0), 1)   # B x B
        #     weight = weight.fill_diagonal_(0).bool()
        #     pos_score = torch.where(weight, score, torch.zeros_like(score))     # B x B
        #     pos_score = torch.sum(pos_score, dim=-1)        # B                
        #     pos_score = torch.div(pos_score, torch.sum(weight, dim=-1))
        #     pos_score = -1 * torch.sum(pos_score)

        if torch.sum(1-labels) <= 1:
            neg_score = 0.
        else:
            weight = (1-labels).view(1, -1).repeat(labels.size(0), 1)   # B x B
            weight = weight.fill_diagonal_(0).bool()
            neg_score = torch.where(weight, score, torch.zeros_like(score))     # B x B
            neg_score = torch.sum(neg_score, dim=-1)        # B                
            neg_score = torch.div(neg_score, torch.sum(weight, dim=-1))
            neg_score = -1 * torch.sum(neg_score)

        return neg_score
    
    @staticmethod
    def reduce_metrics(logging_outputs):
        PPICrossEntropy.reduce_metrics(logging_outputs)
        ce_loss_sum = sum(log.get("ce_loss", 0) for log in logging_outputs)
        sc_loss_sum = sum(log.get("sc_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("celoss", ce_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("scloss", sc_loss_sum / sample_size / math.log(2), sample_size, round=3)
