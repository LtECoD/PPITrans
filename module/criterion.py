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

        output = model(inputs)
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
