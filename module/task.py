import torch
import logging
from fairseq.tasks import register_task, LegacyFairseqTask

from module.reader import PPIDataset
from module.utils import setup_seed

logger = logging.getLogger(__name__)


@register_task("ppi")
class PPITask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        # data reader arguments
        parser.add_argument("--data-dir", type=str)
        parser.add_argument("--max-len", type=int)
    
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        setup_seed(args.seed)

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)
    
    def load_dataset(self, split, combine=False, **kwargs):
        if 'train' in split:
            buffer_size = 8000
        else:
            buffer_size = 0
        self.datasets[split] = PPIDataset(split, buffer_size, self.args)
    
    def reduce_metrics(self, logging_outputs, criterion):
        criterion.__class__.reduce_metrics(logging_outputs)
    
    def begin_epoch(self, epoch, model):
        for key in self.datasets:
            if "train" in key:
                self.datasets[key].shuffle()