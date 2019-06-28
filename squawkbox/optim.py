from pytorch_pretrained_bert.optimization import BertAdam
import torch

from squawkbox.utils import Registrable


class Optimizer(Registrable, torch.optim.Optimizer):
    pass


Registrable._registry[Optimizer] = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "bert-adam": BertAdam
}


class LRScheduler(Registrable, torch.optim.lr_scheduler._LRScheduler):
    pass


Registrable._registry[LRScheduler] = {
    "lambda": torch.optim.lr_scheduler.LambdaLR,
    "step": torch.optim.lr_scheduler.StepLR
}
