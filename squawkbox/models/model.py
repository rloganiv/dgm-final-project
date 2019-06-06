import torch

from squawkbox.utils import Registrable


class Model(torch.nn.Module, Registrable):
    def __init__(self):
        super().__init__()
