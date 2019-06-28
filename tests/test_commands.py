import argparse
from pathlib import Path
import shutil
import tempfile
from unittest import TestCase

import torch

from squawkbox.commands.train import _train
from squawkbox.models import Model


@Model.register('null')
class NullModel(Model):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(*args, **kwargs):
        return {'loss': torch.tensor(0.0, requires_grad=True),
                'hidden': None}


class TestTrainCommand(TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.gettempdir()) / 'TESTOUT'
        self.args = argparse.Namespace()
        self.args.config = Path('tests/fixtures/config.yaml')
        self.args.output_dir = self.tmp_dir
        self.args.cuda = False
        self.args.cuda_device = None
        self.args.fp16 = False
        self.args.resume = False

    def test_runs(self):
        _train(self.args)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
