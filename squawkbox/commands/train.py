"""
Command-line utility for training a model
"""
import logging
import os
from pathlib import Path
import sys

import yaml

from squawkbox.models import Model
from squawkbox.data import MidiDataset, pad_and_combine_instances


logger = logging.getLogger(__name__)


def load_model(config):
    registry_key = config.pop('type')
    model_class = Model.get(registry_key)
    return model_class(**config)


def load_optimizer(config):
    pass


def _train(args):
    """Training function"""
    # Exit early if specified
    if args.output_dir.exists() and not args.resume:
        logger.error('Directory "%s" already exists. Exiting.')
        sys.exit(1)
    config = yaml.load(args.config)
    model = load_model(config['model'])
    optimizer = load_optimizer(config['optimizer'])

    checkpoint_path = args.output_dir / 'checkpoint.pt'
    if (args.output_dir / 'checkpoint.pt').exists() and args.resume:
        logger.info('Found existing checkpoint. Resuming training.')
        state_dict = torch.load(checkpoint_path)
        start_epoch = state_dict['epoch']
        best_metric = state_dict['best_metric']
        o
    else:
        logger.info('No existing checkpoint. Training from scratch.')
        start_epoch = 0

    train_dataset = MidiDataset(config['train_path'])
    validation_dataset = MidiDataset(config['validation_path'])




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path, help='path to config .yaml file')
    parser.add_argument('output_dir', type=Path, help='output directory to save model to')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('-r', '--resume', action='store_true',
                        help='will continue training existing checkpoint')
    args, _ = parser.parse_known_args()

    if os.environ.get("DEBUG"):
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=level)

    _train(args)