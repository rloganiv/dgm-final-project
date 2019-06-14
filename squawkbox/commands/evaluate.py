"""
Command-line utitlity for evaluating a model
"""
import logging
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from squawkbox.data import MidiDataset, pad_and_combine_instances
from squawkbox.models import Model


logger = logging.getLogger(__name__)


def _evaluate(args):
    if not args.config.exists():
        logger.error('Config does not exist. Exiting.')
        sys.exit(1)

    if not args.checkpoint.exists():
        logger.error('Checkpoint does not exist. Exiting.')
        sys.exit(1)

    if not args.data.exists():
        logger.error('Data does not exist. Exiting.')
        sys.exit(1)

    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file)

    torch.manual_seed(config.get('seed', 5150))
    np.random.seed(config.get('seed', 1336) + 1)

    # Initialize model components from config
    model = Model.from_config(config['model'])
    if args.cuda:
        logger.info('Using cuda')
        if args.cuda_device is not None:
            model = model.cuda(args.cuda_device)
        else:
            model = model.cuda()

    logger.info('Restoring model checkpoint.')
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict['model'])

    train_config = config['training']
    embedding_type = train_config.get('embedding_type', 'wallclock')

    dataset = MidiDataset(args.data,
                          embedding_type=embedding_type)
    data_loader = DataLoader(dataset,
                             batch_size=train_config['batch_size'],
                             shuffle=False,
                             num_workers=train_config.get('num_workers', 0),
                             collate_fn=pad_and_combine_instances)

    # Evaluation loop
    logger.info('Evaluating')
    total_validation_loss = 0
    n_tokens = 0
    for instance in tqdm(data_loader):
        if args.cuda:
            instance = {key: value.cuda(args.cuda_device) for key, value in instance.items()}
        with torch.no_grad():
            output_dict = model(**instance)
            loss = output_dict['loss']
        n_tokens_in_batch = instance['src'].ne(0).sum().item()
        total_validation_loss += loss.item() * n_tokens_in_batch
        n_tokens += n_tokens_in_batch
    metric = total_validation_loss / n_tokens
    logger.info('Loss: %0.4f', metric)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path, help='path to config .yaml file')
    parser.add_argument('checkpoint', type=Path, help='model checkpoint')
    parser.add_argument('data', type=Path, help='evaluation data')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--cuda-device', type=int, help='CUDA device num', default=None)
    args, _ = parser.parse_known_args()

    if os.environ.get("DEBUG"):
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=level)

    _evaluate(args)
