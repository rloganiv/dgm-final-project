"""
Command-line utility for training a model
"""
import logging
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from squawkbox.data import MidiDataset, pad_and_combine_instances
from squawkbox.models import Model
from squawkbox.optim import Optimizer, LRScheduler


logger = logging.getLogger(__name__)


def _train(args):
    """Training function"""
    # Front matter
    if args.output_dir.exists() and not args.resume:
        logger.error('Directory "%s" already exists. Exiting.')
        sys.exit(1)
    else:
        logger.info('Creating directory "%s"', args.output_dir)
        args.output_dir.mkdir()
        shutil.copy(args.config, args.output_dir / 'config.yaml')

    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    torch.manual_seed(config.get('seed', 5150))
    np.random.seed(config.get('seed', 1336) + 1)

    # Initialize model components from config
    model = Model.from_config(config['model'])
    optimizer = Optimizer.from_config(config['optimizer'], params=model.parameters())
    if 'lr_scheduler' in config:
        lr_scheduler = LRScheduler.from_config(config['lr_scheduler'])
    else:
        lr_scheduler = None

    if args.cuda:
        model = model.cuda()

    # Restore checkpoint
    checkpoint_path = args.output_dir / 'checkpoint.pt'
    best_checkpoint_path = args.output_dir / 'model.pt'
    if (args.output_dir / 'checkpoint.pt').exists() and args.resume:
        logger.info('Found existing checkpoint. Resuming training.')
        state_dict = torch.load(checkpoint_path)
        start_epoch = state_dict['epoch']
        best_metric = state_dict['best_metric']
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
    else:
        logger.info('No existing checkpoint. Training from scratch.')
        start_epoch = 0
        best_metric = float('inf')

    train_dataset = MidiDataset(config['train_data'])
    validation_dataset = MidiDataset(config['validation_data'])

    train_config = config['training']
    for epoch in range(start_epoch, train_config['epochs']):
        logger.info('Epoch: %i', epoch)

        # Training loop
        logger.info('Training...')
        model.train()
        train_loader = DataLoader(train_dataset,
                                  batch_size=train_config['batch_size'],
                                  shuffle=True,
                                  num_workers=train_config.get('num_workers', 0),
                                  collate_fn=pad_and_combine_instances)
        train_tqdm = tqdm(train_loader, desc='Loss: NA')
        for instance in train_tqdm:
            if args.cuda:
                instance = {key: value.cuda() for key, value in instance.items()}

            instance_chunks = {key: torch.split(value, train_config['chunk_size'], dim=1)
                              for key, value in instance.items()}

            output_dict = {"hidden": None}
            for chunk_id in range(len(instance_chunks['src'])):
                instance_chunk = {key: value[chunk_id] for key, value in instance_chunks.items()}
                optimizer.zero_grad()
                output_dict = model(hidden = output_dict["hidden"], **instance_chunk)
                loss = output_dict['loss']
                loss.backward()
                optimizer.step()
                output_dict['hidden'].detach()
            train_tqdm.set_description('Loss: %0.4f' % loss.item())

        # Validation loop
        logger.info('Validating...')
        model.eval()
        validation_loader = DataLoader(validation_dataset,
                                       batch_size=train_config['batch_size'],
                                       shuffle=True,
                                       num_workers=train_config.get('num_workers', 0),
                                       collate_fn=pad_and_combine_instances)
        validation_tqdm = tqdm(validation_loader, desc='Loss: NA')
        total_validation_loss = 0
        number_of_instances = 0
        for instance in validation_tqdm:
            if args.cuda:
                instance = {key: value.cuda() for key, value in instance.items()}
            output_dict = model(**instance)
            loss = output_dict['loss']
            total_validation_loss += loss.item()
            number_of_instances += train_config['batch_size']
            validation_tqdm.set_description('Loss: %0.4f' % (total_validation_loss / number_of_instances))
        metric = total_validation_loss / number_of_instances
        logger.info('Validation Loss: %0.4f', metric)

        # Checkpoint
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_metric': best_metric
        }
        if lr_scheduler:
            state_dict['lr_scheduler'] = lr_scheduler.state_dict()

        # If model is best encountered overwrite best model checkpoint.
        if metric < best_metric:
            logger.info('Best model so far.')
            state_dict['best_metric'] = metric
            torch.save(state_dict, best_checkpoint_path)

        # Save current model.
        torch.save(state_dict, checkpoint_path)

    if lr_scheduler is not None:
        lr_scheduler.step()


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
