"""
Command-line utility for training a model
"""
import logging
import os
from pathlib import Path
import shutil
import sys

from apex import amp
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from squawkbox.data import MidiDataset, pad_and_combine_instances
from squawkbox.models import Model
from squawkbox.optim import Optimizer, LRScheduler
from squawkbox.transform import Transform


logger = logging.getLogger(__name__)


def _train(args):
    """Training function"""
    # Front matter, fail early if possible
    if not args.config.exists():
        logger.error('Config does not exist. Exiting.')
        sys.exit(1)

    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    if args.output_dir.exists() and not (args.resume or args.force):
        logger.error('Directory "%s" already exists. Exiting.' % str(args.output_dir))
        sys.exit(1)
    else:
        logger.info('Creating directory "%s"', args.output_dir)
        if not args.output_dir.exists():
            args.output_dir.mkdir()
        shutil.copy(args.config, args.output_dir / 'config.yaml')

    # Set up logging
    fh = logging.FileHandler(args.output_dir / 'output.log')
    logging.getLogger().addHandler(fh)

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
        logger.info('Using cuda')
        if args.cuda_device is not None:
            model = model.cuda(args.cuda_device)
        else:
            model = model.cuda()

        if args.fp16:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

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

    train_config = config['training']

    transforms = train_config.get('transforms', [])
    transforms = [Transform.from_config(x) for x in transforms]

    train_dataset = MidiDataset(config['train_data'], transforms=transforms)
    validation_dataset = MidiDataset(config['validation_data'])

    step = 0
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
        optimizer.zero_grad()
        for instance in train_tqdm:
            if args.cuda:
                instance = {key: value.cuda(args.cuda_device) for key, value in instance.items()}

            output_dict = model(**instance)
            loss = output_dict['loss']
            if args.fp16:
                with amp.scale_loss(loss,  optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            step += 1
            if not step % train_config.get('accumulation_steps', 1):
                optimizer.step()
                optimizer.zero_grad()

            train_tqdm.set_description('Loss: %0.4f' % loss.item())

        # Validation loop
        logger.info('Validating...')
        model.eval()
        validation_loader = DataLoader(validation_dataset,
                                       batch_size=train_config['batch_size'],
                                       shuffle=False,
                                       num_workers=train_config.get('num_workers', 0),
                                       collate_fn=pad_and_combine_instances)
        validation_tqdm = tqdm(validation_loader, desc='Loss: NA')
        total_validation_loss = 0
        n_tokens = 0
        for instance in validation_tqdm:
            if args.cuda:
                instance = {key: value.cuda(args.cuda_device) for key, value in instance.items()}

            output_dict = model(**instance)
            loss = output_dict['loss']
            n_tokens_in_batch = instance['src'].ne(0).sum().item()
            total_validation_loss += loss.item() * n_tokens_in_batch
            n_tokens += n_tokens_in_batch
            validation_tqdm.set_description('Instance Loss: %0.4f - Total Loss: %0.4f' % (loss.item(), total_validation_loss / n_tokens))
        metric = total_validation_loss / n_tokens
        logger.info('Validation Loss: %0.4f', metric)

        # Checkpoint
        model_state_dict = model.state_dict()
        state_dict = {
            'model': model_state_dict,
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
            best_metric = metric
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
    parser.add_argument('--cuda-device', type=int, help='CUDA device num', default=None)
    parser.add_argument('--fp16', action='store_true',
                        help='Enables half precision training')
    parser.add_argument('-r', '--resume', action='store_true',
                        help='will continue training existing checkpoint')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite existing checkpoint')
    args, _ = parser.parse_known_args()

    if os.environ.get("DEBUG"):
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=level)

    _train(args)
