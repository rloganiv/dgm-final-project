"""
Command line utility for sampling from a trained model.
"""
from collections import deque
import csv
import logging
import os
from pathlib import Path
from typing import Deque, IO, List, Tuple
import time
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
import sys

from squawkbox.midi import Midi
from squawkbox.tokenizer import Tokenizer
from squawkbox.models import Model
from squawkbox.modules.sampler import Sampler
from squawkbox.data import IDX_TO_TOKEN, MidiDataset, pad_and_combine_instances

logger = logging.getLogger(__name__)

def _sample(args):

    model_path = args.model_dir / "model.pt"
    config_path = args.model_dir / "config.yaml"
    samples_folder = args.model_dir / "samples"# / str(time.time())

    if not config_path.exists():
        logger.error('Config does not exist. Exiting.')
        sys.exit(1)
    if not model_path.exists():
        logger.error('Model does not exist. Exiting.')
        sys.exit(1)

    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    if not samples_folder.exists():
        logger.info('Creating directory "%s"', samples_folder)
        samples_folder.mkdir()

    samples_folder = args.out

    if not samples_folder.exists():
        logger.info('Creating directory "%s"', samples_folder)
        samples_folder.mkdir()

    # torch.manual_seed(config.get('seed', 5150))
    # np.random.seed(config.get('seed', 1336) + 1)

    model = Model.from_config(config['model'])
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict["model"])
    if args.cuda:
        logger.info('Using cuda')
        if args.cuda_device is not None:
            dev = torch.device("cuda:{}".format(args.cuda_device))
        else:
            dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    model = model.to(dev)

    if args.conditional is not None:
        dataset = MidiDataset(
            args.conditional,
            transforms=None,
            embedding_type=config["training"].get("embedding_type", "wallclock")
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0 ,
            collate_fn=pad_and_combine_instances
        )
        instance = next(iter(loader))
        src = instance["src"][:,:args.conditional_len].expand(args.num_samples,-1).to(dev)
        timestamps = instance["timestamp"][:,:args.conditional_len].expand(args.num_samples,-1).to(dev)
    else:
        src = None
        timestamps = None

    logger.info("Generating samples")
    sampler = Sampler(
        decoder = model,
        embedding_type = config["training"].get("embedding_type", "wallclock"),
        temp = args.temperature,
        top_k = args.top_k,
        top_p = args.top_p,
        max_length = args.max_length
    )
    sampler.to(dev)

    samples = sampler(
        src = src,
        timestamps = timestamps,
        batch_size = args.num_samples,
        dev = dev
    )

    tokenizer = Tokenizer(scale=args.scale)
    logger.info("Samples generated, saving in {}".format(samples_folder))
    for i, sample in enumerate(samples):
        tokens = " ".join(sample)
        with open(samples_folder / "sample_{}.midi".format(i) , 'wb') as midi_file:
            midi_file.write(tokenizer.detokenize(tokens))

    if args.conditional is not None:
        original_ids = instance["src"].squeeze().tolist()
        original_toks = [IDX_TO_TOKEN[idx] for idx in original_ids]
        tokens = " ".join(original_toks)
        with open(samples_folder / "original.midi", 'wb') as midi_file:
            midi_file.write(tokenizer.detokenize(tokens))

    logger.info("Saving complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=Path, help='path to directory containing model checkpoint and .yaml config file')
    parser.add_argument('out', type=Path, help='subfolder name to hold this batch of samples')
    parser.add_argument('--max_length', type=int, help="max length of a sample", default=4096)
    parser.add_argument('--scale', type=int, help="max length of a sample", default=1)
    parser.add_argument('--num_samples', type=int, help='number of samples to generate', default=32)
    parser.add_argument('--temperature', type=float, help='float value for temperature based sampling', default=None)
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--cuda-device', type=int, help='CUDA device num', default=None)
    parser.add_argument('--top-k', type=int, help='K for top-k based sampling scheme.', default=None)
    parser.add_argument('--top-p', type=float, help='P for top-p (nucleus) based sampling scheme.', default=None)
    parser.add_argument('--conditional', type=Path, help="path to .txt file containing tokenized a midi file to condition samples on - will only condition on first midi sequence")
    parser.add_argument('--conditional_len', type=int, help="number of tokens in conditional sequence to condition on", default=512)
    args, _ = parser.parse_known_args()

    if os.environ.get("DEBUG"):
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=level)

    _sample(args)

