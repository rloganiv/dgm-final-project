"""
Command line utility for tokenizing / detokinizing MIDI.
"""
from collections import deque
import csv
import logging
import os
from pathlib import Path
from typing import Deque, IO, List, Tuple

from overrides import overrides
from tqdm import tqdm

from squawkbox.midi import Midi
from squawkbox.tokenizer import Tokenizer


logger = logging.getLogger(__name__)


def _tokenize(args):
    tokenizer = Tokenizer(scale=args.scale, max_tokens=args.max_tokens)
    with open(args.input, 'rb') as midi_file:
        midi = Midi.load(midi_file)
    with open(args.output, 'w') as token_file:
        token_file.write(tokenizer.tokenize(midi))


def _process_maestro(args):
    tokenizer = Tokenizer(scale=args.scale, max_tokens=args.max_tokens)

    if not args.csv.exists():
        raise IOError('"%s" does not exist. Terminating', args.csv)

    if args.root_dir is None:
        root_dir = args.csv.parents[0]
    else:
        root_dir = args.root_dir

    if not args.output_dir.exists():
        logger.info('Creating directory %s', args.output_dir)
        args.output_dir.mkdir(parents=True)

    if (args.output_dir / 'train.txt').exists():
        raise RuntimeError('Train data already exists. Terminating.')
    if (args.output_dir / 'validation.txt').exists():
        raise RuntimeError('Validation data already exists. Terminating.')
    if (args.output_dir / 'test.txt').exists():
        raise RuntimeError('Test data already exists. Terminating.')

    with open(args.csv) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in tqdm(reader):
            fname = root_dir / row['midi_filename']
            split = row['split']
            with open(fname, 'rb') as midi_file:
                midi = Midi.load(midi_file)
            with open(args.output_dir / (split + '.txt'), 'a') as token_file:
                token_file.write(tokenizer.tokenize(midi) + '\n')


def _detokenize(args):
    logger.warning('Detokenizer currently only supports writing NoteOn events.')
    tokenizer = Tokenizer(scale=args.scale)
    with open(args.input, 'r') as token_file:
        tokens = token_file.read()
    with open(args.output, 'wb') as midi_file:
        midi_file.write(tokenizer.detokenize(tokens))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='command line utility for tokenizing / detokenizing MIDI files')
    subparsers = parser.add_subparsers(title='commands', metavar='')

    tokenize_description = 'converts a MIDI file to a sequence of tokens'
    tokenize_parser = subparsers.add_parser('tokenize',
                                            description=tokenize_description,
                                            help=tokenize_description)
    tokenize_parser.add_argument('input', type=str, help='path to input .midi file')
    tokenize_parser.add_argument('output', type=str, help='path to output .txt file')
    tokenize_parser.add_argument('--scale', type=int, default=1, help='scale factor')
    tokenize_parser.add_argument('--max-tokens', type=int, default=None, help='max tokens')
    tokenize_parser.set_defaults(func=_tokenize)

    batch_tokenize_description = 'tokenizes/splits the entire MAESTRO dataset'
    batch_tokenize_parser = subparsers.add_parser('process-maestro',
                                                  description=batch_tokenize_description,
                                                  help=batch_tokenize_description)
    batch_tokenize_parser.add_argument('csv', type=Path, help='path to .csv file')
    batch_tokenize_parser.add_argument('-o', '--output_dir', type=Path, default=Path('.'),
                                       help='directory to serialize output to')
    batch_tokenize_parser.add_argument('--root_dir', type=Path, default=None,
                                       help='directory containing MIDI files; '
                                            'default behavior is to use the directory ' 'containing the csv')
    batch_tokenize_parser.add_argument('--scale', type=int, default=1, help='scale factor')
    batch_tokenize_parser.add_argument('--max-tokens', type=int, default=None, help='max tokens')
    batch_tokenize_parser.set_defaults(func=_process_maestro)

    detokenize_description = 'converts a sequence of tokens to a MIDI file'
    detokenize_parser = subparsers.add_parser('detokenize',
                                              description=detokenize_description,
                                              help=detokenize_description)
    detokenize_parser.add_argument('input', type=str, help='path to input .txt file')
    detokenize_parser.add_argument('output', type=str, help='path to output .midi file')
    detokenize_parser.add_argument('--scale', type=int, default=1, help='scale factor')
    detokenize_parser.set_defaults(func=_detokenize)

    args = parser.parse_args()

    if os.environ.get("DEBUG"):
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=level)

    if 'func' in dir(args):
        args.func(args)
    else:
        parser.print_help()
