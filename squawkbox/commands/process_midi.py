"""
Command line utility for tokenizing / detokinizing MIDI.
"""
from collections import deque
import logging
import os
from typing import Deque, IO, List, Tuple

from overrides import overrides

from squawkbox.midi import MidiFile


logger = logging.getLogger(__name__)


def _tokenize(args):
    with open(args.input, 'rb') as f:
        midi_file = MidiFile.load(f)
    raise NotImplementedError


def _detokenize(args):
    raise NotImplementedError


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
    tokenize_parser.set_defaults(func=_tokenize)

    detokenize_description = 'converts a sequence of tokens to a MIDI file'
    detokenize_parser = subparsers.add_parser('detokenize',
                                              description=detokenize_description,
                                              help=detokenize_description)
    detokenize_parser.add_argument('input', type=str, help='path to input .txt file')
    detokenize_parser.add_argument('output', type=str, help='path to output .midi file')
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
