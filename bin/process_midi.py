"""
Command line utility for tokenizing / detokinizing MIDI.
"""
import logging
import os
from typing import IO

HEADER_IDENTIFIER = b'MThd'
TRACK_IDENTIFIER = b'MTrk'

logger = logging.getLogger(__name__)


class MidiError(Exception): pass


class MidiObject:
    """
    An object for storing parsed MIDI information.
    """
    def __init__(self, header=None, tracks=None):
        self._header = header
        self._tracks = tracks

    def update(self, identifier: bytes, chunk: bytes):
        if identifier == HEADER_IDENTIFIER:
            self.header = MidiHeader.from_bytes(chunk)
            logging.debug(f'Parsed Header: {self.header}')
        else:
            midi_track = MidiTrack.from_bytes(chunk, header=self.header)


class MidiHeader:
    def __init__(self, format_type, ntracks, pulses_per_quarter_note):
        self.format_type = format_type
        self.ntracks = ntracks
        self.pulses_per_quarter_note = pulses_per_quarter_note

    def __repr__(self):
        return (f'MidiHeader(format_type={self.format_type}, '
                        f'ntracks={self.ntracks}, '
                        f'ppqn={self.pulses_per_quarter_note})')

    @classmethod
    def from_bytes(cls, chunk):
        format_type = int.from_bytes(chunk[0:2], byteorder='big')
        ntracks = int.from_bytes(chunk[2:4], byteorder='big')
        tickdiv = int.from_bytes(chunk[4:6], byteorder='big')
        timing_interval = (tickdiv >> 16) & 1

        # TODO: Handle timecode timining intervals
        if timing_interval == 1:
            raise NotImplementedError('Timecode timing intervals not supported.')

        pulses_per_quarter_note = 0
        for i in range(15):
            pulses_per_quarter_note += tickdiv >> i
        pulses_per_quarter_note = pulses_per_quarter_note

        return cls(format_type, ntracks, pulses_per_quarter_note)


class MidiTrack:
    def __init__(self):
        pass

    @classmethod
    def from_bytes(cls, chunk, header=None):
        if header is None:
            raise MidiError('Cannot parse a MIDI track info without a header.')
        if header.format_type != 1:
            raise NotImplementedError('Format 0 and 2 files not supported.')
        import pdb; pdb.set_trace()
        for byte in chunk:
            pass





def generate_chunks(file: IO[bytes]):
    """
    Generates chunks from a MIDI file

    Parameters
    ==========
    file : ``IO[bytes]``
        A file-like object containing bytes to be read.
    """
    while True:
        identifier = file.read(4)
        if identifier == b'':
            break
        chunklen = int.from_bytes(file.read(4), byteorder='big')
        chunk = file.read(chunklen)
        yield identifier, chunk


def _tokenize(args):
    midi_object = MidiObject()

    with open(args.input, 'rb') as f:
        for identifier, chunk in generate_chunks(f):
            midi_object.update(identifier, chunk)


def _detokenize(args):
    pass


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
