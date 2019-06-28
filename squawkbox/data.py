from collections import defaultdict
import logging

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Total Tokens: 360
SPECIAL_TOKENS = ['pad', 'start', 'end', 'continue']
NOTE_EVENTS = ['note:%i:%i' % (i, j) for i in range(128) for j in [0, 60]]
WAIT_EVENTS = ['wait:%i' % i for i in range(1, 101)]

IDX_TO_TOKEN = [*SPECIAL_TOKENS, *NOTE_EVENTS, *WAIT_EVENTS]
TOKEN_TO_IDX = {token: i for i, token in enumerate(IDX_TO_TOKEN)}


def pad_and_combine_instances(batch):
    """
    A collate function for padding and combining instance dictionaries.
    """
    batch_size = len(batch)
    max_field_lengths = defaultdict(int)
    for instance in batch:
        for field, tensor in instance.items():
            if len(tensor.size()) == 0:
                continue
            elif len(tensor.size()) == 1:
                max_field_lengths[field] = max(max_field_lengths[field], tensor.size()[0])
            elif len(tensor.size()) > 1:
                raise ValueError('Padding multi-dimensional tensors not supported')

    out_dict = {}
    for i, instance in enumerate(batch):
        for field, tensor in instance.items():
            if field not in out_dict:
                if field in max_field_lengths:
                    size = (batch_size, max_field_lengths[field])
                else:
                    size = (batch_size,)
                out_dict[field] = tensor.new_zeros(size)
            if field in max_field_lengths:
                out_dict[field][i, :tensor.size()[0]] = tensor
            else:
                out_dict[field][i] = tensor

    return out_dict


class MidiDataset(Dataset):
    def __init__(self,
                 file_path,
                 transforms=None,
                 embedding_type='wallclock'):
        """
        Loads tokenized MIDI data into tensors, and optionally applies some transformation during training

        Parameters
        ==========
        embedding_type : ``str``
            Method for measuring timestamps. Options:  'wallclock', 'positional'
        """
        assert embedding_type in {'wallclock', 'positional'}
        self._instances = self.read_instances(file_path)
        self._transforms = transforms
        self._embedding_type = embedding_type

    def __getitem__(self, idx):

        tokens = self._instances[idx]

        if self._transforms is not None:
            for transform in self._transforms:
                tokens = transform(tokens)

        if self._embedding_type == 'wallclock':
            timestamps = []
            current_time = 0
            for token in tokens:
                timestamps.append(current_time)
                token_type, *values = token.split(':')
                if token_type == 'wait':
                    current_time += int(values[0])
        elif self._embedding_type == 'positional':
            timestamps = list(range(len(tokens)))
        else:
            raise ConfigurationError('Bad embedding type. You must have '
                                     'modified the dataset reader...')

        tokens = [TOKEN_TO_IDX[x] for x in tokens]
        instance = {
            'src': torch.tensor(tokens[:-1], dtype=torch.int64),
            'tgt': torch.tensor(tokens[1:], dtype=torch.int64),
            'timestamp': torch.tensor(timestamps[:-1], dtype=torch.float32),
        }


        return instance

    def __len__(self):
        return len(self._instances)

    def read_instances(self, file_path):
        """Load MidiDataset from a file"""

        # TODO: Add download/cached path stuff.

        instances = []
        with open(file_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                instances.append(tokens)
        logger.debug('Max seq. len: %i', max(len(x) for x in instances))

        return instances
