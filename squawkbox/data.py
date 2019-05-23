import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = ['pad', 'start', 'end']
NOTE_EVENTS = ['note:%i:%i' % (i, j) for i in range(128) for j in range(128)]
WAIT_EVENTS = ['wait:%i' % i for i in range(1024)]  # Note: might need more waits
IDX_TO_TOKEN = [*SPECIAL_TOKENS, *NOTE_EVENTS, *WAIT_EVENTS]
TOKEN_TO_IDX = {token: i for i, token in enumerate(IDX_TO_TOKEN)}


class MidiDataset(Dataset):
    def __init__(self, file_path, transforms=None):
        """
        Loads tokenized MIDI data into tensors, and optionally applies some transformation during training

        Parameters
        ==========
        file_path
        """
        self._instances = self.read_instances(file_path)
        self._transforms = transforms

    def __getitem__(self, idx):
        instance = self._instances[idx]

        if self._transforms is not None:
            for transform in self._transforms:
                instance = transform(instance)

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

                timestamps = []
                current_time = 0
                for token in tokens:
                    timestamps.append(current_time)
                    token_type, *values = token.split(':')
                    if token_type == 'wait':
                        current_time += int(values[0])

                # TODO: MuseNet mentions using relative position embeddings, maybe do this.

                tokens = [TOKEN_TO_IDX[x] for x in line.split()]
                instance = {
                    'src': torch.LongTensor(tokens[:-1]),
                    'tgt': torch.LongTensor(tokens[1:]),
                    'timestamp': torch.FloatTensor(timestamps[:-1]),
                }
                instances.append(instance)

        return instances
