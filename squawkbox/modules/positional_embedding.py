import math

import torch


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, dim):
        super(PositionalEmbedding, self).__init__()
        assert (dim % 2) == 0
        self.dim = dim
        half_dim = int(dim / 2)
        self.freq = 1 / 10000 ** ( torch.arange(0, half_dim, dtype=torch.float32).view(1, 1, half_dim) / half_dim)

    def forward(self, timestamp):
        timestamp = timestamp.unsqueeze(-1)
        x = torch.matmul(timestamp, self.freq)  # (batch_size, seq_len, dim / 2)
        sin = torch.sin(x)
        cos = torch.cos(x)
        return torch.cat((sin, cos), -1)
