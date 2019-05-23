import torch
from torch import nn

class PosEmbedding(nn.Module):

    def __init__(self, embedding_dim):
        super(PosEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x):
        pass

