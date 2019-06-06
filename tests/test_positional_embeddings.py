from math import pi

import torch

from squawkbox.modules.positional_embedding import PositionalEmbedding


def test_positional_embedding():
    # Check freq's
    postional_embedding = PositionalEmbedding(dim=4)
    expected_freq = torch.tensor([[[1, 1 / 100]]])
    assert torch.equal(postional_embedding.freq, expected_freq)

    # Check embeddings
    timestamp = torch.tensor([[0, 2 * pi, 100* 2 * pi]])
    output = postional_embedding(timestamp)
    _, seq_len, dim = output.shape
    assert seq_len == 3
    assert dim == 4
    assert output[0,0,0] == 0
    assert output[0,0,1] == 0
    assert output[0,0,2] == 1
    assert output[0,0,3] == 1
    assert output[0,1,2] == 1
    assert output[0,2,2] == 1
    assert output[0,2,3] == 1
