import torch
from squawkbox.models.baseline import Baseline


def test_baseline():
    baseline = Baseline(vocab_size=7, embedding_dim=2, num_lstm_units=2, num_lstm_layers=2)
    src = torch.tensor([[1,2,3,0],[3,4,5,6]])
    tgt = torch.tensor([[2,3,0,0],[4,5,6,1]])
    out_dict = baseline(src=src, tgt=tgt)
    out_dict['loss'].backward()
    for p in baseline.parameters():
        assert p.grad is not None
    assert out_dict['logits'][0, 2:4].abs().sum() == 0.0
