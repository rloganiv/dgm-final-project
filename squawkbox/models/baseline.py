import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from squawkbox.models import Model


@Model.register('baseline')
class Baseline(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_lstm_units, num_lstm_layers, padding=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding = padding

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=num_lstm_units,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        self.h2o = nn.Linear(num_lstm_units, vocab_size)

    def forward(self, src, tgt=None, hidden=None, **kwargs):
        """
        return logits and loss
        :param src: (batch_size, seq_len)
        :param tgt: (batch_size, seq_len)
        :param hidden: (batch_size, embdding_dim) or None
        :return:


        """

        if tgt is not None:
            masks = tgt.ne(self.padding).float()
        else:
            masks = src.ne(self.padding).float()

        seq_lens = masks.sum(1)
        seq_lens, perm_idx = seq_lens.sort(0, descending=True)
        embeddings = self.embedding(src)[perm_idx]
        if tgt is not None:
            tgt = tgt[perm_idx]

        # pack_padded_sequence
        packed = pack_padded_sequence(embeddings, lengths=seq_lens, batch_first=True)
        if hidden is None:
            lstm_output, hidden_out = self.lstm(packed)
        else:
            lstm_output, hidden_out = self.lstm(packed, hidden)
        # pad_packed_sequence
        lstm_output, _ = pad_packed_sequence(lstm_output, batch_first=True)

        # output = output.contiguous()
        logits = self.h2o(lstm_output) * masks.unsqueeze(-1) # batch_size, seq_len, voc_len
        output = {'logits': logits, 'hidden': hidden_out}

        if tgt is not None:
            ll = F.log_softmax(logits, dim=-1).gather(2,tgt.unsqueeze(-1))
            ll = ll * masks.unsqueeze(-1)
            output['loss'] = -ll.mean(dim=1).mean(dim=0)

        return output

