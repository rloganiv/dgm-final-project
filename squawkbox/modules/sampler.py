import torch
import torch.nn as nn
import torch.nn.functional as F

from squawkbox.data import TOKEN_TO_IDX, IDX_TO_TOKEN

class Sampler(nn.Module):

    def __init__(self, decoder, embedding_type, temp=None, top_k=None, top_p=None, max_length=4096):
        super(Sampler, self).__init__()

        self.decoder = decoder
        self.temp = temp
        self.top_k = top_k
        self.top_p = top_p
        self.embedding_type = embedding_type
        self.max_length = max_length
        self.SOS = TOKEN_TO_IDX["start"]
        self.EOS = TOKEN_TO_IDX["end"]
        self.padding = TOKEN_TO_IDX["pad"]
        self.cont = TOKEN_TO_IDX["continue"]
        self.pos_adjustment = nn.Embedding(len(IDX_TO_TOKEN), 1)

    def _delta_time(self, sample, dev):
        tokens = [IDX_TO_TOKEN[x] for x in sample.squeeze().tolist()]
        waits = [int(x.split(':')[-1]) if 'wait' in x else 0 for x in tokens]
        return torch.tensor(waits, device=dev,dtype=torch.float32).unsqueeze(-1)


    def _temper(self, logits):
        """
        Adjusts the logged probabilities by a temperature. If the temperature is 1, they probabilities are unchanged. Returns probabilities, not logits
        """
        if self.temp is None:
            t = 1
        else:
            t = self.temp

        return F.softmax(logits / t, dim = -1)

    def _sample(self, probs):
        """
        Sample based on probs with shape (batch_size, vocab_size) with a random sampling scheme.
        Returns sample of type LongTensor and shape (batch_size, 1).
        """
        # position zero is an invalid item to sample
        probs[:, 0] = 0.0
        return torch.multinomial(probs, 1)

    def _sample_top_k(self, probs):
        """
        Sample based on probs with shape (batch_size, vocab_size) with a top k scheme
        Returns sample of type LongTensor and shape (batch_size, 1).
        """
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
        new_distribution = sorted_probs[:, :self.top_k] / sorted_probs.sum(-1, keepdim=True)
        sample_indices = self._sample(new_distribution)
        sample = sorted_indices.gather(-1, sample_indices)
        return sample


    def _sample_top_p(self, probs):
        """
        Sample based on probs with shape (batch_size, vocab_size) with a top p scheme
        Returns sample of type LongTensor and shape (batch_size, 1).
        """
        sorted_probs, sorted_indices = probs.sort(dim=-1)
        cum_probs = probs.cumsum(dim=-1)
        sorted_mask = (sorted_probs <= self.top_p).type(torch.float32)
        masked_probs = probs * torch.zeros_like(probs).scatter(1, sorted_indices, sorted_mask)
        return self._sample(masked_probs)

    def _to_tokens(self, src):
        out = []
        for sequence in src:
            tokens = [IDX_TO_TOKEN[i] for i in sequence.tolist()]
            if 'end' in tokens:
                cutoff = tokens.index('end')
                out.append(tokens[:cutoff])
            else:
                out.append(tokens)
        return out

    def forward(self, src=None, timestamps=None, batch_size=None, dev=None, **kwargs):
        """
        Generate samples either conditioned on 'src' or from nothing. Either both 'src' and 'timestamps'
        must be specified or 'batch_size'. If all three are specified, the 'batch_size' will be
        overridden by the size of the first dimenion of the two tensors.

        Input:
        - 'src': A LongTensor of shape (batch_size, sequence_length)
        - 'timestamps': A FloatTensor of shape (batch_size, sequence_length)
        - 'batch_size': An int specifying how many samples to generate.
        Output:
        - A list of samples, each sample being a list of str tokens.
        """

        if src is not None:
            assert(timestamps is not None)
            assert((len(src.shape) == len(timestamps.shape)) and
                   (src.shape[0] == timestamps.shape[0]) and
                   (src.shape[1] == timestamps.shape[1]))
            batch_size = src.shape[0]
        else:
            assert(batch_size is not None)
            src = torch.LongTensor([[self.SOS]]).expand(batch_size, 1).to(dev)
            timestamps = torch.zeros(batch_size, 1, dtype=torch.float32).to(dev)

        hidden = None
        for _ in range(self.max_length):

            with torch.no_grad():
                output = self.decoder(src, timestamps=timestamps)

            # get the last step's logits for the next step and temper them
            probs = self._temper(output["logits"][:, -1, :])

            if self.top_k is not None:
                sample = self._sample_top_k(probs)
            elif self.top_p is not None:
                sample = self._sample_top_p(probs)
            else:
                sample = self._sample(probs)

            src = torch.cat((src, sample), -1)

            if self.embedding_type == 'positional':
                new_timestamp = timestamps[:, -1].unsqueeze(-1) + 1
            elif self.embedding_type == 'wallclock':
                delta = self._delta_time(sample, dev)
                new_timestamp = timestamps[:, -1].unsqueeze(-1) + delta
            timestamps = torch.cat((timestamps, new_timestamp), -1)

        samples = self._to_tokens(src)

        return samples

