import torch
import torch.nn.functional as F
from torch import nn


from squawkbox.modules import GPT2Model, GPT2LMHead, GPT2Config, PositionalEmbedding
from squawkbox.models import Model

@Model.register('gpt2')
class GPT2_Standard(nn.Module):
    """
    Standard gpt2 based model to be used for musenet.

    Parameters
    ==========
    n_ctx : ``int``
        integer multiple of chunk_size, how many previous hidden states to keep track of
    chunk_size : ``int``
        the number of steps to be processed in a single pass
    """
    def __init__(self,
                 vocab_size,
                 n_positions,
                 n_ctx,
                 n_embd,
                 n_layers,
                 n_head,
                 layer_norm_epsilon,
                 initializer_range,
                 padding=0,
                 **kwargs):

        super(GPT2_Standard, self).__init__()
        config = GPT2Config(
            vocab_size_or_config_json_file=vocab_size,
            n_positions=n_positions,
            n_ctx=n_ctx,
            n_embd=n_embd,
            n_layer=n_layers,
            n_head=n_head,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range
        )

        self.n_ctx = n_ctx
        self.padding = padding
        self.main_model = GPT2Model(config=config)
        self.lm_head = GPT2LMHead(model_embeddings_weights=self.main_model.wte.weight, config=config)

        # Overwrite the regular positional embeddings with custom ones
        self.main_model.wpe = PositionalEmbedding(n_embd)

    def forward(self,
                src,
                tgt=None,
                timestamps=None,
                hidden=None,
                **kwargs):
        hidden_out, presents = self.main_model(input_ids=src,
                                               position_ids=timestamps,
                                               token_type_ids=None)

        logits = self.lm_head(hidden_state=hidden_out)

        # only keep track of the last n_ctx position's computed hidden states
        output = {'logits': logits}
        if tgt is not None:
            loss_function = nn.CrossEntropyLoss(ignore_index=self.padding)
            ll = loss_function(logits.view(-1, logits.shape[-1]), tgt.view(-1))
            output['loss'] = ll

        return output
