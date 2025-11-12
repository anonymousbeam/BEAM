from torch.nn import Linear, Module, init
from torch.nn.functional import gelu

class BEAM(Module):
    def __init__(self, original_attention_block, device, dtype, rank=4, alpha=4):
        super().__init__()
        self.original_attention_block = original_attention_block
        self.hidden_dim = original_attention_block.query_key_value.in_features if hasattr(original_attention_block, "query_key_value") else original_attention_block.config.hidden_size
        self.rank = rank

        # BEAM adapter weights
        self.beam_down = Linear(
            self.hidden_dim, rank, bias=False, device=device, dtype=dtype
        )
        self.beam_up = Linear(
            rank, self.hidden_dim, bias=False, device=device, dtype=dtype
        )

        # BEAM scalar network
        self.beam_scale_layer = Linear(
            self.hidden_dim, 2*self.rank, bias=True, device=device, dtype=dtype
        )
        self.beam_scale_pred = Linear(
            2*self.rank, 1, bias=True, device=device, dtype=dtype 
        )

        self.scale = rank // alpha

        init.normal_(self.beam_down.weight, std=0.01)
        init.normal_(self.beam_up.weight, std=0.01)
        init.normal_(self.beam_scale_layer.weight, std=0.01)
        init.normal_(self.beam_scale_pred.weight, std=0.01)

    def forward(self, hidden_states, *args, **kwargs):
        out = self.original_attention_block(hidden_states, *args, **kwargs)
        hidden_states, *rest = out

        # Compute BEAM output
        beam_output = self.beam_up((self.beam_down(hidden_states)))
        # Use the last hidden state to compute scalar
        last_hidden_state = hidden_states[:, -1, :]  # shape: (batch_size, hidden_dim)
        
        scale_features = gelu(self.beam_scale_layer(last_hidden_state))  # shape: (batch_size, rank)
        scale = self.beam_scale_pred(scale_features).squeeze(-1)
        self.scale = scale

        # Apply scale and add to hidden states
        hidden_states = hidden_states + scale.view(-1, 1, 1) * beam_output

        return (hidden_states, *rest)