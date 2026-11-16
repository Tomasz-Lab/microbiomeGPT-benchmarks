"""[purbancz] Transformer stacks shared by the transformer-based components."""

from torch import Tensor, nn

from gut_former.models.components._initialization import xavier_init_


class TransformerStack(nn.Module):
    """[purbancz] A stack of transformer layers with the same settings, run in order."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        num_layers: int,
        num_heads: int,
        dropout: float,
        feedforward_dim: int,
    ):
        super().__init__()
        for name, value in (
            ("num_layers", num_layers),
            ("num_heads", num_heads),
            ("feedforward_dim", feedforward_dim),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}.")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embedding_dim = embedding_dim
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # [purbancz] Xavier for every linear map in the layers, as in GutFormer.
        for layer in self.layers:
            assert isinstance(layer, nn.TransformerEncoderLayer)
            xavier_init_(layer.self_attn.out_proj, layer.linear1, layer.linear2)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """[purbancz] (batch_size, sequence_length, embedding_dim), same shape out."""
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
