"""[purbancz] The original Gut-FORMer model rebuilt from components (ADR 0006).

These classes must keep computing what the original `BacteriaModel` computed
(guarded by tests/gut_former/test_gut_former_reference.py). A change in behaviour
means a new class. The defaults are the values hardcoded in the original class.

As in the original model, every Linear (including the attention output projections) gets
Xavier initialization; embeddings, layer norms and attention input projections keep their defaults.

The "gut_former" in the name of this file should also stay after the package refactor.
"""

import torch
from torch import Tensor, nn

from gut_former.models.components import Decoder, Encoder, Head, Latent, Tokenizer
from gut_former.schema import DataSchema


def _transformer_stack(
    embedding_dim: int, num_layers: int, num_heads: int, dropout: float, feedforward_dim: int
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=embedding_dim,
        nhead=num_heads,
        dim_feedforward=feedforward_dim,
        dropout=dropout,
        batch_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers)


def _init_linear_layers(module: nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            nn.init.xavier_uniform_(submodule.weight)
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)


class GutFormerTokenizer(Tokenizer):
    """[purbancz] Token = shared projection of the value + identity embedding of the feature."""

    def __init__(self, data_schema: DataSchema, embedding_dim: int):
        super().__init__()
        self.value_projection = nn.Linear(1, embedding_dim)
        self.embeddings = nn.ModuleDict(
            {
                modality: nn.Embedding(len(names), embedding_dim)
                for modality, names in data_schema.feature_names.items()
            }
        )
        _init_linear_layers(self)

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            modality: self.value_projection(values) + self.feature_embeddings(modality)
            for modality, values in inputs.items()
        }

    def feature_embeddings(self, modality: str) -> Tensor:
        embedding = self.embeddings[modality]
        assert isinstance(embedding, nn.Embedding)
        return embedding.weight


class GutFormerEncoder(Encoder):
    """[purbancz] Concatenates all modalities, two transformer stacks."""

    def __init__(
        self,
        embedding_dim: int,
        num_layers: int = 1,
        num_heads: int = 2,
        dropout: float = 0.05,
        final_num_layers: int = 1,
        final_dropout: float = 0.1,
        feedforward_dim: int = 2048,
    ):
        super().__init__()
        self.stack = _transformer_stack(
            embedding_dim, num_layers, num_heads, dropout, feedforward_dim
        )
        # [purbancz] One head per embedding dimension, as in the original model
        self.final_stack = _transformer_stack(
            embedding_dim, final_num_layers, embedding_dim, final_dropout, feedforward_dim
        )
        _init_linear_layers(self)

    def forward(self, tokens: dict[str, Tensor]) -> Tensor:
        hidden_states: Tensor = self.final_stack(
            self.stack(torch.cat(list(tokens.values()), dim=1))
        )
        return hidden_states


class GutFormerLatent(Latent):
    """[purbancz] Mean over the sequence, then a linear map to the latent and back."""

    def __init__(self, embedding_dim: int, latent_size: int):
        super().__init__()
        self.compress = nn.Linear(embedding_dim, latent_size)
        self.expand = nn.Linear(latent_size, embedding_dim)
        _init_linear_layers(self)

    def to_latent(self, hidden_states: Tensor) -> Tensor:
        latent: Tensor = self.compress(hidden_states.mean(dim=1))
        return latent

    def from_latent(self, latent: Tensor) -> Tensor:
        decoded: Tensor = self.expand(latent)
        return decoded


class GutFormerDecoder(Decoder):
    """[purbancz] One transformer stack, shared by all modalities and applied to each separately."""

    def __init__(
        self,
        embedding_dim: int,
        num_layers: int = 1,
        num_heads: int = 2,
        dropout: float = 0.1,
        feedforward_dim: int = 2048,
    ):
        super().__init__()
        self.stack = _transformer_stack(
            embedding_dim, num_layers, num_heads, dropout, feedforward_dim
        )
        _init_linear_layers(self)

    def forward(self, queries: dict[str, Tensor]) -> dict[str, Tensor]:
        return {modality: self.stack(query) for modality, query in queries.items()}


class GutFormerHead(Head):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, 1)
        _init_linear_layers(self)

    def forward(self, hidden_states: Tensor) -> Tensor:
        reconstruction: Tensor = self.projection(hidden_states)
        return reconstruction
