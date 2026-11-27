from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from mgptbench.models.components._transformer_stack import TransformerStack


class Decoder(nn.Module, ABC):
    """[purbancz] Turns the per-feature queries of each modality into hidden states.

    Modalities may be decoded separately or attend to each other; that is up to
    the implementation.
    """

    @abstractmethod
    def forward(self, queries: dict[str, Tensor]) -> dict[str, Tensor]:
        """[purbancz] modality name -> (batch_size, num_features, embedding_dim),
        same keys and shapes out."""


class SharedTransformerDecoder(Decoder):
    """[purbancz] One transformer stack, shared by all modalities and applied to each separately."""

    def __init__(self, embedding_dim: int, stack: Mapping[str, Any]):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.stack = TransformerStack(embedding_dim, **stack)

    def forward(self, queries: dict[str, Tensor]) -> dict[str, Tensor]:
        return {modality: self.stack(query) for modality, query in queries.items()}
