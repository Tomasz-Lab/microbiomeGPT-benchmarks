from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from mgptbench.models.components._transformer_stack import TransformerStack


class Encoder(nn.Module, ABC):
    """[purbancz] Fuses the tokens of all modalities into one sequence.

    How they are fused (jointly from the start, or encoded separately and mixed later)
    is up to the implementation.
    """

    @abstractmethod
    def forward(self, tokens: dict[str, Tensor]) -> Tensor:
        """[purbancz] modality name -> (batch_size, num_features, embedding_dim) becomes
        (batch_size, sequence_length, embedding_dim); sequence_length is set by the encoder."""


class JointTransformerEncoder(Encoder):
    """[purbancz] Concatenates all modalities into one sequence, then runs the stacks in order."""

    def __init__(self, embedding_dim: int, stacks: Sequence[Mapping[str, Any]]):
        super().__init__()
        if not stacks:
            raise ValueError("JointTransformerEncoder needs at least one stack.")
        self.embedding_dim = embedding_dim
        self.stacks = nn.Sequential(*(TransformerStack(embedding_dim, **s) for s in stacks))

    def forward(self, tokens: dict[str, Tensor]) -> Tensor:
        hidden_states: Tensor = self.stacks(torch.cat(list(tokens.values()), dim=1))
        return hidden_states
