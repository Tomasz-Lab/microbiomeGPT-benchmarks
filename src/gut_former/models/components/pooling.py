from abc import ABC, abstractmethod

from torch import Tensor, nn


class Pooling(nn.Module, ABC):
    """[purbancz] Reduces a sequence to one vector per sample.

    A building block for Latent implementations that pool; not a slot of its own in
    ComposableAutoEncoder, because not every Latent pools.
    """

    @abstractmethod
    def forward(self, hidden_states: Tensor) -> Tensor:
        """[purbancz] (batch_size, sequence_length, embedding_dim) to (batch_size, embedding_dim)."""


class MeanPooling(Pooling):
    """[purbancz] Mean over the sequence."""

    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states.mean(dim=1)
