from abc import ABC, abstractmethod

from torch import Tensor, nn

from gut_former.models.components._initialization import xavier_init_
from gut_former.models.components.pooling import Pooling


class Latent(nn.Module, ABC):
    """[purbancz] Maps the encoded sequence to one latent vector per sample and back."""

    @abstractmethod
    def to_latent(self, hidden_states: Tensor) -> Tensor:
        """[purbancz] (batch_size, sequence_length, embedding_dim) to (batch_size, latent_size)."""

    @abstractmethod
    def from_latent(self, latent: Tensor) -> Tensor:
        """[purbancz] (batch_size, latent_size) to (batch_size, embedding_dim)."""


class PooledLinearLatent(Latent):
    """[purbancz] Pools the sequence, then a linear map to the latent and back."""

    def __init__(self, embedding_dim: int, latent_size: int, pooling: Pooling):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.latent_size = latent_size
        self.pooling = pooling
        self.compress = nn.Linear(embedding_dim, latent_size)
        self.expand = nn.Linear(latent_size, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # [purbancz] Not the pooling: it initializes its own weights, if any
        xavier_init_(self.compress, self.expand)

    def to_latent(self, hidden_states: Tensor) -> Tensor:
        latent: Tensor = self.compress(self.pooling(hidden_states))
        return latent

    def from_latent(self, latent: Tensor) -> Tensor:
        decoded: Tensor = self.expand(latent)
        return decoded
