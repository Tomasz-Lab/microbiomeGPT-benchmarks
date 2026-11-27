from abc import ABC, abstractmethod

from torch import Tensor, nn

from mgptbench.models.components._initialization import xavier_init_


class Head(nn.Module, ABC):
    """[purbancz] Projects hidden states to one reconstructed value per feature."""

    @abstractmethod
    def forward(self, hidden_states: Tensor) -> Tensor:
        """[purbancz] (batch_size, num_features, embedding_dim) to (batch_size, num_features, 1)."""


class LinearHead(Head):
    """[purbancz] One linear projection to a single value, shared by all features."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projection = nn.Linear(embedding_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        xavier_init_(self.projection)

    def forward(self, hidden_states: Tensor) -> Tensor:
        reconstruction: Tensor = self.projection(hidden_states)
        return reconstruction
