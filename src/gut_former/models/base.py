"""[purbancz] Model-level interface - ADR 0006."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from torch import Tensor, nn


@dataclass
class ModelOutput:
    """[purbancz]
    reconstructions: modality -> (batch_size, num_features, 1)
    latent: (batch_size, latent_size)
    aux: extra tensors for the loss
    """

    reconstructions: dict[str, Tensor]
    latent: Tensor
    aux: dict[str, Tensor] = field(default_factory=dict)


class MicrobiomeModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, inputs: dict[str, Tensor]) -> ModelOutput:
        """[purbancz] inputs: modality -> (batch_size, num_features, 1)."""
