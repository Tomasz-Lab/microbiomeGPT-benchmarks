"""[purbancz] Model-level interface - ADR 0006."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class ModelOutput:
    """[purbancz] What every model returns; learning methods extend it with what their loss needs.

    latent: (batch_size, latent_size), the deterministic sample representation used downstream
    (e.g. the mean for a VAE, the pooled embedding for a masked encoder)
    """

    latent: Tensor


class MicrobiomeModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, inputs: dict[str, Tensor]) -> ModelOutput:
        """[purbancz] inputs: modality -> (batch_size, num_features, 1)."""
