"""[purbancz] Autoencoder interface: the models AutoEncoderSystem trains - ADR 0006."""

from abc import abstractmethod
from dataclasses import dataclass

from torch import Tensor

from gut_former.models.base import MicrobiomeModel, ModelOutput


@dataclass
class AutoEncoderOutput(ModelOutput):
    """[purbancz] reconstructions: modality -> (batch_size, num_features, 1)"""

    reconstructions: dict[str, Tensor]


class AutoEncoderModel(MicrobiomeModel):
    @abstractmethod
    def forward(self, inputs: dict[str, Tensor]) -> AutoEncoderOutput:
        """[purbancz] inputs: modality -> (batch_size, num_features, 1)."""
