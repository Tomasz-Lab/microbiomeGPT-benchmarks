from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass

import pandas as pd

SampleId = Hashable


@dataclass(frozen=True)
class Split:
    """
    Attributes:
        train: Sample ids used for training.
        val: Sample ids used for validation.
        test: Sample ids held out for final evaluation; empty when not used.
    """

    train: tuple[SampleId, ...]
    val: tuple[SampleId, ...]
    test: tuple[SampleId, ...] = ()


class Splitter(ABC):
    @abstractmethod
    def split(self, samples: pd.DataFrame) -> Split:
        pass
