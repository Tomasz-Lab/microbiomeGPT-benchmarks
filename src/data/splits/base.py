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

    def __post_init__(self) -> None:
        # Guards every splitter and real data, e.g. a duplicated sample id in metadata
        # landing twice in one part or in two parts at once.
        parts = {"train": self.train, "val": self.val, "test": self.test}
        for name, ids in parts.items():
            if len(set(ids)) != len(ids):
                raise ValueError(f"Split part {name!r} contains duplicated sample ids")

        train, val, test = set(self.train), set(self.val), set(self.test)
        if train & val or train & test or val & test:
            raise ValueError("Split parts overlap: a sample id occurs in more than one part")


class Splitter(ABC):
    @abstractmethod
    def split(self, metadata: pd.DataFrame) -> Split:
        pass
