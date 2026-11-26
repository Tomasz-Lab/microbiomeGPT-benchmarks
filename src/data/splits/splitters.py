import pandas as pd

from data.splits.base import Split, Splitter


class RandomSplitter(Splitter):
    def __init__(
        self,
        val_size: float = 0.1,
        test_size: float = 0.0,
        seed: int = 0,
        stratify_by: str | None = None,
    ):
        pass

    def split(self, metadata: pd.DataFrame) -> Split:
        pass


class CohortSplitter(Splitter):
    def __init__(
        self,
        group_by: str,
        val_size: float = 0.1,
        test_size: float = 0.0,
        seed: int = 0,
    ):
        pass

    def split(self, metadata: pd.DataFrame) -> Split:
        pass
