import pandas as pd

from data.splits.base import Split, Splitter


class RandomSplitter(Splitter):
    """Baseline: samples assigned at random, optionally stratified by a label column."""

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


class HoldoutCohortSplitter(Splitter):
    """Named cohorts held out as test; remaining cohorts split into train/val.

    Measures generalisation to unseen studies.
    """

    def __init__(
        self,
        group_by: str,
        test_groups: list[str],
        val_size: float = 0.1,
        seed: int = 0,
    ):
        pass

    def split(self, metadata: pd.DataFrame) -> Split:
        pass
