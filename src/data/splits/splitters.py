import pandas as pd
from sklearn.model_selection import train_test_split

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
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed
        self.stratify_by = stratify_by

    def split(self, metadata: pd.DataFrame) -> Split:
        holdout_size = self.val_size + self.test_size
        train, holdout = train_test_split(
            metadata,
            test_size=holdout_size,
            random_state=self.seed,
            stratify=metadata[self.stratify_by] if self.stratify_by else None,
        )
        if self.test_size == 0:
            return Split(train=tuple(train.index), val=tuple(holdout.index))

        val, test = train_test_split(
            holdout,
            test_size=self.test_size / holdout_size,
            random_state=self.seed,
            stratify=holdout[self.stratify_by] if self.stratify_by else None,
        )
        return Split(train=tuple(train.index), val=tuple(val.index), test=tuple(test.index))


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
