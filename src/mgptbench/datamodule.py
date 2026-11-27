from functools import cached_property

import lightning.pytorch as L
import pandas as pd
from torch.utils.data import DataLoader

from data.modataset.base import MultiOmicsDataset
from data.splits import Split, Splitter
from mgptbench.dataset import BacteriaDataset
from mgptbench.schema import DataSchema


class BacteriaDataModule(L.LightningDataModule):
    """Splits a loaded `MultiOmicsDataset` into taxonomy/pathways DataLoaders.

    Args:
        dataset: The loaded dataset.
        splitter: Assigns samples to train/val/test, from `dataset.metadata`.
        taxonomy: Name of the dataset modality used as the model's taxonomy input.
        pathways: Name of the dataset modality used as the model's pathways input.
    """

    def __init__(
        self,
        dataset: MultiOmicsDataset,
        splitter: Splitter,
        taxonomy: str,
        pathways: str,
        batch_size: int = 16,
    ):
        super().__init__()
        self.dataset = dataset
        self.taxonomy: pd.DataFrame = dataset[taxonomy].data.sort_index()
        self.pathways: pd.DataFrame = dataset[pathways].data.sort_index()

        self.batch_size = batch_size
        self.splitter = splitter

    @cached_property
    def split(self) -> Split:
        return self.splitter.split(self.dataset.metadata)

    def setup(self, stage: str | None = None):
        if stage in ("fit", "validate", None):
            # list(...) for now: .loc treats a tuple as (row, col). This goes away when the
            # DataModule selects via dataset.select_samples(split.<part>) instead of pandas ~Fasamik39
            self.train_dataset = BacteriaDataset(
                self.taxonomy.loc[list(self.split.train)], self.pathways.loc[list(self.split.train)]
            )
            self.val_dataset = BacteriaDataset(
                self.taxonomy.loc[list(self.split.val)], self.pathways.loc[list(self.split.val)]
            )
            if self.split.test:
                # Currently we are doing nothing with the test set, but we want to keep it around for future ~Fasamik39
                self.test_dataset = BacteriaDataset(
                    self.taxonomy.loc[list(self.split.test)],
                    self.pathways.loc[list(self.split.test)],
                )

        if stage in ("predict", None):
            self.predict_dataset = BacteriaDataset(self.taxonomy, self.pathways)

    @property
    def predict_index(self) -> pd.Index:
        """Sample IDs in the order of predict_dataloader."""
        return self.taxonomy.index

    def data_schema(self) -> DataSchema:
        return DataSchema(
            {
                "taxonomy": tuple(self.taxonomy.columns),
                "pathways": tuple(self.pathways.columns),
            }
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False)
