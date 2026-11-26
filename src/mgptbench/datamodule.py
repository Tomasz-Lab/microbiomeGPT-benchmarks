from dataclasses import asdict

import lightning.pytorch as L
import pandas as pd
from torch.utils.data import DataLoader

from data.modataset.base import MultiOmicsDataset
from data.splits import Splitter
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

    def setup(self, stage: str | None = None):
        if stage in ("fit", "validate", None):
            # One split of the shared index keeps taxonomy and pathways rows aligned.
            # list(): .loc reads a tuple as (rows, cols), not as a list of labels
            split_idx = self.splitter.split(self.dataset.metadata)
            self.train_dataset = BacteriaDataset(
                self.taxonomy.loc[list(split_idx.train)], self.pathways.loc[list(split_idx.train)]
            )
            self.val_dataset = BacteriaDataset(
                self.taxonomy.loc[list(split_idx.val)], self.pathways.loc[list(split_idx.val)]
            )
            if split_idx.test:
                self.test_dataset = BacteriaDataset(
                    self.taxonomy.loc[list(split_idx.test)], self.pathways.loc[list(split_idx.test)]
                )
            self.split_ids = asdict(split_idx)

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
