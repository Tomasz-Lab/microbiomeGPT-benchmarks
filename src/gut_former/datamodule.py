import lightning.pytorch as L
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.modataset.base import MultiOmicsDataset
from gut_former.dataset import BacteriaDataset
from gut_former.schema import DataSchema


class BacteriaDataModule(L.LightningDataModule):
    """Splits a loaded `MultiOmicsDataset` into taxonomy/pathways DataLoaders.

    Args:
        dataset: The loaded dataset.
        taxonomy: Name of the dataset modality used as the model's taxonomy input.
        pathways: Name of the dataset modality used as the model's pathways input.
    """

    def __init__(
        self,
        dataset: MultiOmicsDataset,
        taxonomy: str,
        pathways: str,
        taxa_multiplier: float,
        pathways_multiplier: float,
        batch_size: int = 16,
        val_size: float = 0.1,
        seed: int = 0,
    ):
        super().__init__()
        self.taxonomy: pd.DataFrame = dataset[taxonomy].data.sort_index() * taxa_multiplier
        self.pathways: pd.DataFrame = dataset[pathways].data.sort_index() * pathways_multiplier
        if not self.taxonomy.index.equals(self.pathways.index):
            raise ValueError(
                f"Modalities '{taxonomy}' and '{pathways}' of dataset '{dataset.name}' must "
                "contain the same sample IDs."
            )
        self.batch_size = batch_size
        self.val_size = val_size
        self.seed = seed

    def setup(self, stage: str | None = None):
        if stage in ("fit", "validate", None):
            # One split of the shared index keeps taxonomy and pathways rows aligned
            idx_train, idx_val = train_test_split(
                self.taxonomy.index.to_numpy(), test_size=self.val_size, random_state=self.seed
            )
            self.train_dataset = BacteriaDataset(
                self.taxonomy.loc[idx_train], self.pathways.loc[idx_train]
            )
            self.val_dataset = BacteriaDataset(
                self.taxonomy.loc[idx_val], self.pathways.loc[idx_val]
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
