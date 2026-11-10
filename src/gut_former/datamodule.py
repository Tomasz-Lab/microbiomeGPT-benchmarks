from pathlib import Path

import lightning.pytorch as L
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from gut_former.dataset import BacteriaDataset
from gut_former.schema import DataSchema
from settings.paths import DATASET_DIR


def frame_paths(data_dir: str | Path, name: str) -> tuple[Path, Path]:
    """Taxonomy and pathways CSV files of a dataset."""
    return Path(data_dir) / f"taxonomy_{name}.csv", Path(data_dir) / f"pathways_{name}.csv"


def load_frames(
    data_dir: str | Path, name: str, taxa_multiplier: float, pathways_multiplier: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load taxonomy and pathways tables with the shared preprocessing."""
    taxonomy_path, pathways_path = frame_paths(data_dir, name)
    taxonomy = (
        pd.read_csv(taxonomy_path, index_col=[0], low_memory=False).fillna(0).sort_index()
        * taxa_multiplier
    )
    pathways = (
        pd.read_csv(pathways_path, index_col=[0], low_memory=False).fillna(0).sort_index()
        * pathways_multiplier
    )
    if not taxonomy.index.equals(pathways.index):
        raise ValueError(
            f"Taxonomy and pathways samples differ for dataset '{name}'; "
            "both tables must contain the same sample IDs."
        )
    return taxonomy, pathways


class BacteriaDataModule(L.LightningDataModule):
    def __init__(
        self,
        name: str,
        taxa_multiplier: float,
        pathways_multiplier: float,
        batch_size: int = 16,
        data_dir: str | Path | None = None,
        val_size: float = 0.1,
        seed: int = 0,
    ):
        super().__init__()
        # Not `name`: LightningDataModule already defines that attribute
        self.dataset_name = name
        self.taxa_multiplier = taxa_multiplier
        self.pathways_multiplier = pathways_multiplier
        self.batch_size = batch_size
        self.data_dir = data_dir if data_dir is not None else DATASET_DIR
        self.val_size = val_size
        self.seed = seed

        self._frames: tuple[pd.DataFrame, pd.DataFrame] | None = None
        # D: 0003
        self.input_files: list[Path] = []
        self.split_ids: dict[str, list] = {}

    @property
    def taxonomy(self) -> pd.DataFrame:
        return self._loaded_frames()[0]

    @property
    def pathways(self) -> pd.DataFrame:
        return self._loaded_frames()[1]

    def _loaded_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._frames is None:
            raise RuntimeError("Call setup() before accessing the data.")
        return self._frames

    def setup(self, stage: str | None = None):
        if self._frames is None:
            self._frames = load_frames(
                self.data_dir, self.dataset_name, self.taxa_multiplier, self.pathways_multiplier
            )
            self.input_files = list(frame_paths(self.data_dir, self.dataset_name))

        if stage in ("fit", "validate", None):
            # One split of the shared index keeps taxonomy and pathways rows aligned
            idx_train, idx_val = train_test_split(
                self.taxonomy.index, test_size=self.val_size, random_state=self.seed
            )
            self.train_dataset = BacteriaDataset(
                self.taxonomy.loc[idx_train], self.pathways.loc[idx_train]
            )
            self.val_dataset = BacteriaDataset(
                self.taxonomy.loc[idx_val], self.pathways.loc[idx_val]
            )
            self.split_ids = {"train": list(idx_train), "val": list(idx_val)}

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
