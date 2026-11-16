from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

from data.modalities.base import Modality


@dataclass(frozen=True)
class MultiOmicsDataset:
    """Multi-omics dataset defined by its modalities and metadata.

    Attributes:
        modalities: Modalities by name, e.g. "taxonomy_species" -> Modality(spec=TAXONOMY, data=DataFrame).
        metadata: Per-sample metadata indexed by `sample_id`.
    """

    modalities: Mapping[str, Modality]
    metadata: pd.DataFrame

    def __post_init__(self) -> None:
        pass

    def __getitem__(self, name: str) -> Modality:
        pass

    def select_samples(self, ids: Iterable[str]) -> "MultiOmicsDataset":
        pass

    def select_modalities(self, names: Iterable[str]) -> "MultiOmicsDataset":
        pass
