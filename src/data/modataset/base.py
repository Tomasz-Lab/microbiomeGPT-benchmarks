from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import pandas as pd

from data.modalities.base import Modality
from data.modalities.registry import MODALITIES, ModalityName
from data.sources.base import Source


@dataclass(frozen=True)
class MultiOmicsDatasetSpec:
    """Recipe for a `MultiOmicsDataset`: which modalities and metadata it consists of and how to load them.

    Holds no data and performs no data validation.

    Attributes:
        name: Dataset name, e.g. "cMD".
        modalities: Source of each modality, keyed by its name in the modality registry (`MODALITIES`),
            e.g. "taxonomy_metaphlan" -> CsvSource(...).
        metadata: Source of the per-sample metadata.
    """

    name: str
    modalities: Mapping[ModalityName, Source]
    metadata: Source

    def __post_init__(self) -> None:
        # Check names here, so an unknown name fails when the recipe is built, before any file is read.
        for name in self.modalities:
            MODALITIES.get(name)


@dataclass(frozen=True)
class MultiOmicsDataset:
    """Multi-omics dataset defined by its modalities and metadata.

    Attributes:
        name: Dataset name, e.g. "cMD" (taken from the spec it was built from).
        modalities: Modalities by name, e.g. "taxonomy_species" -> Modality(spec=TAXONOMY, data=DataFrame).
        metadata: Per-sample metadata indexed by `sample_id`.
        sources: Source each modality was read from, keyed by modality name, plus "metadata";
            empty for a dataset built in memory.
    """

    name: str
    modalities: Mapping[ModalityName, Modality]
    metadata: pd.DataFrame
    sources: Mapping[str, Source] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Modality:
        return self.modalities[name]

    def select_samples(self, ids: Iterable[str]) -> "MultiOmicsDataset":
        ids = list(ids)  # used once per modality and for metadata, so an iterator would run out
        modalities = {
            name: modality.select_samples(ids) for name, modality in self.modalities.items()
        }
        metadata = self.metadata.loc[self.metadata.index.intersection(ids)]
        return MultiOmicsDataset(
            name=self.name, modalities=modalities, metadata=metadata, sources=self.sources
        )

    def select_modalities(self, names: Iterable[str]) -> "MultiOmicsDataset":
        modalities = {name: self.modalities[name] for name in names}
        sources = {
            name: info
            for name, info in self.sources.items()
            if name in modalities or name == "metadata"
        }
        return MultiOmicsDataset(
            name=self.name, modalities=modalities, metadata=self.metadata, sources=sources
        )
