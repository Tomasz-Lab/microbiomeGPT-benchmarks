from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

from data.modalities.base import Modality, ModalitySpec
from data.sources.base import Source


@dataclass(frozen=True)
class ModalityEntry:
    """One declared modality: what it is (spec) paired with where it comes from (source)."""

    spec: ModalitySpec
    source: Source


@dataclass(frozen=True)
class MetadataEntry:
    """Declared metadata: where it comes from. Schema to be defined later."""

    source: Source


@dataclass(frozen=True)
class MultiOmicsDatasetSpec:
    """Recipe for a `MultiOmicsDataset`: which modalities and metadata it consists of and how to load them.

    Holds no data and performs no data validation.

    Attributes:
        name: Dataset name, e.g. "cMD".
        modalities: Declared modalities by name, e.g. "taxonomy_metaphlan" -> ModalityEntry(spec, source).
        metadata: Declared metadata.
    """

    name: str
    modalities: Mapping[str, ModalityEntry]
    metadata: MetadataEntry

    def __post_init__(self) -> None:
        # Planned: each key equals its entry's `spec.name`, so the name cannot drift between the two.
        pass


@dataclass(frozen=True)
class MultiOmicsDataset:
    """Multi-omics dataset defined by its modalities and metadata.

    Attributes:
        name: Dataset name, e.g. "cMD" (taken from the spec it was built from).
        modalities: Modalities by name, e.g. "taxonomy_species" -> Modality(spec=TAXONOMY, data=DataFrame).
        metadata: Per-sample metadata indexed by `sample_id`.
    """

    name: str
    modalities: Mapping[str, Modality]
    metadata: pd.DataFrame

    def __getitem__(self, name: str) -> Modality:
        return self.modalities[name]

    def select_samples(self, ids: Iterable[str]) -> "MultiOmicsDataset":
        ids = list(ids)  # used once per modality and for metadata, so an iterator would run out
        modalities = {
            name: modality.select_samples(ids) for name, modality in self.modalities.items()
        }
        metadata = self.metadata.loc[self.metadata.index.intersection(ids)]
        return MultiOmicsDataset(name=self.name, modalities=modalities, metadata=metadata)

    def select_modalities(self, names: Iterable[str]) -> "MultiOmicsDataset":
        modalities = {name: self.modalities[name] for name in names}
        return MultiOmicsDataset(name=self.name, modalities=modalities, metadata=self.metadata)
