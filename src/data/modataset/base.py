from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import pandas as pd

from data.modalities.base import Modality
from data.modalities.registry import MODALITIES, ModalityName
from data.sources.base import Source


def _format_ids(ids: Iterable[str]) -> str:
    # key=str: sample IDs read from CSV can be ints, which also must not break the message.
    ordered = sorted(ids, key=str)
    preview = ", ".join(str(i) for i in ordered[:5])
    if len(ordered) > 5:
        preview += f" (+{len(ordered) - 5} more)"
    return preview


class DatasetValidationError(ValueError):
    """Base for MultiOmicsDataset cross-modality validation errors."""


class SampleAlignmentError(DatasetValidationError):
    """Per-sample modalities do not share the same sample IDs.

    Attributes:
        missing: Modality name -> IDs present in another modality but not in this one.
    """

    def __init__(self, missing: dict[str, frozenset[str]], n_samples: int) -> None:
        self.missing = missing
        lines = ["Modalities are not aligned on sample IDs:"]
        for name in sorted(missing):
            ids = missing[name]
            lines.append(f"  {name}: missing {len(ids)} of {n_samples} ({_format_ids(ids)})")
        super().__init__("\n".join(lines))


class MetadataCoverageError(DatasetValidationError):
    """Some sample IDs present in the modalities are missing from metadata.

    Attributes:
        missing: IDs present in the modalities but not in metadata.
    """

    def __init__(self, missing: frozenset[str]) -> None:
        self.missing = missing
        super().__init__(
            f"Metadata is missing {len(missing)} sample(s) present in modalities: "
            f"{_format_ids(missing)}"
        )


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

    def __post_init__(self) -> None:
        ids_by_modality = self._per_sample_ids()
        self._check_samples_aligned(ids_by_modality)
        self._check_metadata_covers_samples(ids_by_modality)

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

    def _per_sample_ids(self) -> dict[ModalityName, frozenset[str]]:
        ids_by_modality = {}
        for name, modality in self.modalities.items():
            if modality.spec.scope == "per_sample":
                ids_by_modality[name] = modality.sample_ids()
        return ids_by_modality

    def _check_samples_aligned(self, ids_by_modality: dict[ModalityName, frozenset[str]]) -> None:
        all_ids = set()
        for ids in ids_by_modality.values():
            all_ids |= ids

        missing = {}
        for name, ids in ids_by_modality.items():
            if ids != all_ids:
                missing[name] = frozenset(all_ids - ids)

        if missing:
            raise SampleAlignmentError(missing, n_samples=len(all_ids))

    def _check_metadata_covers_samples(
        self, ids_by_modality: dict[ModalityName, frozenset[str]]
    ) -> None:
        metadata_ids = set(self.metadata.index)
        missing = set()
        for ids in ids_by_modality.values():
            missing |= ids - metadata_ids

        if missing:
            raise MetadataCoverageError(frozenset(missing))
