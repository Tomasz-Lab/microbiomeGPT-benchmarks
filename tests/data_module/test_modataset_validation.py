from dataclasses import replace

import pytest

from data.modalities.base import Modality
from data.modataset.base import MetadataCoverageError, MultiOmicsDataset, SampleAlignmentError


def test_dataset_accepts_aligned_modalities_and_matching_metadata(modalities, tables):
    MultiOmicsDataset(name="dummy", modalities=modalities, metadata=tables["metadata"])


def test_dataset_rejects_modality_with_subset_of_samples(modalities, tables):
    modalities["pathways"] = modalities["pathways"].select_samples(["s1", "s2"])

    with pytest.raises(SampleAlignmentError, match="pathways") as exc_info:
        MultiOmicsDataset(name="dummy", modalities=modalities, metadata=tables["metadata"])

    assert exc_info.value.missing == {"pathways": {"s3"}}


def test_dataset_rejects_modalities_with_same_size_but_different_samples(modalities, tables):
    pathways = modalities["pathways"]
    modalities["pathways"] = Modality(
        spec=pathways.spec, data=pathways.data.rename(index={"s3": "s4"})
    )

    with pytest.raises(SampleAlignmentError) as exc_info:
        MultiOmicsDataset(name="dummy", modalities=modalities, metadata=tables["metadata"])

    assert exc_info.value.missing == {"taxonomy": {"s4"}, "pathways": {"s3"}}


def test_dataset_ignores_samples_of_reference_modality(modalities, tables):
    taxonomy = modalities["taxonomy"]
    reference_spec = replace(taxonomy.spec, name="reference", scope="reference")
    modalities["reference"] = Modality(
        spec=reference_spec, data=taxonomy.data.rename(index={"s1": "ref1"})
    )

    MultiOmicsDataset(name="dummy", modalities=modalities, metadata=tables["metadata"])


def test_dataset_accepts_metadata_with_extra_samples(modalities, tables):
    metadata = tables["metadata"]
    metadata.loc["s4"] = [50, "control"]

    MultiOmicsDataset(name="dummy", modalities=modalities, metadata=metadata)


def test_dataset_rejects_metadata_missing_samples(modalities, tables):
    metadata = tables["metadata"].drop(index=["s1", "s2"])

    with pytest.raises(MetadataCoverageError) as exc_info:
        MultiOmicsDataset(name="dummy", modalities=modalities, metadata=metadata)

    assert exc_info.value.missing == {"s1", "s2"}


def test_dataset_checks_alignment_before_metadata(modalities, tables):
    modalities["pathways"] = modalities["pathways"].select_samples(["s1", "s2"])
    metadata = tables["metadata"].drop(index="s1")

    with pytest.raises(SampleAlignmentError) as exc_info:
        MultiOmicsDataset(name="dummy", modalities=modalities, metadata=metadata)

    assert not isinstance(exc_info.value, MetadataCoverageError)
