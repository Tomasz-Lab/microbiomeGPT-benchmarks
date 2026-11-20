import pandas as pd
import pytest

from data.modalities.base import Modality, ModalitySpec
from data.modalities.kinds import CompositionalTable
from data.modataset.base import MultiOmicsDataset, SampleAlignmentError

KIND = CompositionalTable(total=1.0, tolerance=1e-6)
TAXONOMY = ModalitySpec(name="taxonomy", kind=KIND, scope="per_sample")
PATHWAYS = ModalitySpec(name="pathways", kind=KIND, scope="per_sample")
REFERENCE = ModalitySpec(name="reference", kind=KIND, scope="reference")


def build_dataset(
    tables: dict[str, pd.DataFrame], specs: tuple[ModalitySpec, ...] = (TAXONOMY, PATHWAYS)
) -> MultiOmicsDataset:
    modalities = {}
    for spec in specs:
        modalities[spec.name] = Modality(spec=spec, data=tables[spec.name])
    return MultiOmicsDataset(name="dummy", modalities=modalities, metadata=tables["metadata"])


def test_dataset_accepts_aligned_modalities_and_matching_metadata(tables):
    build_dataset(tables)


def test_dataset_rejects_modality_with_subset_of_samples(tables):
    tables["pathways"] = tables["pathways"].drop(index="s3")

    with pytest.raises(SampleAlignmentError, match="pathways") as exc_info:
        build_dataset(tables)

    assert exc_info.value.missing == {"pathways": {"s3"}}


def test_dataset_rejects_modalities_with_same_size_but_different_samples(tables):
    tables["pathways"] = tables["pathways"].rename(index={"s3": "s4"})

    with pytest.raises(SampleAlignmentError) as exc_info:
        build_dataset(tables)

    assert exc_info.value.missing == {"taxonomy": {"s4"}, "pathways": {"s3"}}


def test_dataset_ignores_samples_of_reference_modality(tables):
    tables["reference"] = tables["taxonomy"].rename(index={"s1": "ref1"})

    build_dataset(tables, specs=(TAXONOMY, PATHWAYS, REFERENCE))
