from pandas.testing import assert_frame_equal

from data.load import load_dataset
from data.modalities.registry import MODALITY_SPECS
from data.modataset.base import MultiOmicsDataset


def test_load_dataset_builds_dataset_from_recipe(recipe, tables):
    dataset = load_dataset(recipe)

    assert isinstance(dataset, MultiOmicsDataset)
    assert dataset.name == "dummy"
    assert set(dataset.modalities) == {"taxonomy", "pathways"}

    for name in recipe.modalities:
        modality = dataset[name]
        # Data stays bound to its spec from the registry
        assert modality.spec is MODALITY_SPECS[name]
        # Sample id moved from the first column to the index; values are unchanged
        assert_frame_equal(modality.data, tables[name])

    assert_frame_equal(dataset.metadata, tables["metadata"])


def test_select_samples_cuts_all_modalities_and_metadata_together(dataset, tables):
    ids = ["s1", "s3"]

    selected = dataset.select_samples(ids)

    assert isinstance(selected, MultiOmicsDataset)
    assert selected.name == dataset.name
    assert set(selected.modalities) == set(dataset.modalities)
    for name, modality in selected.modalities.items():
        # Still bound to its definition, holding only the selected rows
        assert modality.spec is dataset[name].spec
        assert_frame_equal(modality.data, tables[name].loc[ids])
    # Metadata stays aligned with the modalities
    assert_frame_equal(selected.metadata, tables["metadata"].loc[ids])

    # The original dataset is unchanged
    for name, modality in dataset.modalities.items():
        assert_frame_equal(modality.data, tables[name])
    assert_frame_equal(dataset.metadata, tables["metadata"])


def test_select_modalities_keeps_only_named_modalities(dataset, tables):
    selected = dataset.select_modalities(["pathways"])

    assert isinstance(selected, MultiOmicsDataset)
    assert selected.name == dataset.name
    assert set(selected.modalities) == {"pathways"}
    assert selected["pathways"] is dataset["pathways"]
    # Samples and metadata are untouched
    assert_frame_equal(selected.metadata, tables["metadata"])

    # The original dataset still has every modality
    assert set(dataset.modalities) == {"taxonomy", "pathways"}
