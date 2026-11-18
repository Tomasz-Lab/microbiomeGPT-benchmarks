from pandas.testing import assert_frame_equal

from data.load import load_dataset
from data.modataset.base import MultiOmicsDataset


def test_load_dataset_builds_dataset_from_recipe(recipe, tables):
    dataset = load_dataset(recipe)

    assert isinstance(dataset, MultiOmicsDataset)
    assert dataset.name == "dummy"
    assert set(dataset.modalities) == {"taxonomy", "pathways"}

    for name, entry in recipe.modalities.items():
        modality = dataset[name]
        # Data stays bound to the definition it was declared with
        assert modality.spec is entry.spec
        # Sample id moved from the first column to the index; values are unchanged
        assert_frame_equal(modality.data, tables[name])

    assert_frame_equal(dataset.metadata, tables["metadata"])
