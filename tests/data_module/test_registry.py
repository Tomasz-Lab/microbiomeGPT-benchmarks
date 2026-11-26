"""A dataset recipe accepts only modalities registered in `MODALITIES`."""

import pytest

from data.modataset.base import MultiOmicsDatasetSpec
from data.sources.files import CsvSource


def test_recipe_accepts_registered_modality():
    recipe = MultiOmicsDatasetSpec(
        name="dummy",
        modalities={"taxonomy_metaphlan": CsvSource("taxonomy.csv")},
        metadata=CsvSource("metadata.csv"),
    )

    assert set(recipe.modalities) == {"taxonomy_metaphlan"}


def test_recipe_rejects_unregistered_modality():
    with pytest.raises(KeyError, match="Unknown modality 'not_registered'"):
        MultiOmicsDatasetSpec(
            name="dummy",
            modalities={"not_registered": CsvSource("x.csv")},
            metadata=CsvSource("metadata.csv"),
        )
