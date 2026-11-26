"""Shared fixtures for the data domain model: a small dummy dataset, declared and loaded the real way.

Vocabulary (from ADRs 0007-0010):
- table: the expected content of one source after loading, a DataFrame indexed by `sample_id`.
- recipe: a `MultiOmicsDatasetSpec`, the declaration of a dataset: which modalities and metadata it
  has and where each one is read from. It holds no data.
- dataset: a `MultiOmicsDataset`, what `load_dataset(recipe)` returns: the loaded modalities and
  metadata, ready to use.

Flow: tables -> written to CSV files -> recipe points at the files -> load_dataset(recipe) -> dataset.
`loaded_dataset` goes through that flow; `dataset` is built directly from `tables`.
"""

import numpy as np
import pandas as pd
import pytest

from data.load import load_dataset
from data.modalities import registry
from data.modalities.base import Modality, ModalitySpec
from data.modalities.kinds import CompositionalTable
from data.modataset.base import MultiOmicsDataset, MultiOmicsDatasetSpec
from data.sources.files import CsvSource

SAMPLE_IDS = ["s1", "s2", "s3"]

# Test-only specs: tests check the mechanism, not the numbers in registry.py.
TAXONOMY = ModalitySpec(
    name="taxonomy",
    kind=CompositionalTable(
        total=1.0,
    ),
    scope="per_sample",
)
PATHWAYS = ModalitySpec(
    name="pathways",
    kind=CompositionalTable(
        total=1.0,
    ),
    scope="per_sample",
)


@pytest.fixture
def tables() -> dict[str, pd.DataFrame]:
    """Expected content of each source after loading, by name, indexed by `sample_id`.

    Modality rows sum to 1.0, so the data stays valid once validation checks are added.
    """
    index = pd.Index(SAMPLE_IDS, name="sample_id")
    return {
        "taxonomy": pd.DataFrame(
            {"species_a": [0.5, 0.2, 1.0], "species_b": [0.5, 0.8, 0.0]}, index=index
        ),
        "pathways": pd.DataFrame({"PWY-1": [0.1, 0.3, 0.6], "PWY-2": [0.9, 0.7, 0.4]}, index=index),
        "metadata": pd.DataFrame(
            {"age": [30, 45, 60], "condition": ["control", "IBD", "control"]}, index=index
        ),
    }


@pytest.fixture
def compositional_table() -> pd.DataFrame:
    """Valid compositional table: every row sums to 1.0."""
    return pd.DataFrame(
        {"species_a": [0.5, 0.2, 1.0], "species_b": [0.5, 0.8, 0.0]},
        index=pd.Index(SAMPLE_IDS, name="sample_id"),
    )


@pytest.fixture
def compositional_table_within_tolerance() -> pd.DataFrame:
    """Compositional table whose rows deviate slightly from 1.0 (1.001, 0.9999, 1.0)."""
    return pd.DataFrame(
        {"species_a": [0.5, 0.2, 1.0], "species_b": [0.501, 0.7999, 0.0]},
        index=pd.Index(SAMPLE_IDS, name="sample_id"),
    )


@pytest.fixture
def compositional_table_wrong_sum() -> pd.DataFrame:
    """Invalid compositional table: rows s2 (0.9) and s3 (1.2) do not sum to 1.0, s1 does."""
    return pd.DataFrame(
        {"species_a": [0.5, 0.2, 0.9], "species_b": [0.5, 0.7, 0.3]},
        index=pd.Index(SAMPLE_IDS, name="sample_id"),
    )


@pytest.fixture
def compositional_table_negative() -> pd.DataFrame:
    """Invalid compositional table: rows s2 and s3 contain negative values, yet every row sums to 1.0."""
    return pd.DataFrame(
        {"species_a": [0.5, -0.2, 1.1], "species_b": [0.5, 1.2, -0.1]},
        index=pd.Index(SAMPLE_IDS, name="sample_id"),
    )


@pytest.fixture
def compositional_table_mixed_types() -> pd.DataFrame:
    """Invalid compositional table: column species_b mixes strings with ints (object dtype)."""
    return pd.DataFrame(
        {"species_a": [0.5, 0.2, 1.0], "species_b": [1, "abc", 0]},
        index=pd.Index(SAMPLE_IDS, name="sample_id"),
    )


@pytest.fixture
def compositional_table_bool() -> pd.DataFrame:
    """Invalid compositional table: column species_b is bool, species_a is float."""
    return pd.DataFrame(
        {"species_a": [0.5, 0.2, 1.0], "species_b": [True, False, False]},
        index=pd.Index(SAMPLE_IDS, name="sample_id"),
    )


@pytest.fixture
def compositional_table_bad_column_names() -> pd.DataFrame:
    """Invalid compositional table: species_a follows the 'species_*' convention, the rest do not
    (another naming convention and an empty, unnamed column)."""
    return pd.DataFrame(
        [[0.5, 0.3, 0.2], [0.2, 0.4, 0.4], [1.0, 0.0, 0.0]],
        columns=["species_a", "k__Bacteria|s__Prevotella_copri", ""],
        index=pd.Index(SAMPLE_IDS, name="sample_id"),
    )


@pytest.fixture
def compositional_table_missing() -> pd.DataFrame:
    """Invalid compositional table: rows s2 and s3 contain NaN, s1 does not."""
    return pd.DataFrame(
        {"species_a": [0.5, np.nan, np.nan], "species_b": [0.5, 0.8, np.nan]},
        index=pd.Index(SAMPLE_IDS, name="sample_id"),
    )


@pytest.fixture
def compositional_table_duplicate_ids() -> pd.DataFrame:
    """Invalid compositional table: sample ids s1 and s2 occur twice, s3 once."""
    return pd.DataFrame(
        {"species_a": [0.5, 0.2, 1.0, 0.5, 0.2], "species_b": [0.5, 0.8, 0.0, 0.5, 0.8]},
        index=pd.Index(["s1", "s2", "s3", "s1", "s2"], name="sample_id"),
    )


@pytest.fixture
def recipe(tmp_path, tables, monkeypatch) -> MultiOmicsDatasetSpec:
    """Declaration of the dummy dataset, reading each table from its own CSV file.

    Files look like the real ones: the sample id is the first column, named "sample_id".
    The test-only specs are registered for the duration of the test, since a recipe
    and load_dataset look up specs by name in the registry.
    """
    for spec in (TAXONOMY, PATHWAYS):
        monkeypatch.setitem(registry.MODALITY_SPECS, spec.name, spec)

    paths = {}
    for name, table in tables.items():
        paths[name] = tmp_path / f"{name}.csv"
        table.to_csv(paths[name])

    return MultiOmicsDatasetSpec(
        name="dummy",
        modalities={
            "taxonomy": CsvSource(paths["taxonomy"]),
            "pathways": CsvSource(paths["pathways"]),
        },
        metadata=CsvSource(paths["metadata"]),
    )


@pytest.fixture
def loaded_dataset(recipe) -> MultiOmicsDataset:
    """The dummy dataset loaded from `recipe` through the full flow (sources -> load_dataset)."""
    return load_dataset(recipe)


@pytest.fixture
def modalities(tables) -> dict[str, Modality]:
    """Modalities of the dummy dataset built from `tables`: taxonomy and pathways, both s1-s3."""
    return {
        "taxonomy": Modality(spec=TAXONOMY, data=tables["taxonomy"]),
        "pathways": Modality(spec=PATHWAYS, data=tables["pathways"]),
    }


@pytest.fixture
def dataset(modalities, tables) -> MultiOmicsDataset:
    """The dummy dataset built directly from `tables`, bypassing files and loading.

    For tests of what a dataset can do (e.g. selecting), so they don't fail when loading does.
    """
    return MultiOmicsDataset(name="dummy", modalities=modalities, metadata=tables["metadata"])
