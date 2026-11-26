import pytest

from data.splits import RandomSplitter

# (val_size, test_size, expected sizes of train/val/test) for 40 samples
SIZE_CASES = [
    pytest.param(0.25, 0.0, (30, 10, 0), id="val-only"),
    pytest.param(0.25, 0.25, (20, 10, 10), id="val-and-test"),
]


@pytest.mark.parametrize(("val_size", "test_size", "_"), SIZE_CASES)
def test_random_split_partitions_all_samples(metadata, val_size, test_size, _):
    split = RandomSplitter(val_size=val_size, test_size=test_size).split(metadata)

    train, val, test = set(split.train), set(split.val), set(split.test)
    # Every sample lands in exactly one part
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert train | val | test == set(metadata.index)


@pytest.mark.parametrize(("val_size", "test_size", "expected"), SIZE_CASES)
def test_random_split_sizes_match_requested_fractions(metadata, val_size, test_size, expected):
    split = RandomSplitter(val_size=val_size, test_size=test_size).split(metadata)

    assert (len(split.train), len(split.val), len(split.test)) == expected


@pytest.mark.parametrize(("val_size", "test_size", "_"), SIZE_CASES)
def test_stratified_split_keeps_class_proportions(metadata, val_size, test_size, _):
    split = RandomSplitter(val_size=val_size, test_size=test_size, stratify_by="condition").split(
        metadata
    )

    expected = metadata["condition"].value_counts(normalize=True)
    for ids in (split.train, split.val, split.test):
        if not ids:
            continue
        proportions = metadata.loc[list(ids), "condition"].value_counts(normalize=True)
        assert proportions.to_dict() == pytest.approx(expected.to_dict())
