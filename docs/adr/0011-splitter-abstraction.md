# 0011. Splitter abstraction

Status: Proposed
Date: 2026-11-26

## Context

The train/val split was hard-coded in `BacteriaDataModule.setup()` as a single random `train_test_split`. Adding any other kind of split (stratified, held-out cohort, with a test set) meant editing the DataModule. The split was also recorded only as counts and hashes, so the exact assignment of samples could not be recovered later.

## Decision

1. **`Splitter` interface.** An abstract `Splitter` with one method `split(metadata: pd.DataFrame) -> Split`. Each split strategy is a concrete class: `RandomSplitter` (random, optionally stratified) and `HoldoutCohortSplitter` (named cohorts held out as test).
2. **`Split` returns sample ids, not data.** `Split` is a frozen dataclass with `train`, `val` and `test` tuples of sample ids. The dataset has several modalities, and the sample id is the one key they all share, so one `Split` selects the same samples from every modality. Ids are also cheap to store and compare.
3. **The DataModule receives a splitter.** `BacteriaDataModule` takes a `Splitter` in its constructor. It doesn't know which split strategy is used. The splitter is chosen in `conf/datamodule/*.yaml` via `_target_`.
4. **The split is saved with the run.** `data_state` logs the sample count and an id hash for each part, and writes the full ids to `split.json` in the run's output directory, so the split can be reproduced and compared between runs.

## Consequences

### Positive
- A new kind of split is a new `Splitter` class and a config change; the DataModule stays untouched.
- One `Split` keeps all modalities aligned, and splitters can be tested on a metadata frame alone, without data or a DataModule.
- `split.json` and the hashes show exactly which samples were in each part of a run.

### Negative
- Splitters see only `metadata`, so a split that needs modality data (e.g. on feature values) doesn't fit the interface.