# 0010. Load dataset

Status: Proposed
Date: 2026-11-16

## Context

Previous ADRs defined modality kinds, modalities, the multi-omics dataset and sources. Each describes one part of a dataset, but nothing connects them: no single component builds a dataset from its recipe. We need one orchestrator that runs these steps in order and returns a `MultiOmicsDataset`.

## Decision

1. `load_dataset(spec: MultiOmicsDatasetSpec) -> MultiOmicsDataset`. The input is an instantiated recipe, not a raw Hydra config. 
2. **Responsibilities.** `load_dataset` only orchestrates. For a given recipe it:
   - reads each modality's raw data from its source;
   - looks up the modality's spec in the registry by name and passes the raw data to it, which validates it, and binds the result into a `Modality`;
   - reads the metadata from its source in the same way;
   - assembles the `MultiOmicsDataset`, whose constructor validates the cross-modality invariants;
   - returns the dataset.
3. Every check belongs to the component that owns it: sources (reading), modalities (per-modality schema), the dataset (cross-modality invariants). `load_dataset` does not catch or wrap their errors; they propagate to the caller unchanged.
4. Knowledge of file formats or backends, modality schemas, the policy for samples that don't match across modalities, data transformations, and caching.

## Consequences

### Positive
- Training, NMF and notebooks all get a dataset through the same function, which replaces the per-use-case loaders.
- New formats, backends or modality kinds don't change `load_dataset`, because all specific knowledge lives in sources and modalities.
- Because errors propagate unchanged, a failure points directly to the component that rejected the data.
- Taking a recipe object rather than a config means it can be called from plain Python, including tests and notebooks.

### Negative

- Everything is read and validated before anything is returned. One bad modality fails the whole load, and large datasets are read in full.
