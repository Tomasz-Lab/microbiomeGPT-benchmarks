# 0008. Dataset representation: MultiOmicsDataset

Status: Proposed
Date: 2026-11-16

## Context
We lack a recipe for building the desired dataset. Right now each use case has its own loader: NMF, for example, has its own specific loader for the same kind of data the ML model uses. There is no single source of truth that declares how a dataset should be built and which components it needs to load.

## Decision

1. Dataset definition and scope

We separate two concepts:

Dataset recipe (`MultiOmicsDatasetSpec`) is the declaration of a dataset. It describes the "package" of data we want:
- which modalities the dataset contains, and how to load each one
- which metadata the dataset contains, and how to load it.

The recipe is the single source of truth for how a dataset is built. It holds no data and performs no validation.

Dataset (`MultiOmicsDataset`) is the validated container that the recipe produces. Its responsibilities are:
- Holding the loaded modalities and metadata.
- Validating cross-modality invariants. These are properties of the dataset as a whole, not of any single modality:
  - sample IDs are consistent across modalities;
  - metadata covers every sample ID.
- Acting as an accessor. Code that receives a dataset can trust that it is valid and complete. It exposes operations such as selecting a modality or subsetting samples.

Out of scope:
- Validating the content of individual modalities (schema, columns, values). Each modality is responsible for that.
- Loading data (file formats, paths, database access). Sources are responsible for that.


2. Naming: MultiOmicsDataset and MultiOmicsDatasetSpec

The container is named MultiOmicsDataset and its recipe MultiOmicsDatasetSpec.

Rationale:
- "Dataset" alone is too generic to describe the domain concept: a set of omics modalities that share samples and metadata.
- "Dataset" and "DataLoader" are established PyTorch terms (torch.utils.data.Dataset, DataLoader). Reusing them would suggest our class is a PyTorch dataset with that interface (__getitem__, __len__), which it is not.
- The MultiOmics prefix keeps the domain object clearly separate from framework-level data utilities. If a PyTorch Dataset is needed later, it can be built from a MultiOmicsDataset as a separate adapter.


## Consequences

Positive

- Clear declaration. The Hydra recipe states explicitly which modalities, metadata and sources a dataset consists of. Code only needs the dataset name (e.g. load_dataset(name="cmd")) to load it, provided the recipe is correct.
- Easy to compose new datasets. A new combination of modalities or sources needs only a new or modified recipe, with no new loader code.
- Single source of truth. The same recipe and the same MultiOmicsDataset are used everywhere: training, NMF, notebooks, analysis scripts. This replaces the per-use-case loaders.
- Guaranteed validity. Every consumer receives a validated object and can use it directly as an accessor (selecting modalities, subsetting samples) without re-checking the data.

Negative

- Additional abstraction layer. Contributors must learn the recipe → load → container flow instead of reading a file directly.
- Strict validation can reject real data. Real datasets often have samples missing from one modality or from the metadata. We need an explicit policy (fail, or intersect and drop with a report); otherwise the strict invariants will block legitimate use.
- Coupling to Hydra for declaration. Recipes live in Hydra config, so using a dataset outside Hydra (e.g. in a notebook) needs either Hydra's compose API or a plain-Python way to build a MultiOmicsDatasetSpec.