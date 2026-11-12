# 0007. Modality abstraction

Status: Accepted
Date: 2026-11-12

## Context
The code has no well-defined abstraction for our modalities. Nowhere in the code is it defined exactly what taxonomy or a compositional table means. To improve the quality of our data abstraction and make it transferable to other use cases, we need to define these abstractions.

## Decision
> We follow Domain-Driven Design, which models software around explicitly named core concepts of the problem domain and the rules they must satisfy ([Fowler, "Domain Driven Design"](https://martinfowler.com/bliki/DomainDrivenDesign.html)).

1. `ModalityKind` describes the structure of a modality's data. For example, taxonomy is a `CompositionalTable`: each row is a sample, and its values sum to the same total (e.g. 1.0 or 100). A phylogenetic tree is a different kind, a *graph*. With an explicit kind, other components (loaders, models) know what structure they receive.

2. For now we implement one kind, `CompositionalTable`, used by both taxonomy and pathways. We also add a placeholder `Graph` kind, which is not used yet, to show that the design supports more kinds.

3. `ModalityKind` is an abstract interface. Each kind declares its own parameters (e.g. `total`, `tolerance` for `CompositionalTable`) and implements `validate()`, which checks data against them.

4. `ModalitySpec` describes one concrete, named modality. It holds only the fields shared by all kinds:
   - `name`: stable key used across the system (e.g. `"taxonomy"`),
   - `kind`: a `ModalityKind` instance with concrete parameters,
   - `scope`: `per_sample` (one row per sample) or `reference` (shared across samples, e.g. a phylogenetic tree),
   - `description` (optional): what the values mean and where they come from.

   A field that does not apply to every kind belongs to the kind, not to `ModalitySpec`.

5. The code lives in `src/data/modalities/`, split into three layers:

   ```
   src/data/modalities/
   ├── __init__.py   
   ├── base.py       # ModalityKind, ModalitySpec
   ├── kinds.py      # CompositionalTable, Graph...
   └── registry.py   # TAXONOMY, PATHWAYS...
   ```

   - `base.py`: the abstractions.
   - `kinds.py`: concrete kinds, each defining the parameters and validation rules for one structure of data.
   - `registry.py`: the concrete modalities used in the project.


## Consequences

Positive:
- Every modality has one explicit definition: its structure, its rules and the meaning of its values. Code no longer relies on implicit assumptions.
- A new kind of data (e.g. a graph) means adding a new `ModalityKind`, without changing `ModalitySpec` or existing kinds.

Negative:
- More abstraction up front: two concepts (`ModalityKind`, `ModalitySpec`) to learn before adding a modality.
- Existing code (`BacteriaDataset`, `DataSchema`) must be migrated to the new concepts.