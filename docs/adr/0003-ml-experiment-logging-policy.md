# 0003. ML experiment logging policy

Status: Proposed
Date: 2026-11-09

## Context
With decision 0002, we decided that W&B will be the default way to handle ML experiment tracking. However, it has not yet been established what exactly (what kind of information) should and should not be tracked, and why.

The main goals of tracking are to:
- easily compare different models and pick the best one for a given scenario
- be able to recreate any run (and its model weights) from the logged information
- do not send sensitive or non-public data to an external service

## Decision

### 1. Run identification
- project: the W&B project the run belongs to
- run name:
    - a label for a single run, shown in the run list and chart legends
    - short and easy to read
    - convention: `<model-name>__<dataset>__<%m%d-%H%M%S>`, e.g. `gut_former__sample__1109-141437`
    - `__` separates the parts, because model and dataset names may contain a single `_`
    - not used for filtering (use the run parameters below) and not guaranteed to be unique (the run ID is)
- model name
- model version
- dataset name
- run purpose: `debug`, `test-run`, `baseline`, etc.
- job type: `train`, `inference`, etc.
- group: set when runs belong together, e.g. the jobs of one Hydra multirun, so they can be
  compared side by side
- notes: a short description of the run's purpose

### 2. Code
Restore the exact code of any run, even with uncommitted changes.

- git commit hash
- git branch name
- dirty flag: `true` if the run started with uncommitted changes
- `code.patch`: saved in the run directory when dirty, with all uncommitted changes, including
  new files in `src/` and `conf/`

### 3. Environment and hardware
- OS 
- hostname
- Core python lib version (torch etc.) just for quick info (the full info stays in uv.lock)
- hardware used (CPU / GPU type and count)
- execution environment: local or HPC cluster (with Slurm job ID)

### 4. Data
- dataset name
- root: common directory of the input files (machine-specific, for information only)
- input files: 
    - path relative to root
    - sha256 of each file
- number of samples, 
- per split (train, val, …) 
    - its sample count
    - a hash of its sample IDs

### 5. Training configuration
- full Hydra config (model architecture, hyperparameters, number of epochs, etc.)
- random seed
- optimizer and learning rate scheduler etc.

### 6. Training process and results
- loss and metrics per epoch (training and validation)
- final metrics on the test set
- best epoch and best value of the monitored metric

### 7. Logs
- console output (stdout / stderr)

### 8. Artifacts
- model weights stay local, as we have not yet established whether we can upload them
- only the local checkpoint path and its hash are logged, to link the weights with the run

### 9. What is not logged
- datasets or any part of the raw data (until a data policy is decided)
- model weights (until a weights  policy is decided)
- secrets (API keys, passwords, tokens)

## Consequences
Positive:
- it is clear for everyone what can and cannot be logged
- no sensitive data is sent to W&B
- the main goals of tracking are addressed, so we should have enough information to compare models or recreate them

Negative:
- logging some of the information (dataset hash, split definition, checkpoint hash) requires extra code in the project
- full reproduction still depends on local storage, because datasets and model weights are not stored in W&B
- the policy needs to be revisited once the data policy (sending data and weights to W&B) is decided
