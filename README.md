# GUT-FORMer

![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/License-CC_BY_NC_SA_4.0-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Ruff](https://img.shields.io/badge/code_style-ruff-purple)

GUT-FORMer is a Transformer-based model that jointly encodes microbiome taxonomy (which organisms are present) and functional pathways (what they are doing) into a shared latent space. It learns to reconstruct both modalities simultaneously, enabling cross-modal prediction and compact microbiome embeddings.

Developed by TomaszLab.

## 🚀 Setup

### 📋 Prerequisites

- [uv](https://docs.astral.sh/uv/) — Dependency & Python management

[Here](https://docs.astral.sh/uv/getting-started/installation/) are instruction how it can be installed.

### ⚙️ Installation

Set up the environment and install all dependencies:

```bash
uv sync
```
> Note: uv sync will automatically download the correct Python version if it is not already installed.

### 🏃 Running Code in the Environment

You can interact with the virtual environment in two ways:

**Option 1:** Using uv run (Recommended)

```bash
uv run <command>
# Examples:
# uv run python src/gut_former/main.py --help
# uv run pytest
```

**Option 2:** Manual Activation

Execute commands or scripts directly within the project's environment without manual activation:

```bash
source .venv/bin/activate
```

## 📂 Data

Place input files in the `data/` directory:

```
data/
├── taxonomy_{dataset}.csv     # samples × species (pipe-separated taxonomy strings)
├── pathways_{dataset}.csv     # samples × pathways
└── metadata_{dataset}.csv     # sample metadata, must include study_condition column
```

All files are indexed by sample ID. Set `dataset.name` to the `{dataset}` name when running an
experiment.

A sample dataset (`dataset.name=sample`) is included in the repository to get started quickly.

## ⚙️ Hydra Configuration

All workflows run through the Hydra entry point:

```bash
uv run python src/gut_former/main.py experiment=<experiment> [key=value ...]
```

The default experiment is `nmf_pathways`. Available experiments are:

| Experiment | Workflow |
|---|---|
| `gut_former_training` | Train GUT-FORMer (PyTorch Lightning) |
| `gut_former_inference` | Run GUT-FORMer inference (PyTorch Lightning) |
| `gut_former_training_legacy` | Train GUT-FORMer with the original loop |
| `gut_former_inference_legacy` | Run GUT-FORMer inference with the original script |
| `nmf_pathways` | Run NMF (pathways by default) |
| `bicv_pathways` | Select an NMF rank using bi-cross-validation |

Experiments compose the `task`, `model`, and `dataset` config groups (training also composes
`trainer`). Override their settings using dotted keys, for example `dataset.name=my_dataset` or
`trainer.max_epochs=55`. To inspect the
resolved settings without running a workflow:

```bash
uv run python src/gut_former/main.py --cfg job experiment=gut_former_training
```

Hydra does not change the working directory. Each run writes its workflow outputs and Hydra
configuration/log files under
`outputs_hydra/<experiment>/<date>/<time>/`. Multiruns started with Hydra's `-m` option use
`multirun_hydra/<experiment>/<date>/<time>/<job-number>/`.

## 🧠 Model

Training and inference run on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable):
`GutFormerModule` (`src/gut_former/lit_module.py`) holds the training, validation and prediction
steps, and `BacteriaDataModule` (`src/gut_former/datamodule.py`) loads and splits the data. Both
can be used directly from notebooks or HPC scripts without Hydra.

### Train from scratch

```bash
uv run python src/gut_former/main.py \
  experiment=gut_former_training \
  dataset.name=sample \
  trainer.max_epochs=55
```

Training settings are supplied as Hydra overrides:

| Override | Config default | Description |
|---|---|---|
| `dataset.name` | `sample` | Dataset name |
| `trainer.max_epochs` | `55` | Number of training epochs |
| `trainer.accelerator` | `auto` | `auto` (CUDA if available, else CPU), `cpu`, `cuda`/`gpu` |
| `trainer.devices` | `auto` | Number of devices |
| `task.batch_size` | `16` | Batch size |
| `task.optimizer.lr` | `1.893292917167e-4` | Learning rate |
| `task.seed` | `0` | Random seed |
| `task.resume_from` | `null` | Lightning `.ckpt` to resume (restores optimizer state and epoch) |
| `task.init_weights` | `null` | Model weights to start from (`.ckpt` or legacy `.pt`); optimizer starts fresh |
| `model.embedding_dim` | `128` | Embedding dimension |
| `model.latent_size` | `64` | Latent space dimension |

Outputs in the Hydra run directory:

- `{date}_{dataset}_training_stats.csv` — per-epoch metrics (same columns as before; `test_*`
  columns hold the held-out validation split)
- `checkpoints/last.ckpt` and `checkpoints/epoch=…-val_loss=….ckpt` (lowest validation loss)
- `lightning_logs/` — Lightning's CSV logs and hyperparameters

On Apple Silicon `auto` uses the CPU: the loss uses `torch.cdist`, whose backward pass is not
implemented for MPS.

### Resume and fine-tuning

Resume an interrupted run from a Lightning checkpoint (`trainer.max_epochs` is the total, not the
number of additional epochs):

```bash
uv run python src/gut_former/main.py \
  experiment=gut_former_training \
  trainer.max_epochs=100 \
  task.resume_from=/path/to/checkpoints/last.ckpt
```

Fine-tune from existing weights, for example on another dataset, with a fresh optimizer:

```bash
uv run python src/gut_former/main.py \
  experiment=gut_former_training \
  dataset.name=my_dataset \
  trainer.max_epochs=1000 \
  task.init_weights=/path/to/checkpoint.pt
```

With a Lightning checkpoint, both `task.resume_from` and `task.init_weights` fail early if the
dataset's taxonomy or pathway columns (names or order) differ from the checkpoint's. Legacy `.pt`
files store no feature names, so they are loaded with a warning instead.

### Inference

Inference requires a trained checkpoint, either a Lightning `.ckpt` or a legacy `.pt` file:

```bash
uv run python src/gut_former/main.py \
  experiment=gut_former_inference \
  dataset.name=sample \
  task.checkpoint_path=/path/to/checkpoints/last.ckpt
```

A Lightning checkpoint stores the model shape and feature names, so inference fails early if the
dataset's taxonomy or pathway columns differ from training. For a legacy `.pt` file the model is
rebuilt from `model.*` and the dataset, without that check.

Inference outputs (`latent_{dataset}.csv`, `pred_taxonomy_{dataset}.csv`, and
`pred_pathways_{dataset}.csv`) are saved in the Hydra run directory.

### Legacy scripts

The original scripts are kept in `src/gut_former/legacy/` and run with the `*_legacy`
experiments. They use the old settings (`task.epochs`, `task.checkpoint_path`) and write a bare
`{date}_{dataset}_checkpoint.pt` state dict:

```bash
uv run python src/gut_former/main.py experiment=gut_former_training_legacy task.epochs=55
uv run python src/gut_former/main.py \
  experiment=gut_former_inference_legacy \
  task.checkpoint_path=/path/to/checkpoint.pt
```

## 🧬 NMF

NMF decomposes microbiome data into latent signatures. Supports both pathway and taxonomy data, with optional filtering by health status.

```bash
uv run python src/gut_former/main.py \
  experiment=nmf_pathways \
  dataset.name=sample
```

The `nmf_pathways` experiment defaults to pathway data and all samples. Settings can be
overridden with:

| Override | Config default | Description |
|---|---|---|
| `dataset.kind` | `pathways` | Data type: `pathways` or `taxonomy` |
| `dataset.name` | `sample` | Dataset name |
| `dataset.subset` | `all` | Sample subset: `all`, `healthy`, or `non-healthy` |
| `model.parameters.n_components` | `8` | Number of NMF signatures |
| `model.parameters.n_runs` | `10` | NMF runs |

**Examples:**

```bash
# Pathway NMF, all samples
uv run python src/gut_former/main.py experiment=nmf_pathways dataset.name=sample

# Taxonomy NMF, healthy samples only
uv run python src/gut_former/main.py \
  experiment=nmf_pathways \
  dataset.kind=taxonomy \
  dataset.name=sample \
  dataset.subset=healthy
```

NMF outputs are saved in the Hydra run directory:
- `H_{n}_{dataset}_{type}_{filter}.csv` — signature matrix
- `W_train_{n}_{dataset}_{type}_{filter}.csv` — sample loadings, train set
- `W_val_{n}_{dataset}_{type}_{filter}.csv` — sample loadings, val set
- `nmf_stats_{n}_{dataset}_{type}_{filter}.csv` — per-run metrics

## 🔍 biCV — Rank Selection

Bi-Cross-Validation selects the optimal number of NMF signatures for pathway data by evaluating reconstruction quality on held-out data blocks across a range of ranks. For taxonomy, the number of signatures is established in the literature.

```bash
uv run python src/gut_former/main.py \
  experiment=bicv_pathways \
  dataset.name=sample
```

The experiment defaults to pathway data. Rank-selection settings are:

| Override | Config default | Description |
|---|---|---|
| `dataset.name` | `sample` | Dataset name |
| `task.n_components_min` | `2` | Minimum number of signatures to test |
| `task.n_components_max` | `10` | Maximum number of signatures to test |
| `model.parameters.n_runs` | `10` | NMF runs per fold per signature |

The `bicv_nmf_stats.csv` output in the Hydra run directory contains metrics for all folds,
signatures, and runs.

The optimal signature count is printed at the end of the run.


## Data profiling

Generate an interactive Sweetviz report with dataset summaries, column distributions, and feature
associations:

```bash
uv run python src/gut_former/main.py task=run_data_profiling
```

By default, this profiles `data/metadata_sample.csv`. Set `task.path=path/to/data.csv` to use
another CSV. The HTML report, `data_profiling_report.html`, is saved in the Hydra run directory
under `outputs_hydra/`.


## 🔧 Troubleshooting

To start fresh (remove virtual environment and reinstall):

```bash
make clean
make install
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.
