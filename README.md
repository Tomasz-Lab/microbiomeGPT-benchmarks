# mGPTbench

![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/License-CC_BY_NC_SA_4.0-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Ruff](https://img.shields.io/badge/code_style-ruff-purple)

mGPTbench benchmarks candidate models for microbiomeGPT, a foundation model of the human gut
microbiome. The models learn representations of microbiome samples from taxonomy (which
organisms are present), functional pathways (what they are doing) and other modalities.
The repository trains, runs and compares them with shared data handling, configuration
and experiment tracking.

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
# uv run python src/mgptbench/main.py --help
# uv run pytest
```

**Option 2:** Manual Activation

Execute commands or scripts directly within the project's environment without manual activation:

```bash
source .venv/bin/activate
```

### 🧪 Tests

Run all tests with `uv run pytest`, or a specific test file with `uv run pytest tests/nmf/test_bicv.py`.

## 📂 Data

A dataset is declared by a recipe in `conf/modataset/<name>.yaml`: which modalities it contains,
which modality spec each one follows, and which source (CSV, Parquet, ...) reads it. Select it with
`modataset=<name>`. Every workflow loads it the same way, with `data.load.load_dataset`.

Two dummy datasets are included in `data/` to get started:

| Recipe | Modalities | Sources |
|---|---|---|
| `gut_metaphlan_humann` (default) | `taxonomy_metaphlan`, `pathways_humann` | CSV |
| `gut_sylph_humann` | `taxonomy_sylph`, `pathways_humann` | Parquet + CSV |

Tables have the sample ID in the first column. To use your own data, add a recipe that points to
your files.

## ⚙️ Hydra Configuration

All workflows run through the Hydra entry point:

```bash
uv run python src/mgptbench/main.py experiment=<experiment> [key=value ...]
```

The default experiment is `nmf_pathways`. Available experiments are:

| Experiment | Workflow |
|---|---|
<<<<<<< HEAD
| `gut_former_training` | Train GUT-FORMer (PyTorch Lightning) |
| `gut_former_inference` | Run GUT-FORMer inference (PyTorch Lightning) |
| `nmf_pathways` | Run NMF (pathways by default) |
| `bicv_pathways` | Select an NMF rank using bi-cross-validation |

Experiments compose the `task`, `model`, and `modataset` config groups (the GUT-FORMer experiments
also compose `datamodule` and `trainer`). Override their settings using dotted keys, for example `modataset=gut_sylph_humann` or
=======
| `autoencoder_training` | Train an autoencoder, GUT-FORMer by default (PyTorch Lightning) |
| `autoencoder_inference` | Run inference from an autoencoder checkpoint (PyTorch Lightning) |
| `gut_former_training_legacy` | Train GUT-FORMer with the original loop |
| `gut_former_inference_legacy` | Run GUT-FORMer inference with the original script |
| `nmf_pathways` | Run NMF (pathways by default) |
| `bicv_pathways` | Select an NMF rank using bi-cross-validation |

Experiments compose the `task`, `model`, and `dataset` config groups (the autoencoder experiments
also compose `datamodule` and `trainer`). Override their settings using dotted keys, for example `dataset.name=my_dataset` or
>>>>>>> develop
`trainer.max_epochs=55`. To inspect the
resolved settings without running a workflow:

```bash
uv run python src/mgptbench/main.py --cfg job experiment=autoencoder_training
```

Hydra does not change the working directory. Each run writes its workflow outputs and Hydra
configuration/log files under
`outputs_hydra/<experiment>/<date>/<time>/`. Multiruns started with Hydra's `-m` option use
`multirun_hydra/<experiment>/<date>/<time>/<job-number>/`.

## 🧠 Model

Training and inference run on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable):
`AutoEncoderSystem` (`src/mgptbench/system.py`) holds the training, validation and prediction
steps, and `BacteriaDataModule` (`src/mgptbench/datamodule.py`) loads and splits the data. Both
can be used directly from notebooks or HPC scripts without Hydra.

After `setup()`, the data module exposes a `DataSchema` (`src/mgptbench/schema.py`) with the
feature names of each modality. The model takes the schema and reads its input sizes from
it, so the runners don't pass data dimensions to the model.

Models are composed from components (ADR 0006).

`conf/model/gut_former.yaml` builds the original Gut-FORMer model, which is frozen in code
with all its hyperparameters, sizes included. New models compose their own components in
`conf/model/` and expose their settings there.

`conf/model/transformer_autoencoder.yaml` is the configurable counterpart for experiments: the
same architecture built from the reusable components in `src/mgptbench/models/components/`,
with GutFormer's values written out. Select it with `model=transformer_autoencoder` and override
any setting, for example:

```bash
uv run python src/mgptbench/main.py experiment=autoencoder_training model=transformer_autoencoder \
  model.tokenizer.embedding_dim=64 model.encoder.stacks.0.num_layers=2 model.latent.latent_size=32
```

`embedding_dim` is set once on the tokenizer and the other components refer to it. Components
whose sizes don't fit together fail at the first training batch with PyTorch's shape error.

The Lightning system (the `LightningModule`, named "system" to avoid confusing it with the model,
as in the [Lightning style guide](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html))
is chosen in config, like the model: `conf/system/autoencoder.yaml` points to `AutoEncoderSystem`.
To try another training system, add a config to `conf/system/` and select it with
`system=<name>`; the system must accept the same constructor arguments as `AutoEncoderSystem`
(`model`, `loss_fn`, `optimizer`, `model_config`, `data_schema`).

### Train from scratch

```bash
uv run python src/mgptbench/main.py \
  experiment=autoencoder_training \
  dataset.name=sample \
  trainer.max_epochs=55
```

Training settings are supplied as Hydra overrides:

| Override | Config default | Description |
|---|---|---|
| `modataset` | `gut_metaphlan_humann` | Dataset recipe |
| `datamodule.taxonomy` | `taxonomy_metaphlan` | Dataset modality used as the taxonomy input (e.g. `taxonomy_sylph`) |
| `datamodule.pathways` | `pathways_humann` | Dataset modality used as the pathways input |
| `trainer.max_epochs` | `55` | Number of training epochs |
| `trainer.accelerator` | `auto` | `auto` (Lightning picks CUDA, MPS or CPU), `cpu`, `cuda`/`gpu`, `mps` |
| `trainer.devices` | `auto` | Number of devices |
| `trainer.profiler` | `null` | Profiler: `simple`, `advanced` or `pytorch` (off by default) |
| `datamodule.batch_size` | `16` | Batch size |
| `datamodule.val_size` | `0.1` | Fraction of samples held out for validation |
| `task.optimizer.lr` | `1.893292917167e-4` | Learning rate |
| `task.seed` | `0` | Random seed |
| `task.resume_from` | `null` | Lightning `.ckpt` to resume (restores optimizer state and epoch) |
| `model.embedding_dim` | `128` | Embedding dimension |
| `model.latent_size` | `64` | Latent space dimension |
| `task.init_weights` | `null` | Model weights to start from (`.ckpt` or legacy `.pt`); optimizer starts fresh |

Outputs in the Hydra run directory:

- `lightning_logs/version_0/metrics.csv` — every metric the model logs (`train_*` and `val_*`),
  written by Lightning's `CSVLogger`. Training and validation values are on separate rows; use
  `pd.read_csv(path).groupby("epoch").first()` for one row per epoch.
- `lightning_logs/version_0/hparams.yaml` — the run's hyperparameters
- `checkpoints/last.ckpt` and `checkpoints/epoch=…-val_loss=….ckpt` (lowest validation loss)

On Apple Silicon `auto` selects MPS, where training fails: the loss uses `torch.cdist`, whose
backward pass is not implemented for MPS. Train there with `trainer.accelerator=cpu`; inference
has no backward pass and runs on MPS.

### Resume and fine-tuning

Resume an interrupted run from a Lightning checkpoint (`trainer.max_epochs` is the total, not the
number of additional epochs):

```bash
uv run python src/mgptbench/main.py \
  experiment=autoencoder_training \
  trainer.max_epochs=100 \
  task.resume_from=/path/to/checkpoints/last.ckpt
```

The resumed run is a new Hydra run, so its `metrics.csv` contains only the epochs trained after
resuming; the earlier epochs stay in the original run's directory.

Checkpoints, however, keep going in the folder of the checkpoint you resume from, so the best
checkpoint is chosen across all epochs, before and after resuming. Resuming from an older
checkpoint of a finished run therefore continues that run and overwrites its `last.ckpt`. To
branch off instead, copy the checkpoint to a new folder first.

Fine-tune from existing weights, for example on another dataset, with a fresh optimizer:

```bash
uv run python src/mgptbench/main.py \
  experiment=autoencoder_training \
  dataset.name=my_dataset \
  trainer.max_epochs=1000 \
  task.init_weights=/path/to/checkpoint.ckpt
```

With a Lightning checkpoint, both `task.resume_from` and `task.init_weights` fail early if the
dataset's taxonomy or pathway columns (names or order) differ from the checkpoint's.

### Inference

Inference requires a trained checkpoint, a Lightning `.ckpt`:

```bash
uv run python src/mgptbench/main.py \
  experiment=autoencoder_inference \
  dataset.name=sample \
  task.checkpoint_path=/path/to/checkpoints/last.ckpt
```

A Lightning checkpoint stores the data schema (feature names per modality), so inference fails
early if the dataset's taxonomy or pathway columns differ from training.

The inference trainer comes from `conf/trainer/inference.yaml`; override it the same way, e.g.
`trainer.accelerator=cpu`.

Inference outputs (`latent_{dataset}.csv`, `pred_taxonomy_{dataset}.csv`, and
`pred_pathways_{dataset}.csv`) are saved in the Hydra run directory by the `PredictionWriter`
callback. Its `filenames` setting in `conf/trainer/inference.yaml` maps each model output to a
file: modality outputs take the dataset's feature names as columns, and other outputs (the latent
space) get `{column_prefix}{i}` columns (`z0`, `z1`, …).

### Legacy scripts

The original scripts are kept in `src/mgptbench/legacy/` and run with the `*_legacy`
experiments. They use the old settings (`task.epochs`, `task.checkpoint_path`) and write a bare
`{date}_{dataset}_checkpoint.pt` state dict:

```bash
uv run python src/mgptbench/main.py experiment=gut_former_training_legacy task.epochs=55
uv run python src/mgptbench/main.py \
  experiment=gut_former_inference_legacy \
  task.checkpoint_path=/path/to/checkpoint.pt
```

## 🧬 NMF

NMF decomposes microbiome data into latent signatures. Supports both pathway and taxonomy data, with optional filtering by health status.

```bash
uv run python src/mgptbench/main.py \
  experiment=nmf_pathways \
  modataset=gut_metaphlan_humann
```

The `nmf_pathways` experiment defaults to pathway data and all samples. Settings can be
overridden with:

| Override | Config default | Description |
|---|---|---|
| `modataset` | `gut_metaphlan_humann` | Dataset recipe |
| `task.modality` | `pathways_humann` | Dataset modality to decompose |
| `task.multiplier` | `100` | Factor applied to the modality's values |
| `task.collapse_to_genus` | `false` | Collapse MetaPhlAn species to genus (taxonomy) |
| `task.subset` | `all` | Sample subset: `all`, `healthy`, or `non-healthy` (needs `study_condition` in metadata) |
| `model.parameters.n_components` | `8` | Number of NMF signatures |
| `model.parameters.n_runs` | `10` | NMF runs |

**Examples:**

```bash
# Pathway NMF, all samples
uv run python src/mgptbench/main.py experiment=nmf_pathways dataset.name=sample

# Taxonomy NMF, healthy samples only
uv run python src/mgptbench/main.py \
  experiment=nmf_pathways \
  task.modality=taxonomy_metaphlan \
  task.multiplier=1 \
  task.collapse_to_genus=true \
  task.subset=healthy
```

NMF outputs are saved in the Hydra run directory:
- `H_{n}_{dataset}_{modality}_{filter}.csv` — signature matrix
- `W_train_{n}_{dataset}_{modality}_{filter}.csv` — sample loadings, train set
- `W_val_{n}_{dataset}_{modality}_{filter}.csv` — sample loadings, val set
- `nmf_stats_{n}_{dataset}_{modality}_{filter}.csv` — per-run metrics

## 🔍 biCV — Rank Selection

Bi-Cross-Validation selects the optimal number of NMF signatures for pathway data by evaluating reconstruction quality on held-out data blocks across a range of ranks. For taxonomy, the number of signatures is established in the literature.

```bash
uv run python src/mgptbench/main.py \
  experiment=bicv_pathways \
  modataset=gut_metaphlan_humann
```

The experiment defaults to pathway data. Rank-selection settings are:

| Override | Config default | Description |
|---|---|---|
| `modataset` | `gut_metaphlan_humann` | Dataset recipe |
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
uv run python src/mgptbench/main.py task=run_data_profiling
```

It profiles the metadata of the selected dataset recipe (`modataset=<name>`), read through the
recipe's metadata source. The HTML report, `data_profiling_report.html`, is saved in the Hydra run directory
under `outputs_hydra/`.


## 🔧 Troubleshooting

To start fresh (remove virtual environment and reinstall):

```bash
make clean
make install
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.
