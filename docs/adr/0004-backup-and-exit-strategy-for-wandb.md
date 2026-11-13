# 0004. Backup and exit strategy for W&B

Status: Proposed
Date: 2026-11-09

## Context
ADR 0002 made W&B our experiment tracker and left a follow-up: 

- decide how we keep a copy of our experiment history and where we migrate if W&B is no longer available.

W&B may become unavailable at any time (loss of the free tier, changed terms, account
blocked), and we may have no warning. Experiment history is crucial for this project,
so we must not lose it and must be able to move to another tool, such as MLflow.

## Decision
1. The local run directory is the source of truth. W&B is a copy of it, used for its UI.
2. All run directories are stored in one agreed location (to be decided, for now it user machine), set via `OUTPUT_HYDRA`.
3. Every training run uses two loggers: `CSVLogger` (tool-independent format) and the
   current tracking tool (`WandbLogger`).
4. We keep the local `wandb/` directory after sync and do not delete it.
5. Code logs only through the Lightning logger interface (`self.log`, `log_hyperparams`).
   We do not call the `wandb` API directly, so switching the tracker is a config change.

## Consequences
Positive:
- if W&B disappears, no metrics or run metadata are lost
- migration means importing local files into the new tool, without needing the W&B API
- switching to another logger, such as `MLFlowLogger`, needs only a config change

Negative:
- dashboards, reports and notes or tags edited in the W&B UI cannot be migrated
- the local storage location must be chosen and needs its own backup
- runs take extra disk space (CSV logs and local W&B files)

## Follow-up
- Choose the storage location and its owner
- Verify that local run directories can be imported into MLflow