# W&B local output

Why W&B and what we log: 
[ADR 0002](../adr/0002-ml-experiment-tracking.md),
[ADR 0003](../adr/0003-ml-experiment-logging-policy.md),
[ADR 0004](../adr/0004-backup-and-exit-strategy-for-wandb.md).

## Where it lands

`WandbLogger` writes to the Hydra run directory (`save_dir: ${hydra:runtime.output_dir}` in
`conf/trainer/default.yaml`), so every training run has its own `wandb/` with exactly one run:

```
$OUTPUT_HYDRA/<experiment>/<date>/<time>/
├── .hydra/                          # Hydra config and overrides
├── checkpoints/last.ckpt
├── main.log
└── wandb/
    └── run-<YYYYMMDD_HHMMSS>-<run_id>/     # run name, id, tag ...
        ├── run-<run_id>.wandb              # binary log of everything sent to W&B (used by wandb sync, not for humans to read)
        ├── files/
        │   ├── config.yaml                 # logged hyperparameters
        │   ├── wandb-summary.json          # final metric values
        │   ├── wandb-metadata.json         # host, OS, Python, git, GPU
        │   ├── output.log                  # console output
        │   └── requirements.txt
        └── logs/debug*.log                 # W&B's own logs, only for debugging W&B
```

## How our setup differs from the usual one

Usually all runs share one `wandb/` directory. There the `latest-run` symlink is useful, and
`wandb sync` with no arguments finds every run in that directory.

Hydra gives every run its own directory, so we have a separate `wandb/` in each one. As a result:

- we don't need the `latest-run` symlink, because each `wandb/` holds only one run
  (`settings.symlink: false`)
- `wandb sync` has to be pointed at each run's directory, which is too much typing, so we use
  `make wandb-sync`

## Offline runs

On a machine without internet, set `trainer.logger.0.offline=true`. 
Later, on a machine with access, run `make wandb-sync`.

More: 
[wandb sync](https://docs.wandb.ai/ref/cli/wandb-sync/),
[WandbLogger](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html).
