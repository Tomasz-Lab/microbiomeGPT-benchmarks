# 0005. Run multirun jobs in separate processes

Status: Proposed
Date: 2026-11-09

## Context
Hydra's default launcher runs multirun (`-m`) jobs one after another in one Python process.
`WandbLogger` reuses a W&B run that is still open in the process, so every job after the first
logs into the first job's W&B run and its `wandb/` directory. The runs can't be compared and the
local copy is wrong (ADR 0004). Closing the run from our code would mean calling the W&B API
directly, which ADR 0004 rules out.

## Decision
Training experiments use the `submitit_local` Hydra launcher (`hydra-submitit-launcher`). It
starts every multirun job in its own process, so each job gets its own W&B run. On HPC we switch
to `hydra/launcher=submitit_slurm`, which runs every job as its own Slurm job.

## Consequences
Positive:
- one multirun job = one W&B run = one directory, with no changes to the code
- the same mechanism works locally and on Slurm

Negative:
- a new dependency
- local jobs run in parallel, all at once, so a large sweep on a laptop can run out of memory
- every job has a time limit (`hydra.launcher.timeout_min`), so we must set it high enough
- job output goes to `.submitit/` in the sweep directory, not to the terminal
