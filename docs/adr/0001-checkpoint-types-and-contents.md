# 0001. Checkpoint types and contents

Status: Proposed
Date: 2026-11-05

## Context
Today the repository uses a single checkpoint for three different purposes:
- **Resuming training**: a run was interrupted and we want to continue it.
- **Using trained weights**: inference, benchmarking or fine-tuning.
- **Production**: running the model on a server.

One file serving all three roles causes problems:
- **Size.** Each checkpoint stores optimizer state, and every run keeps two of them permanently.
- **Coupling to code.** Loading requires knowing the model class and having a matching Hydra config.
  Running it requires this repository and PyTorch, which may not be available or desirable in production.
- **No rules.** There is no agreed convention for what a checkpoint must contain, so files
  differ between runs and loading code has to guess.

We need clear rules for which checkpoints exist, what each contains, and why.

## Decision

1. `best.ckpt`
    - It is the final model, taken from the epoch with the best validation metric (not from the last epoch).
    - It contains only what is needed to rebuild and use the model:
        - model weights (`state_dict`)
        - model schema (information how to build model, not model pickle)
        - hyperparameters (loss_fn, optimizer etc)
        - Lightning version
        - epoch (to record which epoch the best model came from; useful later for debugging)

2. `last.ckpt`
    - It is the latest state of training. If training is interrupted, we can continue from this point.
    - It is overwritten after every epoch.
    - It contains:
        - everything that is in `best.ckpt`
        - everything needed to continue training: optimizer, LR scheduler, callback and loop state, AMP scaler, etc.

3. `best` and `last` are configured as two separate `ModelCheckpoint` callbacks.

4. We do not keep checkpoints from intermediate epochs (`checkpoint-N`), because we do not plan
   to average the weights of several best epochs.

5. We do not save a recovery checkpoint (mid-epoch), because our epochs are short enough
   and `last.ckpt` is sufficient.

6. Production does not use checkpoints. ONNX is planned.

7. Checkpoints do not contain pickled objects (e.g. `nn.Module`, loss or optimizer instances)
   whenever possible. Instead, they store plain configs (dicts with `_target_`) from which
   the objects are rebuilt. Reasons:
    - **Fragility.** A pickled object is tied to the exact code that created it: renaming
      a class or changing its attributes can make the checkpoint impossible to load.
    - **Duplication.** A pickled `nn.Module` stores the weights again, next to `state_dict`.
    - **Safe loading.** Checkpoints with pickled objects cannot be opened with
      `torch.load(weights_only=True)`, and unpickling can run arbitrary code.

## Consequences

1. We keep only the best epoch, so we lose the option of averaging several epochs.
   In exchange, we avoid the disk usage of checkpoints we probably do not need.
2. Because there is no recovery checkpoint, if an epoch takes a long time and training is interrupted,
   we may lose a lot of time and compute and have to redo that part of training.
3. Some information in the checkpoint (model config, data schema, loss and optimizer configs)
   duplicates what Hydra already stores in the run's config files. This is intentional:
   the checkpoint must contain everything needed to rebuild the model, so it can be used
   without the original Hydra config files or run directory (e.g. after being copied elsewhere).
    - Configs are stored as plain, fully resolved dicts with `_target_` keys. `_target_` is
      only a class path string; the dict can be instantiated with `hydra.utils.instantiate`
      or a simple import-based helper, so loading does not depend on Hydra config files.
    - The checkpoint still depends on the code: the classes named in `_target_` must exist
      at those paths. Renaming or moving them breaks loading of older checkpoints.
      Removing this dependency is the role of ONNX export (see point 6).