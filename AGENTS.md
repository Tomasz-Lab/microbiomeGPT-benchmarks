# Code Review Instructions

Review this repository as research ML software for microbiome benchmarking.

Prioritize only high-confidence issues that can affect:
- scientific correctness,
- data leakage between train/val/test,
- sample/feature alignment,
- tensor shapes and dtypes,
- NaN or missing-modality handling,
- metric correctness,
- reproducibility and seed propagation,
- Hydra config/code mismatches, especially `_target_`,
- regressions in the public Python API,
- missing regression tests for meaningful behavior changes.

For data pipelines, check carefully for:
- reordered or misaligned samples/features,
- preprocessing fitted on validation/test data,
- duplicate samples across splits,
- incorrect joins/merges,
- inconsistent modality alignment.

For ML code, check:
- loss/metric input semantics,
- metric state reset between phases/folds,
- shape and dtype assumptions,
- use of the test set during HPO/model selection,
- reproducibility across PyTorch, NumPy, data splitting, Hydra/Submitit.

For Hydra changes, verify configs and Python code together:
- valid `_target_`,
- valid constructor arguments,
- no stale or unused config keys,
- renamed fields updated everywhere.

For dependency changes in `pyproject.toml`:
- flag unnecessary dependencies,
- check compatibility for major upgrades, especially PyTorch, Lightning, pandas, NumPy, scikit-learn, Hydra, OmegaConf, and torchmetrics.
- Do not review `uv.lock` or generated `requirements.txt` line by line.

Ignore normal review of:
- `data/**`
- `output/**`
- `outputs/**`
- `multirun/**`
- `wandb/**`
- `lightning_logs/**`
- `figures/**`
- `uv.lock`
- `requirements.txt`
- generated/cache/build files

For `notebooks/**`, report only clear logic or scientific-correctness bugs.

Ignore Markdown changes except `AGENTS.md` and `SECURITY.md`, unless documentation contradicts actual behavior.

Do not report:
- formatting,
- import ordering,
- line length,
- Ruff issues,
- minor naming/style preferences,
- speculative refactors.

Never read, quote, summarize, or reproduce secrets from:
- `.env`
- `.env.*`
- credentials
- API keys
- W&B tokens
- SSH/private keys
- cluster credentials

`.env.example` is allowed.

If a secret appears committed, report that a secret is present without reproducing its value.

For every finding, include:
1. what is wrong,
2. when it happens,
3. impact,
4. minimal fix.

Prefer a small number of high-confidence findings. If unsure whether something is a real defect, do not report it.