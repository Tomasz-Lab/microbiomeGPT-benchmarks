# Security Policy

This repository is research ML software for microbiome benchmarking.

The main security concerns are:
- secrets and credentials,
- sensitive research data,
- unsafe model/checkpoint deserialization,
- dependency supply-chain risk,
- GitHub Actions permissions and secret handling,
- access to HPC/SLURM infrastructure.

## Reporting vulnerabilities

Report security-sensitive issues privately when they involve:
- credentials or tokens,
- private infrastructure,
- sensitive research data,
- exploitable code paths.

Do not post live secrets or sensitive data in public issues.

## Secrets

Never commit:
- `.env`
- W&B API keys,
- GitHub tokens,
- SSH/private keys,
- cluster credentials,
- service credentials.

`.env.example` may be committed, but must contain placeholders only.

If a secret is committed:
1. revoke or rotate it,
2. remove it from the repository,
3. check Git history, CI logs, artifacts, and W&B logs for exposure.

Deleting the secret in a later commit is not sufficient if the credential is still valid.

## Research data

Do not commit patient-identifiable information, PII, or restricted participant data.

Files under `data/` must contain only data that is safe and permitted to distribute.

Before committing data, verify:
- no PII or protected health information,
- redistribution is allowed,
- dataset license/data-use terms are respected.

Derived data may still inherit restrictions from the original dataset.

## Model checkpoints and serialization

Treat external checkpoints and serialized Python objects as untrusted.

Prefer:

```python
torch.load(path, weights_only=True)
```

when compatible with the checkpoint.

Do not use `pickle.load`, `joblib.load`, or unrestricted `torch.load` on untrusted files.

Only deserialize full Python objects from trusted sources.

## Hydra configuration

Hydra `_target_` can import and instantiate Python objects.

Do not run untrusted Hydra configs that can control `_target_` or arbitrary object construction.

Only execute configs from trusted repository or user sources.

## Dependencies

Dependencies are managed through `uv`.

Review changes to `pyproject.toml`, especially major upgrades of:
- PyTorch,
- Lightning,
- pandas,
- NumPy,
- scikit-learn,
- Hydra,
- OmegaConf,
- torchmetrics.

Avoid unnecessary or poorly maintained dependencies.

Treat unexpected lockfile changes as something to investigate.

## GitHub Actions

Use least-privilege permissions.

Prefer:

```yaml
permissions:
  contents: read
```

and add write permissions only where required.

Do not:
- hard-code secrets in workflow files,
- echo secrets,
- expose secrets to pull requests from forks,
- store secrets in artifacts or cache keys.

Be especially careful with `pull_request_target`.

Do not execute untrusted PR code in a privileged `pull_request_target` workflow.

## Third-party GitHub Actions

Prefer maintained and trusted actions.

For sensitive workflows, prefer pinning third-party actions to immutable commit SHAs.

Review:
- requested permissions,
- access to secrets,
- maintenance status,
- publisher trust.

## W&B

W&B credentials must come from environment variables or secret storage.

Do not:
- hard-code `WANDB_API_KEY`,
- log it,
- store it in Hydra configs,
- expose it in notebooks,
- expose it in CI output.

Avoid sending unrelated environment variables or secrets to experiment tracking.

## HPC / SLURM

Do not commit cluster credentials or SSH keys.

Do not run untrusted code, configs, checkpoints, or branches on privileged cluster accounts.

Remember that cluster jobs may have access to:
- shared storage,
- private datasets,
- SSH credentials,
- W&B credentials,
- other research infrastructure.

## Notebooks

Notebooks can execute arbitrary Python code.

Review external notebooks before running them in environments containing:
- secrets,
- private data,
- cluster access.

Before committing notebooks, check outputs for:
- credentials,
- private hostnames,
- usernames,
- sensitive paths,
- research data that