# 🤝 Contributing

## 🛠️ Development

After cloning the repository:

```bash
make setup
```

If any files are modified by the hooks, review the changes, stage them again, and retry the commit.

To run checks manually:
```bash
make check    # Run all: fix + typecheck
```

### Tools

This project uses the following development tools:

| Tool | Purpose | Command |
|------|---------|---------|
| **Ruff** | Linting and formatting | `make fix` |
| **Mypy** | Static type checking | `make typecheck` |


## 📓 Jupyter Notebooks

Register the kernel to access project dependencies in notebooks:

```bash
make kernel   # Register kernel (one time)
make jupyter  # Launch Jupyter
```

Select kernel "Python (gut-former)" in your notebook.


## Configuration Management Principles

- Hydra: Model parameters, training settings, and experiment-specific options belong in Hydra configs. Hydra supports configuration composition and CLI overrides. (Hydra documentation)

- Environment variables  (Deployment-specific configuration): Values that may differ between local development, Docker, HPC, and cloud environments (e.g. dataset and output directories) should be configurable through environment variables. (The Twelve-Factor App — Config)

- settings/ (defined via python code): Settings that normally do not vary between environments, such as Hydra's version_base and default config_name, belong in a centralized settings module. Settings can be imported and used directly in the code



