.PHONY: help install setup hooks clean check fix lint format-check typecheck kernel jupyter wandb-sync wandb-sync-dry

help:
	@echo "Available commands:"
	@echo ""
	@echo "Setup"
	@echo "  make install       - Install dependencies"
	@echo "  make setup         - Install dependencies and Git hooks"
	@echo "  make hooks         - Install pre-commit Git hooks"
	@echo ""
	@echo "Quality"
	@echo "  make check         - Run lint, format check and typecheck"
	@echo "  make fix           - Auto-fix and format code"
	@echo "  make lint          - Run Ruff linting"
	@echo "  make format-check  - Check Ruff formatting"
	@echo "  make typecheck     - Run mypy"
	@echo ""
	@echo "Jupyter"
	@echo "  make kernel        - Register Jupyter kernel"
	@echo "  make jupyter       - Launch Jupyter Notebook"
	@echo ""
	@echo "W&B"
	@echo "  make wandb-sync     - Upload unsynced offline runs from OUTPUT_HYDRA and OUTPUT_MULTIRAN_HYDRA"
	@echo "  make wandb-sync-dry - Show what wandb-sync would upload, without uploading"

# -----------------------------------------
# Setup
# -----------------------------------------

install:
	uv sync

setup: install hooks

hooks:
	uvx pre-commit install

# -----------------------------------------
# Quality
# -----------------------------------------

check: lint format-check typecheck

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy .

fix:
	uv run ruff check --fix .
	uv run ruff format .

# -----------------------------------------
# Jupyter
# -----------------------------------------

kernel:
	uv run python -m ipykernel install --user --name=microbiome-gpt --display-name "Python (microbiome-gpt)"

jupyter:
	uv run jupyter notebook

# -----------------------------------------
# W&B
# -----------------------------------------

# Every run keeps its own wandb/ directory inside the Hydra run directory, so plain
# `wandb sync` (which only looks in ./wandb) finds nothing. This finds every wandb/ directory
# under the output dirs from .env and syncs them. Already synced and online runs are skipped. ~ Fasamik39
# WANDB_DIR sends the CLI's own wandb/ directory and logs to a temp dir instead of the repo root.
WANDB_SYNC_DIR = $${TMPDIR:-/tmp}/gut-former-wandb-sync
WANDB_SYNC = set -a; [ -f .env ] && . ./.env; set +a; \
	export WANDB_DIR="$(WANDB_SYNC_DIR)"; mkdir -p "$$WANDB_DIR"; \
	echo "W&B CLI logs: $$WANDB_DIR/wandb"; \
	if [ -z "$$OUTPUT_HYDRA" ]; then echo "OUTPUT_HYDRA is not set (see .env.example)"; exit 1; fi; \
	for dir in "$$OUTPUT_HYDRA" "$$OUTPUT_MULTIRAN_HYDRA"; do \
		[ -n "$$dir" ] && [ -d "$$dir" ] || continue; \
		echo "Searching $$dir"; \
		find "$$dir" -type d -name wandb -prune -exec uv run wandb sync $(1) {} + || exit 1; \
	done

wandb-sync:
	@$(call WANDB_SYNC,)

wandb-sync-dry:
	@$(call WANDB_SYNC,--dry-run)
