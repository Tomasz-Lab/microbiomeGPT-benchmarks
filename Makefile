.PHONY: help install setup hooks clean check fix lint format-check typecheck kernel jupyter

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
