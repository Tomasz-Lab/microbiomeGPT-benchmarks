.PHONY: help install clean check fix typecheck kernel jupyter

help:
	@echo "Available commands:"
	@echo ""
	@echo "Setup"
	@echo "  make install   - Install dependencies"
	@echo ""
	@echo "Quality"

	@echo "  make check     - Run all (fix + typecheck)"

	@echo "  make fix       - Auto-fix and format code"

	@echo "  make typecheck - Run type checking (mypy)"

	@echo ""
	@echo "Jupyter"
	@echo "  make kernel    - Register Jupyter kernel"
	@echo "  make jupyter   - Launch Jupyter Notebook"


# -----------------------------------------
# Setup
# -----------------------------------------

install:
	uv sync 

# -----------------------------------------
# Quality
# -----------------------------------------


check: fix typecheck


typecheck:
	uv run mypy .

fix:
	uv run ruff check . --fix
	uv run ruff format .




# -----------------------------------------
# Jupyter
# -----------------------------------------

kernel:
	uv run python -m ipykernel install --user --name=microbiome-gpt --display-name "Python (microbiome-gpt)"

jupyter:
	uv run jupyter notebook

