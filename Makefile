.DEFAULT_GOAL := help
.PHONY: help install lock lab clean clean-venv

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create .venv and install all dependencies (incl. dev)
	uv sync

lock: ## Update uv.lock
	uv lock

lab: ## Launch JupyterLab
	uv run jupyter lab

clean: ## Remove caches and notebook checkpoints
	find . -type d \( -name __pycache__ -o -name .ipynb_checkpoints \) -not -path "./.venv/*" -prune -exec rm -rf {} +

clean-venv: ## Remove the virtual environment
	rm -rf .venv
