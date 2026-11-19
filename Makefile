.DEFAULT_GOAL := help
.PHONY: help install hooks lock lab clean clean-venv

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create .venv and install all dependencies (incl. dev)
	uv sync
	@$(MAKE) --no-print-directory hooks

hooks: ## Enable the repo's git hooks (.githooks/) and the notebook output-stripping filter
	git config core.hooksPath .githooks
	chmod +x .githooks/*
	uv run nbstripout --install

lock: ## Update uv.lock
	uv lock

lab: ## Launch JupyterLab
	uv run jupyter lab

clean: ## Remove caches and notebook checkpoints
	find . -type d \( -name __pycache__ -o -name .ipynb_checkpoints \) -not -path "./.venv/*" -prune -exec rm -rf {} +

clean-venv: ## Remove the virtual environment
	rm -rf .venv
