.DEFAULT_GOAL := help
.PHONY: help install hooks lock lab clean clean-venv \
	data-check fetch test verify backtest champion champion-v2 compare graded submit business submission ai ai-plots scan all

help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create .venv and install all dependencies (incl. dev)
	uv sync
	@$(MAKE) --no-print-directory hooks

hooks: ## Enable the repo's git hooks (.githooks/)
	git config core.hooksPath .githooks
	chmod +x .githooks/*

lock: ## Update uv.lock
	uv lock

lab: ## Launch JupyterLab
	uv run jupyter lab

clean: ## Remove caches and notebook checkpoints
	find . -type d \( -name __pycache__ -o -name .ipynb_checkpoints \) -not -path "./.venv/*" -prune -exec rm -rf {} +

clean-venv: ## Remove the virtual environment
	rm -rf .venv

data-check: ## Check data/raw/ and config/assumptions.yaml are in place (no data read)
	uv run python -m src.checks.check_data_layout

fetch: ## Download public data (FRED, Eurostat, NY Fed) into data/raw/external/
	uv run python -m src.external.fetch_public

test: ## Run the unit tests (synthetic data only)
	uv run pytest -q

verify: ## Local checks on real data: data fixes + driver model sanity check
	uv run python -m src.checks.verify_fixes && uv run python -m src.checks.verify_drivers

backtest: ## Full forecast backtest of the baseline models (slow)
	uv run python -m src.testbench.run

champion: ## Champion v1: signal screening + 2024-26 check (exploratory, no CSV)
	uv run python -m src.external.run_champion

champion-v2: ## Champion v2 option (extended energy candidates), files suffixed _v2
	uv run python -m src.external.run_champion --candidates v2

compare: ## Champion v1 vs v2 (pre-registered rule) + provided-series status table
	uv run python -m src.checks.compare_champions && uv run python -m src.checks.provided_series_table

TAG ?= backtest-freeze-v4

graded: ## One-shot graded run 2022-26 (needs a clean tagged HEAD + passing canaries; refuses a tag already graded)
	uv run python -m src.testbench.pipeline --graded

submit: ## The submission CSV, from the graded run of TAG (default backtest-freeze-v4)
	uv run python -m src.testbench.pipeline --submit --tag $(TAG)

business: ## Business modules (pricing levers, TCO, suppliers...) + REPORT.md
	uv run python -m src.business.run_all

ai: ## AI track (pretrained models): backtest, selection, live forecast (needs the local data)
	uv run --group ai python -m src.ai.run

ai-plots: ## AI track: plot scores and live forecast (run make ai first)
	uv run python -m src.ai.plots

scan: ## NDA leak scan of everything that may be pushed (needs the local data)
	uv run python public/scan_leaks.py

submission: submit ## Alias of submit

all: data-check fetch test champion business ## data-check -> fetch -> test -> champion -> business (stops on first failure)
