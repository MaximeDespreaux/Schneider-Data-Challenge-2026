# Schneider Data Challenge 2026

Work repository for the Schneider Electric Data Challenge 2026.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed automatically by uv if missing)
- `make`

## Quickstart

```bash
make install   # creates .venv and installs dependencies from uv.lock
make lab       # launches JupyterLab
```

Run any command inside the environment with `uv run <cmd>`, or activate it with `source .venv/bin/activate`.

## Project layout

```
.
├── data/            # challenge data (not tracked by git); zips extracted to data/raw/
├── notebooks/       # 01_data_overview_eda, 02_forecasting_benchmark
├── reports/figures/ # exported charts
├── src/             # data_loading, features, models, backtest
├── DATA_OVERVIEW.md # file-by-file data review & first EDA findings
├── Makefile
├── pyproject.toml   # project metadata & dependencies
├── uv.lock          # pinned dependency versions
└── README.md
```

## Data

Place the challenge files in `data/`. Its contents are gitignored — only the folder itself is tracked.

## Analysis

- [`notebooks/00_data_profiling_preprocessing.ipynb`](notebooks/00_data_profiling_preprocessing.ipynb) — file-by-file profiling (business context, all columns, missing values, duplicates, quirks) and the preprocessing that writes the model-ready parquet files to `data/processed/` (`src/preprocessing.py`).
- [`DATA_OVERVIEW.md`](DATA_OVERVIEW.md) — senior-DS review of every data file, quality issues, key EDA findings, baselines and roadmap.
- [`notebooks/01_data_overview_eda.ipynb`](notebooks/01_data_overview_eda.ipynb) — the executed EDA behind it.
- [`notebooks/02_forecasting_benchmark.ipynb`](notebooks/02_forecasting_benchmark.ipynb) — leakage-safe rolling-origin benchmark of naive → ETS/ARIMA/Theta → dynamic regression → global LightGBM → conformal ensemble, plus production forecasts, BOM cost fan and forward-buy backtest (`src/features.py`, `src/models.py`, `src/backtest.py`).
- [`notebooks/03_feature_study.ipynb`](notebooks/03_feature_study.ipynb) — correlation matrices (levels vs Δlog, Pearson/Spearman, family blocks, rolling), redundancy pruning, lead–lag CCF + Granger, per-target feature relevance (Spearman, mutual information, regime stability), VIF pruning and a leakage test. It writes `data/processed/feature_selection.parquet` and the training matrices `features_h{3,6,9}.parquet`.
- [`src/data_loading.py`](src/data_loading.py) — `extract_archives()` plus tidy loaders (`load_rm_wide`, `load_energy_wide`, `load_meps`, `load_cru_flat`, `load_cru_stainless`, `load_lme`, `load_epoxy`, `load_benchmark`, `load_supplier_financials`, …).

## Dependencies

- Core: numpy, pandas, pyarrow, scikit-learn, matplotlib, seaborn, xgboost, lightgbm, openpyxl, statsmodels
- Dev: jupyterlab, ipykernel

Add a package with `uv add <pkg>` (or `uv add --dev <pkg>` for dev tools).

## Git hooks

`make install` (or `make hooks`) enables the hooks in `.githooks/`. The `pre-commit` hook blocks commits on `main` that modify config files (`pyproject.toml`, `uv.lock`, `Makefile`, `.gitignore`, `.python-version`, `.githooks/*`) — make those changes on a branch instead.

## Make targets

| Target       | Description                                   |
|--------------|-----------------------------------------------|
| `install`    | Create `.venv`, install deps, enable hooks    |
| `hooks`      | Enable the repo's git hooks (`.githooks/`)    |
| `lock`       | Update `uv.lock`                              |
| `lab`        | Launch JupyterLab                             |
| `clean`      | Remove caches and notebook checkpoints        |
| `clean-venv` | Remove the virtual environment                |
