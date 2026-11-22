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

Full setup and data layout: [docs/SETUP.md](docs/SETUP.md)

```bash
make data-check   # check data/raw/ and config/assumptions.yaml are in place
make all          # data-check -> fetch -> test -> champion -> business
make help         # list all targets
```

## Project layout

```
.
├── data/            # challenge data (not tracked by git)
├── Makefile
├── pyproject.toml   # project metadata & dependencies
├── uv.lock          # pinned dependency versions
└── README.md
```

## Data

Place the challenge files in `data/`. Its contents are gitignored — only the folder itself is tracked.

## Dependencies

- Core: numpy, pandas, pyarrow, scikit-learn, matplotlib, seaborn, xgboost, lightgbm
- Dev: jupyterlab, ipykernel

Add a package with `uv add <pkg>` (or `uv add --dev <pkg>` for dev tools).

## Business modules

Goal A pricing levers, Goal B and decarbonisation analyses (`src/business/`, settings in `config/business.yaml`):

```bash
uv run python -m src.business.run_all
```

Outputs (confidential, gitignored) go to `data/outputs/business/`, with a summary in `REPORT.md`. See `docs/business.md`.

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
| `data-check` | Check data files and local config are in place |
| `fetch`      | Download public data into `data/raw/external/` |
| `test`       | Run the unit tests (synthetic data only)       |
| `verify`     | Local checks on real data (data fixes, drivers)|
| `backtest`   | Full baseline backtest (slow)                  |
| `champion`   | Champion v1 + draft submission CSV             |
| `champion-v2`| Champion v2 option (files suffixed `_v2`)      |
| `compare`    | Champion v1 vs v2 + provided-series table      |
| `business`   | Business modules + `REPORT.md`                 |
| `submission` | Alias of `champion`                            |
| `all`        | data-check, fetch, test, champion, business    |
