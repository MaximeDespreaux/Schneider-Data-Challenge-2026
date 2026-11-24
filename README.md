# Schneider Data Challenge 2026 · Team 1 (MSc DSAIB X-HEC 2025-27)

Forecasting raw-material prices 3, 6 and 9 months ahead for two medium-voltage ring main units (**RM6** and
**RM AirSeT**), turning the forecasts into a product cost, a quote rule for fixed-price offers (Goal A) and a
commercial case for SF6-free AirSeT where no regulation requires it (Goal B).

> **No data in this repository.** The challenge data, the licensed MEPS/CRU steel prices and every output stay in
> `data/` (gitignored) and in `config/assumptions.yaml` (gitignored). This README contains no prices; the results
> below are scores *relative to the naive forecast*.

## 1. The client's questions

1. Where are the raw-material prices heading at 3 / 6 / 9 months, and what does it mean for the ETO offer price?
2. Which pricing proposal for 3-, 6- and 9-month offers: price adjustment, quote validity, indexation clause, risk buffer,
   under downside / baseline / upside scenarios?
3. In markets with no SF6 regulation (APAC, Middle East, Americas), what strategy wins the voluntary switch to AirSeT?

## 2. Approach in one picture

```
data (challenge panel + licensed steel + epoxy + public data)  →  causal panel (publication lags, no look-ahead)
   →  forecasts P10/P50/P90 per material, rolling-origin backtest vs "price stays the same"
   →  BOM cost index per product (cost-weighted: price × kg, base = forecast origin = 100)
   →  margin at risk, quote rule (fixed price + buffer / indexation / copper hedge), procurement levers, TCO
```

| Step | Where | Doc |
|---|---|---|
| Data loading, fixes, official BOM (v1.xlsx), licensed steel, epoxy, brass, electronics | `src/data.py`, `src/licensed.py`, `src/checks/` | [data_fixes.md](docs/data_fixes.md), [SETUP.md](docs/SETUP.md) |
| Public data (FRED/IMF/BLS, Eurostat, NY Fed GSCPI), long history, signals | `src/external/` | [external_data.md](docs/external_data.md) |
| Test bench: rolling origins, baselines, metrics, leak audit | `src/testbench/` | [testbench.md](docs/testbench.md) |
| Hardened backtest: availability per series, canaries, DM / Clark-West / Holm / bootstrap / MDE, graded run | `src/testbench/pipeline.py`, `config/availability.yaml` | [preregistration.md](docs/backtest/preregistration.md) |
| Business modules (Goal A and B) | `src/business/` | [business.md](docs/business.md) |

## 3. Main results (relative to the naive forecast, 2024-26 check)

> **To re-run** (`make champion`). The table predates the fixes of 2026-11-24: the champion is now screened again at every
> origin, inputs are masked by the live ragged edge, and RM6 steel is valued at its own price at each origin. 2024-26 is
> not an untouched holdout: signals are chosen on targets before 2024, but this period also decides champion v1 vs v2 and
> whether LightGBM replaces the champion, and it was looked at when setting the `naive_vol` bands
> (see [preregistration.md](docs/backtest/preregistration.md), "Prior looks").

| Model | Error of P50 (3 / 6 / 9 m) | Quantile loss (3 / 6 / 9 m) | P10–P90 coverage (target 0.80) |
|---|---|---|---|
| **Champion** (public signals + volatility-scaled bands) | ×0.97 / ×0.98 / ×1.00 | **×0.96 / ×0.96 / ×0.96** | 0.82–0.84 |
| `naive_vol` (volatility-scaled bands) | ×1.00 | ×0.99 / ×0.97 / ×0.96 | 0.81–0.85 |
| Driver ridge | ×1.02 / ×1.10 / ×1.18 | ×1.00 / ×1.07 / ×1.10 | 0.83 |
| LightGBM quantile (ML benchmark) | ×1.15 / ×1.22 / ×1.28 | ×1.13 / ×1.21 / ×1.25 | 0.64–0.68 |

What this means:
- **Exchange-traded prices (copper, aluminium, nickel, zinc, iron ore) behave like a random walk**: no model beats
  "the price stays the same" on the central forecast. Their risk is managed (hedging, indexation), not predicted.
- **Slow EU price indices (steel, galvanised, aluminium, plastics) are partly predictable at 3 months**, from public
  leading signals (US steel prices, upstream commodity prices, index-vs-market gap).
- **Ranges matter more than point forecasts**: standard bands collapse in crises; volatility-scaled bands hold.
- The ML benchmark overfits on ~100–400 months of history (also found in our first benchmark): kept as a reported row,
  rejected by the pre-registered rule.
- Diagnostics: ADF + KPSS (log levels non-stationary, monthly changes stationary), STL (weak seasonality), Ljung-Box on
  forecast errors (no remaining autocorrelation at 3 months), Kupiec on coverage (hardened backtest).

## 4. Proof of no look-ahead

- Every model only receives data available at the forecast origin; derived series (BOM indices) are rebuilt at each
  origin with weights known at that origin.
- **Perturbation audit**: the whole future of every series is replaced by noise → forecasts unchanged
  (`uv run python -m src.testbench.audit`).
- **Leak canaries on the real data** (planted leak detected, permuted future gives no gain, truncation and off-by-one
  checks): pass in both modes (`uv run python -m src.testbench.pipeline --canaries`).
- Signals and settings were chosen on data before 2024 (or on the dev window 2010–2021 for the hardened backtest). Two
  later decisions (champion v1 vs v2, LightGBM vs champion) use 2024–26 with a rule fixed beforehand.

## 5. Prerequisites and quickstart

- [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed automatically by uv if missing)
- `make` (on Windows without `make`, run the `uv run python -m ...` command shown for each target in the `Makefile`)

```bash
make install      # creates .venv and installs dependencies from uv.lock
make data-check   # check data/raw/ and config/assumptions.yaml are in place (see docs/SETUP.md)
make all          # data-check -> fetch -> test -> champion -> business
make help         # list all targets
```

Run any command inside the environment with `uv run <cmd>`. Full setup and data layout: [docs/SETUP.md](docs/SETUP.md).
Local configuration: copy `config/assumptions.example.yaml` to `config/assumptions.yaml` and fill the `FILL_LOCALLY`
values (licensed prices: never commit this file).

## 6. Data and licences

| Source | Content | Licence / status | In git? |
|---|---|---|---|
| Challenge panel (`data.zip`) | monthly material and energy prices (IMF via FRED, Eurostat, ECB, US BLS, Ember, Bank of Japan, JEPX) | public sources, use with attribution (Eurostat and Ember: CC BY 4.0) | no |
| BOM (`RM_Prices_XHEC_Challenge_v1.xlsx`) | material masses per product | Schneider internal, challenge use only | no |
| MEPS / CRU steel workbook | real steel prices (carbon, stainless, galvanised) | **licensed**; use authorised by Schneider, kept local, never published | no |
| LME, epoxy, competitor teardown, supplier financials (PitchBook), synthetic price benchmark | challenge files | challenge use only (PitchBook: PitchBook terms) | no |
| Public data added by us | FRED (IMF, BLS, Fed, CBOE, EPU series), Eurostat `sts_inppd_m`, NY Fed GSCPI | free with citation; Eurostat CC BY 4.0 | no (downloaded by `make fetch`) |

| Software / model | Licence |
|---|---|
| numpy, pandas, scipy, scikit-learn, statsmodels, seaborn, jupyterlab | BSD-3-Clause |
| matplotlib | Matplotlib licence (PSF-based) |
| lightgbm, openpyxl, pyyaml, pytest | MIT |
| xgboost, pyarrow, requests | Apache-2.0 |
| AI track: Chronos-Bolt / Chronos-2 pretrained models, torch | Apache-2.0 / BSD-3-Clause |
| AI track: Qwen3-14B run locally through Ollama | Apache-2.0 |
| joblib | BSD-3-Clause |

## 7. Compute and CO₂

Measured on a laptop, CPU only (estimate: ~30 W, ~0.25 kgCO₂e/kWh EU mix):

| Run | Model time | Estimate |
|---|---|---|
| `make champion` (screening + 2024-26 check of 5 models + CSV) | ~130 s | ~0.3 gCO₂e |
| `make business` (12 modules) | ~10–20 s | ~0.03 gCO₂e |
| baseline backtest (6 models × 16 targets × 82 origins) | ~80 s | ~0.2 gCO₂e |

The usage of AI coding assistants by the team is **not** included (not measurable from our side).

## 8. AI tools disclosure

- **Claude (Anthropic), through Claude Code**, was used by team members to write and review code, run analyses and draft
  documentation. Every result was re-run and checked by the team; tests and leak audits are part of the repository.
- **Chronos-Bolt / Chronos-2** (Amazon, Apache-2.0), pretrained time-series models run locally and zero-shot, are
  evaluated in the AI track (`src/ai/`).
- **Qwen3-14B** (Alibaba, Apache-2.0), run locally through Ollama, is tested for feature proposals, news scoring and forecasts. The
  Claude API can be used instead; both go through the same NDA gate (`src/ai/guard.py`). Status and results:
  [src/ai/PLAN.md](src/ai/PLAN.md).
- The `ai_tools` column of the submission CSV lists the tools used.

## 9. Responsible AI

- **Confidentiality**: no challenge data, licensed price or derived output in git; licensed steel prices are used
  internally only; the CI runs on synthetic data.
- **No look-ahead and no cherry-picking**: publication lags, perturbation audit, canaries, choices made before the
  evaluation period, pre-registered decision rules.
- **Honest reporting**: we report where forecasting does *not* work (exchange-traded prices) and keep the naive
  forecast as the reference everywhere.
- **Uncertainty first**: every forecast comes with P10/P50/P90; buffers and quote rules are sized on the ranges, not on
  the central value.
- **Frugality**: small statistical models (seconds of CPU) are preferred unless a heavier model proves better on
  held-out data.
- **Human in the loop**: the outputs support pricing and procurement decisions; they do not take them.

## 10. Data limitations and assumptions

- BOM changed during the challenge (official version: v1.xlsx); differences with earlier versions are documented.
- Unidentified material ("Other") is not priced; a sensitivity analysis shows its impact (small for RM6, material for
  AirSeT).
- Proxies: brass = copper/zinc mix; plastics = EU index × assumed €/kg; electronics = Eurostat C26.1 index.
- Selling prices come from a synthetic benchmark (trends only); real margins and the AirSeT price are assumptions.
- Ranges are calibrated on 1992–2026 history and remain slightly optimistic during strong rallies.
- Recent months of official indices can be revised after publication.

## 11. Business modules

Goal A pricing levers, Goal B and decarbonisation analyses (`src/business/`, settings in `config/business.yaml`):

```bash
uv run python -m src.business.run_all
```

Outputs (confidential, gitignored) go to `data/outputs/business/`, with a summary in `REPORT.md`. See `docs/business.md`.
The business modules read the forecasts of the graded model (`champion_wf`) when a graded run exists, otherwise the
champion, otherwise `naive_vol`, and refresh automatically when a newer forecast file appears.

## 12. Contributions (by branch)

| Member | Main contributions |
|---|---|
| Bhavesh Chauhan | data fixes (licensed steel, epoxy, brass, electronics, BOM v2), business modules, champion v2, organiser clarifications |
| Erwan Franco Tetu | first EDA and forecasting reports, AI track (Chronos) |
| Maxime Despreaux | hardened backtest, statistical tests, graded pipeline, margin-protection replay |
| Mohamed Soro | data consolidation and exploration |
| Julia Rouzou | test bench and leak audit, public data and champion, extra modelling (LightGBM row, KPSS/STL, Ljung-Box) |

## 13. Repository layout

```
.
├── config/          # assumptions and suppliers templates (*.example.yaml), business.yaml, availability.yaml
├── data/            # challenge data and outputs (not tracked by git)
├── docs/            # setup, data fixes, external data, test bench, business, pre-registration
├── notebooks/       # test-bench notebooks, committed without outputs (other notebooks stay local)
├── public/          # NDA-safe copies of the analysis notebooks and reports, and the leak scanner
├── src/             # data, external, testbench, business, checks; first benchmark (features, models, backtest)
│   └── ai/          # AI track: pretrained models, LLM features / news / analyst (see src/ai/PLAN.md)
├── tests/           # unit tests (synthetic data; the few that need the local data skip without it)
├── Makefile
├── pyproject.toml   # project metadata & dependencies
└── uv.lock          # pinned dependency versions
```

Analysis notebooks and reports, in their NDA-safe version ([public/README.md](public/README.md)):

- [00_data_profiling_preprocessing](public/notebooks/00_data_profiling_preprocessing.ipynb): file-by-file profiling and preprocessing (`src/preprocessing.py`)
- [01_data_overview_eda](public/notebooks/01_data_overview_eda.ipynb) and [DATA_OVERVIEW](public/DATA_OVERVIEW.md): data review and first findings
- [02_forecasting_benchmark](public/notebooks/02_forecasting_benchmark.ipynb): first rolling-origin benchmark (`src/features.py`, `src/models.py`, `src/backtest.py`)
- [03_feature_study](public/notebooks/03_feature_study.ipynb): correlations, lead-lag, feature relevance

## 14. Git hooks

`make install` (or `make hooks`) enables the hooks in `.githooks/`. The `pre-commit` hook blocks commits on `main` that modify config files (`pyproject.toml`, `uv.lock`, `Makefile`, `.gitignore`, `.python-version`, `.githooks/*`) — make those changes on a branch instead.

## 15. Make targets

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
| `ai`         | AI track: pretrained models (installs torch)   |
| `ai-plots`   | AI track figures (run `ai` first; private)     |
| `scan`       | NDA leak scan before every push                |
| `all`        | data-check, fetch, test, champion, business    |
