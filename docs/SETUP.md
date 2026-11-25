# Setup: from a fresh clone to all results

> This page contains no data and no licensed prices. It tells you which files to put where, how to fill the local
> config, and in which order to run things.

## a. Prerequisites
- [uv](https://docs.astral.sh/uv/) — installs Python 3.12 automatically if it is missing.
- `make`.
- The data files listed in section c (they are **not** in the repository).

## b. Clone and branch
```bash
git clone https://github.com/erwanfrancotetu/schneider-data-challgenge-2026.git
cd schneider-data-challgenge-2026
git switch <branch>        # e.g. champion/v2
make install               # creates .venv from uv.lock and enables the git hooks
```

## c. Data folder
`data/` is **gitignored: never commit anything in it.** The steel workbook is **licensed** (MEPS/CRU): keep it on your
machine only. If the files are still as delivered (in `data/`, with `data.zip` and the supplier archive extracted in
`data/raw/`), `uv run python -m src.setup_data` copies them into place (never overwrites). Otherwise put the files under `data/raw/` with exactly these names (`make data-check` verifies them):

```
data/raw/
├── panel/                                         from the challenge data.zip
│   ├── rm_prices_monthly_wide.csv
│   ├── rm_prices_monthly_long.csv
│   ├── energy_prices_monthly_wide.csv
│   ├── energy_prices_monthly_long.csv
│   ├── source_catalog.csv
│   ├── energy_source_catalog.csv
│   ├── data_dictionary.csv
│   └── bom_materials.csv
├── RM_Prices_XHEC_Challenge_v1.xlsx               official BOM v2, sent by Schneider ~20 Nov (sheet BOM_Materials)
├── RM_Prices_XHEC_Challenge.xlsx                  OPTIONAL: previous BOM, sensitivity checks only
├── Epoxy for student.xlsx                         Schneider follow-up file (the name Epoxy_for_student.xlsx also works)
├── LME_Copper_Aluminium_EU_Monthly_2016_2026.xlsx challenge file (LME prices, term structure, stocks)
├── RMU_IQI_EU_Price_Benchmark_2017-2026_Synthetic.xlsx   challenge file (synthetic RMU selling prices)
├── Copia_de_Europe_Steel_INDEXES.xlsx             LICENSED MEPS/CRU steel workbook — never commit
├── competitor_teardown_rm6_class.xlsx             competitor teardowns (renamed from
│                                                  benchmark-modules-fonctionnels-rm6-ne-iqi-...-sf6.xlsx; original name also works)
├── suppliers/                                     PitchBook exports, one per company in config/suppliers.local.yaml
│   └── PitchBook_-_<Company>_Company_Financials_<timestamp>.xlsx   (keep the original PitchBook names)
└── external/                                      OPTIONAL, created by `make fetch` (public data)
    ├── fred_monthly.csv
    ├── eurostat_ppi_long.csv
    └── gscpi.csv
```
Supplier files are matched by the `file_keyword` of each company in `config/suppliers.local.yaml` (punctuation and accents
are ignored, so `Supplier_A_(City)` matches `Supplier_A_City`).

## d. config/suppliers.local.yaml and config/assumptions.yaml (local, gitignored)
Supplier names and reference values are under NDA: get the filled `suppliers.local.yaml` privately from the team, or start
from the template:
```bash
cp config/suppliers.example.yaml config/suppliers.local.yaml
```
Same idea for the assumptions:
```bash
cp config/assumptions.example.yaml config/assumptions.yaml
```
Then replace every `null` marked `FILL_LOCALLY`, and fill `material_series_by_product` (which product uses which regional steel
index: under NDA, ask the team). The base month is the forecast origin, **2026-09**. Values are in
EUR per kg (EUR/t ÷ 1000). Where each one comes from:

| Key | Where the value comes from |
|---|---|
| `eur_per_kg_at_base.copper` | D01 at 2026-09 ÷ 1000, from `data/raw/panel/rm_prices_monthly_wide.csv` |
| `eur_per_kg_at_base.aluminium` | D02 at 2026-09 ÷ 1000, same file |
| `eur_per_kg_at_base.carbon_steel` | MEPS Germany CR coil at 2026-09 ÷ 1000, licensed workbook, sheet `MEPS` |
| `eur_per_kg_at_base.stainless_steel` | CRU Germany grade 304 CR-2mm, BASE + Alloy, latest month available at or before 2026-09, ÷ 1000, licensed workbook, sheet `CRU Stainless` |
| `eur_per_kg_at_base.galvanised_steel` | MEPS Germany HDG coil at 2026-09 ÷ 1000, licensed workbook, sheet `MEPS` |
| `eur_per_kg_at_base.epoxy` | column `EPOXY_RESIN_EUR per TON` at 2026-09 ÷ 1000, epoxy file |
| `eur_per_kg_at_base.brass` | (0.63 × D01 + 0.37 × S05 ÷ S06) at 2026-09 ÷ 1000, from `rm_prices_monthly_wide.csv` (S05 = zinc USD/t, S06 = USD per EUR) |
| `pricing.rm6_selling_price_eur` | Schneider Electric P50 for 2026 in the synthetic RMU price benchmark |

`plastics` keeps its placeholder value (no EUR/t plastics data provided). `make data-check` fails while any
`eur_per_kg_at_base` value is still null. Never commit this file: it contains licensed steel prices.

## e. Run order
```bash
make data-check   # 1. files and config in place?
make fetch        # 2. public data
make test         # 3. unit tests
make verify       # 4. optional: checks on the real data
make champion     # 5. champion v1, 2024-26 check (exploratory)
make business     # 6. business modules + REPORT.md
# or all of 1, 2, 3, 5, 6 in one go (stops at the first failure):
make all
```

| Step | What it produces | Where | Approx. runtime |
|---|---|---|---|
| `make data-check` | table of required / optional files; exit code 1 with the fix if something is missing | console | a few seconds |
| `make fetch` | FRED, Eurostat (incl. electronics EL01–EL03, energy E14/E15), NY Fed GSCPI; a source is skipped if its file already has all series | `data/raw/external/` | ~25 s (first time) |
| `make test` | unit tests on synthetic data | console | ~3 s |
| `make verify` | data-fix checks (steel, epoxy, BOM v2, BOM indices) and a DriverRidge sanity check | `data/outputs/checks/` | a few seconds each (~6 s for the driver check) |
| `make champion` | signal screening, 2024-26 check (naive, naive_vol, champion, driver_ridge, LightGBM), exploratory, no CSV; prints its compute footprint | `data/outputs/external/` | ~15 s |
| `make champion-v2` | same with the extended energy candidates, files suffixed `_v2` | same folders | ~20 s |
| `make compare` | v1 vs v2 with the pre-registered rule; status of every provided series | `data/outputs/checks/` | a few seconds |
| `make business` | all business modules, figures, `REPORT.md`, compute footprint | `data/outputs/business/` | ~5 s |
| `make submit` | the submission CSV, read from the graded run of tag `backtest-freeze-v4` (the only command that writes it) | `data/outputs/backtest/graded/<tag>/` | ~1 min (panel build) |
| `make backtest` | full backtest of the baseline models | `data/outputs/testbench/<mode>/` | a few minutes (not part of `make all`) |

## f. Outputs map
| Result | File |
|---|---|
| Submission CSV (champion_wf2, graded run) | `data/outputs/backtest/graded/backtest-freeze-v4/team_01_forecast_202609.csv` |
| Backtest scores (baselines) | `data/outputs/testbench/<mode>/scores.csv`, `overview.csv`, `holdout_selection.csv` |
| Champion holdout scores | `data/outputs/external/champion_holdout_scores.csv` (v2: `..._v2.csv`); forecasts in `champion_holdout.parquet` |
| Signal screening | `data/outputs/external/signal_screen.csv` (v2: `signal_screen_v2.csv`) |
| Champion v1 vs v2 | `data/outputs/checks/champion_v1_vs_v2.csv` |
| Provided-series status | `data/outputs/checks/provided_series_status.csv` |
| Data-fix and driver checks | `data/outputs/checks/verify_fixes_forecasts.parquet`, `verify_drivers_scores.csv` |
| Business summary | `data/outputs/business/REPORT.md` (+ one CSV per module) |
| Business figures | `data/outputs/business/figures/*.png` |

## g. Troubleshooting
- **A file is missing / an error mentions a path** → run `make data-check`; it prints where to put which file, with
  which name.
- **Business forecasts look stale** → the business forecast cache is invalidated automatically when
  `data/outputs/external/champion_holdout.parquet` is newer than it (or when it holds naive_vol forecasts and champion
  forecasts are now available). To force a recompute, delete `data/outputs/business/forecasts_origin_202609.parquet`.
- **Electronics (EL01) missing, or "run fetch_public" warnings** → `make fetch`.
- **`make data-check` says assumptions values are null** → fill them as in section d.

## h. More documentation
- [docs/testbench.md](testbench.md) — forecast test bench, scores, how to plug in a model
- [docs/external_data.md](external_data.md) — public data, signal screening, champion v1 / v2
- [docs/data_fixes.md](data_fixes.md) — licensed steel, epoxy, electronics, BOM v2, driver model
- [docs/business.md](business.md) — business modules and caveats for the slides
