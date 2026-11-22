# Data fixes: licensed steel, epoxy, electronics, new BOM

> This page describes code only. It contains no prices from the licensed files or from the challenge data.
> All data lives in `data/raw/` (gitignored), and the filled `config/assumptions.yaml` is gitignored too.

## New series in the panel

`src/testbench/run.py::build_panel()` now fills these columns. Each one is optional: if its file is missing, the
panel is still built and a warning tells you what to do.

| Series | What it is | Source file / loader | Unit | Publication lag |
|---|---|---|---|---|
| `L01` | Carbon steel, cold-rolled coil, Germany (MEPS) | `data/raw/Copia_de_Europe_Steel_INDEXES.xlsx`, sheet `MEPS`, column "CR coil" — `d.load_licensed_steel()` | EUR/t | 1 month |
| `L02` | Stainless 304, cold-rolled 2 mm, Germany (CRU): base price + alloy surcharge | same file, sheet `CRU Stainless`, columns "BASE" + "Alloy" | EUR/t | 1 month |
| `L03` | Hot-dip galvanised coil, Germany (MEPS) | same file, sheet `MEPS`, column "HDG coil" | EUR/t | 1 month |
| `EP01` | Epoxy resin, Europe | `data/raw/Epoxy for student.xlsx` (or `Epoxy_for_student.xlsx`), column `EPOXY_RESIN_EUR per TON` — `d.load_epoxy_series()` | EUR/t | 1 month |
| `EL01` | Producer price index, NACE C26.1 "electronic components and boards", EU27 | Eurostat `sts_inppd_m`, downloaded by `fetch_public` into `data/raw/external/eurostat_ppi_long.csv` | index (2021 = 100) | 2 months |
| `EL02` | Same index, Hungary | same | index | 2 months |
| `EL03` | Same index, France | same | index | 2 months |
| `BR01` | Brass, derived: 0.63 × copper `D01` + 0.37 × zinc `S05` / `S06` (USD per EUR) | computed from the panel | EUR/t | 1 month |

- `L01`–`L03` are the challenge's own placeholders for licensed steel prices. They were empty in the panel; they are
  now filled from the licensed MEPS/CRU workbook. Months are normalised to the first day of the month (the CRU sheet
  uses mid-month dates). `L02` is left empty when either the base or the surcharge is missing — it is never filled forward.
- `BR01` exists only so the BOM index can price brass. It is **not** a forecast target.
- Publication lags are in `src/data.py::PUBLICATION_LAG` and in `src/testbench/vintage.py::REALTIME_LAG_MONTHS`
  (used by the `realtime` backtest mode).
- `src/external/signals.py::extend_history` extends `L01`–`L03` back in time from the same workbook, `EP01` from the
  epoxy file (it starts in 2018-01 and stays empty before that), and `EL01`–`EL03` from the long Eurostat download.

## Forecast targets

- `d.SCHNEIDER_RM_TARGETS`: Schneider's 14 original material series. **Only these** (plus the two BOM indices) go into
  the submission CSV (`src/external/run_champion.py`).
- `d.EXTRA_TARGETS = ["L01", "L02", "L03", "EP01", "EL01"]`: forecast and scored by the test bench, not submitted.
- `d.RM_TARGETS = SCHNEIDER_RM_TARGETS + EXTRA_TARGETS`.
- `src.testbench.run.default_targets(panel)` keeps only the series that exist and have data in the panel, then adds
  the two BOM indices. `DEFAULT_TARGETS` (14 series + 2 indices) stays as a backward-compatible constant.

## BOM and BOM cost index

- **BOM version.** `d.load_bom()` now defaults to `version="v2"`: `RM_Prices_XHEC_Challenge_v1.xlsx`, the updated
  official BOM Schneider sent around 20 Nov. `version="root"` (`RM_Prices_XHEC_Challenge.xlsx`, previous version) is
  kept for sensitivity checks; `version="csv"` is obsolete. In the config: `bom.version: xlsx_v2`.
- Rows whose label contains "Epoxy" become family `Epoxy`; `Brass` is kept; "To confirm" rows and the summary rows at
  the bottom of the sheet are dropped. The "big unlabelled RM6 slice = carbon steel" rule only applies to the old files.
- **Prices used by the index** (`config/assumptions.yaml`, section `material_series`):
  carbon / stainless / galvanised steel → `L01` / `L02` / `L03` (MEPS/CRU, EUR/t); copper → `D01`; aluminium → `D02`;
  brass → `BR01`; plastics → `S11`; epoxy → `EP01`; electronics → `EL01`.
- Series in EUR/t (`D01`, `D02`, `L01`–`L03`, `EP01`, `BR01`) are used directly. Index series (`S11`, `EL01`, and the
  Eurostat steel indices if selected) are turned into EUR/kg with a constant `k = EUR/kg at base ÷ index at base`, so
  they only drive the *changes*.
- **Electronics** (only if `extras.electronics.include: true`) is one pseudo-row per product: 1 "box" valued at
  `extras.electronics.eur_per_box_at_base.<rm6|airset>` EUR, moving with `EL01`.

**Electronics is off by default.** The electronics price index (`EL01`) is still downloaded and forecast like any
other series, but it is excluded from the submitted BOM index (`extras.electronics.include: false`). Schneider's BOM
has no electronics, and the EUR-per-box figure is a weak assumption: at 300 EUR per box it would carry about a
quarter of AirSeT's index weight. Electronics is handled instead as a separate slider in the business analysis.
- **Epoxy extra** (`extras.epoxy.include`) is now `false`: epoxy is in the v2 BOM. Turning it on with a BOM that
  already has an epoxy row raises an error (it would be counted twice).
- **Base month.** Values at the base month use the last published value at or before that month, because `L02` and
  `EL01` may not be published for the base month yet.
- **Missing series.** If a series the index needs is missing or empty, you get an error that tells you how to fix it.
- **Coverage.** `src.testbench.targets.coverage_table()` shows, per product, the share of BOM mass that is priced and
  what is left out ("Other", unlabelled remainder).

### Switching steel back to Eurostat

In `config/assumptions.yaml` set:

```yaml
steel_price_source: eurostat
```

Steel is then priced with the series in `material_series_eurostat` (`S09` for carbon and stainless, `S10` for
galvanised), as before. The EUR/kg values at the base month are unchanged; only the series that drives the changes
switches.

## Competitor teardowns

`d.load_competitor_teardowns()` reads `data/raw/competitor_teardown_rm6_class.xlsx` (fallback:
`benchmark-modules-fonctionnels-rm6*.xlsx`). It reads every sheet whose name starts with `BOM `, parses the indented
material lines in column A (e.g. `  Copper — 8620 g · 11 pcs`), and returns one row per brand, product and material
with columns `brand, product, material, group, mass_kg, total_kg, broken`. Materials are mapped to groups (copper,
carbon_steel, aluminium, brass, silver, epoxy, plastics, electronics, sf6, packaging, other). `broken = True` when a
teardown totals less than 1 kg (the SGC sheet) — leave those out of any comparison.

## GSCPI fix

`signals.load_external()` reads `data/raw/external/gscpi.csv`, but `fetch_public` used to save only the NY Fed
workbook `gscpi_data.xlsx`. `fetch_public` now also writes `gscpi.csv` (column `GSCPI`, indexed by month start). It
finds the sheet layout when it runs: the sheet whose first column holds dates and second column holds numbers,
usually `GSCPI Monthly Data`. The NY Fed currently serves an old binary `.xls` file under that `.xlsx` URL, which
openpyxl cannot read; in that case `fetch_public` falls back to the CSV behind the NY Fed's interactive chart
(one column per release vintage) and keeps the latest vintage. If `gscpi.csv` is missing, `load_external()` warns and
skips that signal instead of crashing.

## Download options and reliability

FRED does not answer requests that send a custom or browser-like `User-Agent`, so `fetch_public` sends none (the
`requests` default works). Each request has a 20 s timeout and is retried twice with a 2 s pause. Sources can be
fetched one at a time, and a source whose output file already exists is skipped unless `--force` is passed:

```bash
uv run python -m src.external.fetch_public --only eurostat   # or fred | gscpi | all (default)
uv run python -m src.external.fetch_public --force           # download again even if the files exist
```

## Driver model (DriverRidge)

`src/testbench/driver_models.py` predicts each series from its own recent moves plus "market drivers" (`DRIVERS`),
and lets some indices catch up with a market twin (`MARKET_TWIN`). It now uses the new series. DriverRidge is not part
of `baseline_models()` or of the champion; `verify_drivers` only checks that it runs (official comparison: the
holdout selection 2020-23 → 2024-26).

Added entries:
- `L01` (MEPS carbon steel) ← `S04` iron ore, `ED02` gas, `E13` EU utilities: steel cost follows ore and energy.
- `L03` (MEPS galvanised) ← `S04`, `S05` zinc, `ED02`, `E13`: same as carbon steel plus the zinc coating.
- `L02` (CRU stainless 304) ← `S03` nickel, `ED02`, `E13`: the alloy surcharge follows nickel.
- `EP01` (epoxy) ← `ED01` Brent, `ED02` gas: epoxy is made from oil- and gas-based chemicals (benzene).
- `EL01` (electronics PPI) ← `D01` copper, `E13` utilities: boards use copper and energy.

Updated entries:
- `S09` (Eurostat steel) now also uses `L01`, and `S10` (German steel) now also uses `L03`: the Eurostat producer
  price follows the MEPS market price about a month later.
- `RM6_NE_IQI_433520` ← `D01`, `L01`, `S05`, `ED02`, `E13` and `RMAirSeT_IIQ_24kV_IEC` ← `L01`, `L02`, `D01`, `ED02`,
  `E13`: both BOM indices now include copper, because the v2 BOM lists copper in AirSeT too; steel enters through the
  MEPS/CRU prices.

`MARKET_TWIN` additions: `S09` → `L01` and `S10` → `L03` (the Eurostat steel index catching up with the MEPS market
price), next to the existing `S07` → `D01` and `S08` → `D02`.

## Commands

```bash
uv run python -m src.external.fetch_public   # public data: FRED, NY Fed GSCPI, Eurostat (incl. EL01-EL03)
uv run pytest -q                              # unit tests, synthetic data only (run in CI)
uv run python -m src.checks.verify_fixes      # local checks on the real data + one short backtest
uv run python -m src.checks.verify_drivers    # sanity check of DriverRidge on the new series (not model selection)
```

`verify_fixes` needs the files in `data/raw/` and a filled `config/assumptions.yaml` (copy
`config/assumptions.example.yaml` and fill the `FILL_LOCALLY` values). It writes to `data/outputs/business/` and
`data/outputs/checks/` (gitignored) and ends with `ALL CHECKS PASSED` when everything is fine.
