# Backtest hardening: design

Date: 2026-11-19. Branch: `backtest-hardening` (cut from `origin/julia/eda` at `1b83ff6`).
No confidential values in this file (it will be committed). Measured numbers live in `data/` (gitignored).

## 1. Intent

Schneider grades the challenge mainly on how forecasts are backtested and validated, and on the business decisions
they support. The CSV is also tracked against reality after the jury. The test bench on `julia/eda`
(`src/testbench`) already replays history origin by origin with a clean core. This work hardens it to a standard a
quant reviewer would accept: every number it prints can be defended as out-of-sample, leak-free, and statistically
qualified. It is also deliberate training in production-grade backtesting, so rigour beats speed.

## 2. Goals, non-goals, anti-goals

Goals
1. Proven absence of look-ahead: canaries and a truncation test that fail loudly if any model, feature or derived
   target reads the future, run in CI (synthetic data) and locally (real data).
2. One availability table (source, publication lag, revision window, licence) used by every mode.
3. Model selection and combination done walk-forward, so every graded origin (2022-01 on) is out-of-sample,
   including the choice of model (Strategy A).
4. A pooled steel-chain model using licensed CRU/MEPS prices as lagged inputs (Strategy B).
5. Statistics fit for overlapping horizons and small samples: ordered DM with HAC, Clark-West for nested models,
   pooled tests within material groups with Holm, block-bootstrap CIs, a measured minimum detectable effect (MDE).
6. A freeze protocol: pre-registration, tag, one graded run, outputs stamped with code, config and data hashes.
7. Fixes needed for a clean clone to run: reference BOM, committed config, declared dependencies, one data layout.

Non-goals
- Part 4 pricing-policy replay (fixed vs indexed vs hedged quotes), decision tool, slides.
- New external data sources beyond what `julia/eda` already fetches. First-release vintages (ALFRED) are out of
  scope: they would only cover S12/S13, the secondary US proxies, and Eurostat keeps no vintages.
- Deep learning, VARs, regime-switching models, fine-tuning pretrained models.
- Rewriting Julia's interface or baselines. Existing models keep working unchanged.

Anti-goals (failure even if the code runs)
- Any reported score from a run that has not passed the canaries.
- Any selection, tuning or screening decision informed by forecasts whose target month is after the origin being
  scored.
- Licensed values (CRU, MEPS, PitchBook) or challenge data in git, in committed notebook outputs, in CI logs, or in
  the submission CSV.
- A significance claim per series at h = 9 (about 8 effective observations).

## 3. Constraints

- Data is confidential and stays under `data/` (gitignored). CI uses synthetic data only.
- Monthly panel 2016-11 to 2026-09 (119 months); `julia/eda` extends some series to 1992 from identical public
  sources (current vintages only).
- Official origin `2026-09` = "last data used" (Schneider's definition). Horizons 3, 6, 9.
- 16 GB Windows laptop, uv, Python 3.12. A full backtest must stay under about 15 minutes.
- Group repo. Julia owns `julia/eda`; this branch merges back by PR when the user decides. Nothing is committed or
  pushed without the user's word.
- Model contract (unchanged): `forecast(y, steps, exog) -> DataFrame[p10, p50, p90]` indexed by step.

## 4. Approaches considered

| Decision | Options | Chosen | Why |
|---|---|---|---|
| Starting point | harden Julia's bench / rebuild / switch to pricing replay | **harden** | about 80% exists and the core (truncate-then-forecast, assert, perturbation audit) is sound; rebuilding duplicates a teammate's part |
| Model selection | walk-forward selection / fixed 2024 split | **walk-forward** | the 2024-26 window has already informed choices (half-life, champion rule) and holds about 2-3 independent windows at h = 9 |
| Selection output | pick-best / loss-weighted blend / equal median | **all three, plus Strategy B as a candidate** | they are cheap meta-models scored like any model; comparing them answers "do picks line up, and is a blend good enough" |
| CRU/MEPS | exclude / input only | **input only, behind a switch** | we have access; strongest known driver of the EU steel indices; never written to outputs |

## 5. Design

### 5.1 Data layer
- Layout: keep Julia's `data/raw/` (panel CSVs in `data/raw/panel/`, workbooks in `data/raw/`, downloads in
  `data/raw/external/`). A one-off script copies from the user's current `data/Data/` layout; nothing is deleted.
- BOM: read only the reference workbook `RM_Prices_XHEC_Challenge.xlsx` (Schneider-confirmed 2026-11-18). Remove the
  `csv` default; keep the deprecated CSV loadable only behind an explicit flag for comparison.
- Config (as set up by Julia in `f5c2685`): `config/assumptions.example.yaml` is committed with `FILL_LOCALLY`
  placeholders; each member copies it to `config/assumptions.yaml` (gitignored, holds licensed MEPS/CRU levels) and
  fills it. New keys added by this work (CRU/MEPS switch, availability file path) go into the example with
  non-licensed defaults.
- Epoxy: load with the May 2026 epichlorohydrin typo corrected (x10) and documented as a data correction. Resin for
  that month is left as delivered and flagged; epoxy is an input, and an optional target only if the team adds it.
- CRU/MEPS: loader that turns the steel workbook into monthly EUR/t columns (prefix `L_`, e.g. MEPS CR coil DE/FR,
  MEPS HDG DE, CRU stainless 304 DE base + alloy). Added to the panel only when `use_licensed: true`.
- Declare `pyyaml` (and `scipy` if used directly) in `pyproject.toml`.

### 5.2 Availability table (single source of truth)
`config/availability.yaml`, one row per series: provider, how the month is formed (monthly average, end of month,
stamp), publication lag in months for realtime mode, revision window, whether vintages exist, licence. `vintage.py`
and `signals.EXTERNAL_LAGS` read from it; the two existing hard-coded lag dicts are removed. A test asserts every panel
column has a row. Lags are set from the providers' release calendars, not from the download date.

### 5.3 Truth and vintages
- Truth is the latest vintage for every series (documented as a limit). Sensitivity to the revisable last months is
  shown by the realtime mode, which masks them at every origin.

### 5.4 Leakage canaries and audits
Run in CI on synthetic panels; the real-data versions run with `uv run python -m src.testbench.audit`.
- (a) Planted-leak model: reads `y(t+h)` plus noise. Must show absurd skill, and the perturbation audit must flag it.
- (b) Shuffled target: block-shuffled target series. Every model's skill vs naive must collapse to about zero.
- (c) Truncation identity: forecasts from a panel physically cut at the origin equal forecasts from the full panel
  passed through `information_set`, bit for bit, for every model, meta-model, feature builder and derived BOM index.
  Extends the existing perturbation audit to the external and licensed columns.
- (d) Planted off-by-one: a feature deliberately shifted by one month must fail (c). Proves the check fires.
- A graded run refuses to start unless (a) to (d) pass on the current code hash.

### 5.5 Strategy A: walk-forward selection and combination
Two passes.
1. Base pass: every candidate model forecasts at every origin (as today). Stored long format.
2. Meta pass: at origin t, for each (series, horizon), use only base rows whose **target month <= t** (realised at t).
   Score candidates on a trailing window of those rows (pinball loss averaged over P10/P50/P90). Build:
   - `wf_best`: the candidate with the lowest trailing loss;
   - `wf_blend`: quantile-wise weighted average of candidates, weights from trailing loss (inverse loss or softmax,
     temperature fixed before grading), quantiles re-sorted;
   - `wf_median`: quantile-wise median of a shortlist fixed in the pre-registration.
   If fewer than a minimum number of realised rows exist, fall back to `naive`.
- Candidates include Julia's baselines, `naive_vol` at half-lives 3, 6 and 12 (so her earlier half-life choice becomes
  a walk-forward choice), `driver_ridge`, Strategy B, and the champion with walk-forward screening (below).
- Walk-forward screening: the signal screen in `src/external/screen.py` is re-run at each origin (or each January, if
  cost requires) using only targets realised by that origin. The global pre-2024 screen is kept for exploration only.
- Reported diagnostics: pick switch rate per (series, horizon), share of origins each model is picked, weight paths,
  and whether the three meta-models agree.

### 5.6 Strategy B: pooled steel-chain model
- Targets: S09 and S10 (EU steel indices) pooled; S12 (US) handled with the same machinery and US drivers
  (`X_US_STEEL_SCRAP`, `X_US_PPI_STEEL`) as a separate pool.
- Direct h-step regression of the h-month log change on: own 1- and 3-month log change, and CRU/MEPS 1- and 3-month
  log changes lagged one month (EU pool). One coefficient set per pool and horizon, no intercept, shrinkage toward
  zero so it falls back to naive without signal.
- Bands: conformal from the model's own walk-forward residuals, volatility-scaled like `naive_vol`.
- Falls back to naive when `use_licensed: false` for the EU pool.

### 5.7 Statistics
- DM: loss differentials sorted by origin explicitly; Newey-West with `step - 1` lags (step = horizon plus
  publication lag in realtime mode); Harvey-Leybourne-Newbold correction; t(n-1).
- Clark-West for nested comparisons against naive (point forecasts).
- Group-pooled tests: average loss differential across series inside each material group (metals, steel, plastics,
  BOM indices) per origin, then HAC on that series. Holm across groups x horizons.
- Moving-block bootstrap (block length = horizon) for CIs on relative MAE, relative pinball loss and coverage.
- MDE: run placebo models (naive plus independent noise of increasing size) through the full pipeline; report the
  smallest relative-loss gain detectable at 5% per group and horizon.
- Loss sanity test: perfect < naive < anti-correlated forecast, asserted on synthetic data.

### 5.8 Dev window, freeze and graded run
- Dev window: origins 2010-01 to 2021-12, scored only on forecasts whose target month is before 2022-01 (purge, so no
  dev score touches the graded period). All development, debugging, tuning, MDE placebos and runtime checks use the
  dev window only. Scores for graded origins are never printed before the graded run (the CLI refuses unless
  `--graded` with a tagged clean tree).
- Graded window: origins 2022-01 to 2026-09 (targets up to 2026-09), scored once. The walk-forward meta pass still
  uses every realised row before each origin, including dev-window rows.
- Submitted model fixed in advance: `wf_blend`. `wf_best`, `wf_median`, Strategy B and the baselines are reported
  next to it, never used to switch the submission after the graded run.
- `docs/backtest/preregistration.md` committed before grading: candidate set, meta-model rules and parameters,
  shortlist, metrics, PASS rules (group-pooled, Holm, both modes), MDE table, and prior looks declared (Julia's
  2024-26 inspections, our EDA correlations).
- Graded run command refuses unless HEAD is tagged, the tree is clean and canaries passed. Every output folder gets a
  `manifest.json`: git hash, config hash, input file hashes, mode, origins, runtime, package versions.
- Outputs stay in `data/outputs/` (gitignored). Deterministic: seeded RNG, no wall-clock dependence.

### 5.9 Submission
`uv run python -m src.testbench.run --submit` produces `<team>_forecast_202609.csv` at origin 2026-09 from the
pre-registered final meta-model, validated by `validate_submission`.

## 6. Decisions settled after the builder check (designer's calls, marked for review)

1. **BOM index weights must be causal.** `make_bom_deriver` currently calibrates the level constant `k_i` with the
   2026-09 panel value, so past origins' weights carry 2026-09 information and a physically cut panel cannot rebuild
   them. New rule: at origin t, steel EUR/kg comes from the CRU/MEPS level known at t (licensed switch on); plastics
   and the steel fallback (switch off) use config anchors dated at the panel's first month (2016-11), so `k_i` only
   uses data every origin has seen. The CSV row at origin 2026-09 is unchanged in definition (index = 100 at origin).
2. **Config file.** Settled by Julia (`f5c2685`): committed template `config/assumptions.example.yaml`, local filled
   `config/assumptions.yaml` (gitignored, licensed values). The causal BOM weights of point 1 read CRU/MEPS levels
   from `data/` at each origin, so the 2026-09 EUR/kg values in the local config are used only at the live origin.
3. **Panel and warm-up.** The base pass runs on the history extended back to 1992 (Julia's `extend_history`), with
   base origins from 2010-01 (licensed inputs exist from 2008/2012; earlier origins use the switch-off path). The
   meta pass therefore has realised rows long before the graded window (2022-01 onward, see 5.8).
4. **Submission mode mirrors the live ragged edge.** For each input series, the gap between the live origin
   (2026-09) and its last available month in the delivered files is applied at every backtest origin (e.g. a series
   ending 2026-07 is masked by 2 months; a series running past September is cut at the origin). Targets keep
   Schneider's convention (known up to the origin). This makes every backtest origin see what the live run sees.
5. **"Realised at t" in the meta pass** means target month <= t in submission mode and target month <= t - lag(series)
   in realtime mode, lags from `config/availability.yaml`. Meta-pass scoring uses the latest vintage.
6. **No vintages.** Dropped after review: low value (two secondary series), see non-goals.
7. **Target ids.** CSV `target_id`s are the S/D series ids and `RM6_NE_IQI_433520` (slide example). AirSeT uses
   `RMAirSeT_IIQ_24kV_IEC` until Schneider confirms the spelling (added to the question list). Licensed `L_` series
   are inputs only and can never be targets; the loader uses the prefix `LIC_` to avoid colliding with the empty
   official `L01`-`L08` columns.

## 7. Left to the builder
- Trailing window length and minimum realised rows for the meta pass (to be fixed in the pre-registration).
- Inverse-loss vs softmax weighting and its temperature (fixed in the pre-registration).
- Shrinkage strength for Strategy B (fixed a priori or chosen walk-forward, never on the full sample).
- Whether the walk-forward screen runs every origin or every January (cost).
- Test file organisation, module boundaries, CLI flag names.
