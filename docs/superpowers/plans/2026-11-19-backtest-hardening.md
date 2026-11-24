# Backtest Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Julia's test bench into a leak-proof, walk-forward, statistically qualified backtest with a frozen graded run and a one-command submission.

**Architecture:** One availability table drives every information set (realtime lags and the live ragged edge). A base pass runs every candidate at every origin from 2010-01 on the 1992-extended panel. A meta pass (Strategy A) then selects and blends candidates using only rows realised by each origin. Strategy B (pooled steel chain) is one of those candidates. Canaries gate every graded number, and a protocol module enforces dev/graded separation, the tag and the manifest.

**Tech Stack:** Python 3.12, uv, pandas, numpy, scipy, pyyaml, openpyxl, pytest. No new third-party packages.

**Spec:** `docs/superpowers/specs/2026-11-19-backtest-hardening-design.md` (approved 2026-11-19)

## Global Constraints

- Never commit `data/`, `config/assumptions.yaml`, notebook outputs, or any challenge, CRU, MEPS or PitchBook value. This includes tracked docs, test fixtures and CI logs. Synthetic numbers only in tests.
- Licensed series use the prefix `LIC_`. They are inputs only. They can never be a `target_id`, a CSV column, or a printed table row.
- Commits: nothing is committed, tagged or pushed without the user's word in this session (group repo). No Claude attribution lines in commits (user's manual: group repos get none). At the start of execution, ask once whether the per-task commit steps are approved.
- Run everything with `uv run python -X utf8 ...` and tests with `uv run pytest -q`. CI runs `uv sync --frozen` and then `uv run pytest -q tests` on synthetic data, so every test touching `data/` must skip when `data/raw` is absent.
- Model contract is unchanged: `forecast(y, steps, exog) -> DataFrame[p10, p50, p90]`, indexed by step, `p10 <= p50 <= p90`.
- Origins and horizons:
  - Official origin `2026-09`; horizons 3, 6, 9.
  - Dev origins `2010-01` to `2021-12`, scored only where `target_month < 2022-01`.
  - Graded origins `2022-01` to `2026-09`.
  - Base pass origins start at `2010-01`.
- Submitted model: `wf_blend`. It is never switched after the graded run.
- Target ids: S/D series ids, `RM6_NE_IQI_433520`, `RMAirSeT_IIQ_24kV_IEC`.
- Full backtest (one mode, all origins) must stay under about 15 minutes on the 16 GB laptop.
- Deterministic: every RNG is seeded, and nothing depends on the wall clock except the runtime fields in the manifest.

## Review Focus

1. **A local `config/assumptions.yaml` copied before this work** has no `backtest:` or `bom_anchor:` section. The run should stop with a message naming the missing section and the example file, not a `KeyError`. The test is in Task 5.
2. **Input series whose delivered file ends before the live origin** (CRU stainless ends one month before 2026-09; E20/E21/ED06 end two months before). At every backtest origin, submission mode must mask those same months. The test is in Task 3.
3. **Origins before a licensed series starts.** MEPS starts 2012, so origins 2010-01 to 2012-10 have no MEPS. The BOM deriver and Strategy B must fall back cleanly, with no NaN forecast and no exception. Tests are in Tasks 5 and 9.
4. **Meta pass at an origin where some candidates have fewer realised rows than the minimum.** This covers early origins and `steel_chain`, which runs only on three series. Those candidates must be dropped from that origin's weights. If none qualify, the origin falls back to `naive`. The test is in Task 8.
5. **Realtime mode for the derived BOM indices.** "Realised at t" must use the BOM rows' own lag from the availability table. A BOM row with no availability row must raise an error rather than default to lag 0. Tests are in Tasks 3 and 8.

---

## File structure

| File | Responsibility |
|---|---|
| `config/availability.yaml` (new, committed) | one row per series: provider, formation, realtime lag, revision window, vintages, licence |
| `config/assumptions.example.yaml` (modify) | add `backtest:` and `bom_anchor:` sections with non-licensed defaults / `FILL_LOCALLY` |
| `src/data.py` (modify) | BOM loader deprecation flag, epoxy typo correction, drop `PUBLICATION_LAG` |
| `src/licensed.py` (new) | parse the CRU/MEPS workbook into monthly `LIC_` columns |
| `src/testbench/availability.py` (new) | load availability, realtime lags, ragged-edge lags, coverage check |
| `src/testbench/vintage.py` (modify) | `information_set` takes lags in both modes; hard-coded lag dict removed |
| `src/testbench/panel.py` (new) | backtest config loader and the one panel builder |
| `src/testbench/targets.py` (modify) | causal BOM deriver (no panel value at a date after the origin) |
| `src/testbench/backtest.py` (modify) | `lags` passthrough, per-model `targets` filter |
| `src/testbench/canaries.py` (new) | canaries (a) to (d), meta-pass identity, report file |
| `src/testbench/meta.py` (new) | Strategy A meta pass and diagnostics |
| `src/testbench/steel_chain.py` (new) | Strategy B |
| `src/external/screen.py`, `src/external/models.py` (modify) | walk-forward screen, `ChampionWF` |
| `src/testbench/candidates.py` (new) | candidate set and `wf_median` shortlist (pre-registered) |
| `src/testbench/stats.py` (new) | ordered DM, Clark-West, pooled DM + Holm, block bootstrap, MDE placebos |
| `src/testbench/protocol.py` (new) | dev/graded windows, grade guard, manifest |
| `src/testbench/run.py`, `src/testbench/audit.py` (modify) | CLI `--dev / --canaries / --graded / --submit` |
| `docs/backtest/preregistration.md` (new) | frozen rules before grading |
| `docs/testbench.md` (modify) | how to run, what each output means |

Tests: one file per new module under `tests/`. Real-data tests are decorated with `needs_data = pytest.mark.skipif(not (RAW / "panel").exists(), reason="no challenge data")`, defined in `tests/conftest.py`.

---

### Task 1: Dependencies, data-layer fixes, test scaffolding

**Files:**
- Create: `src/setup_data.py`
- Modify: `pyproject.toml`, `uv.lock`, `src/data.py:27-33,73-87,142-147`, `notebooks/build_01_eda.py:130,135`, `notebooks/01_eda_julia.ipynb` (same two source lines)
- Create: `tests/conftest.py`, `tests/test_data_layer.py`

**Interfaces:**
- Produces: `load_bom(version="root", rm6_carbon_share=0.45, allow_deprecated=False)`, `correct_epoxy(df) -> pd.DataFrame`, `load_epoxy(correct=True) -> pd.DataFrame` with columns renamed to `EPX_BENZENE, EPX_ACETONE, EPX_ECH, EPX_OTHER, EPX_RESIN`, `EPOXY_CORRECTIONS: list[dict]`, fixture `needs_data`.

- [ ] **Step 1: Write the failing tests** in `tests/test_data_layer.py`

```python
def test_deprecated_bom_needs_flag():
    with pytest.raises(ValueError, match="deprecated"):
        d.load_bom("csv")

def test_epoxy_ech_typo_corrected_x10_and_resin_flagged():
    idx = pd.date_range("2026-03-01", periods=4, freq="MS")
    df = pd.DataFrame({"EPX_ECH": [2500.0, 2480.0, 257.0, 2600.0],
                       "EPX_RESIN": [4900.0, 4930.0, 3600.0, 4980.0]}, index=idx)
    out = d.correct_epoxy(df)
    assert out.at[pd.Timestamp("2026-05-01"), "EPX_ECH"] == 2570.0
    assert out.at[pd.Timestamp("2026-05-01"), "EPX_RESIN"] == 3600.0   # left as delivered
    assert out.attrs["corrections"][0]["month"] == "2026-05"
    assert df.at[pd.Timestamp("2026-05-01"), "EPX_ECH"] == 257.0       # input not mutated

def test_no_publication_lag_table_left_in_data():
    assert not hasattr(d, "PUBLICATION_LAG")
```

- [ ] **Step 2: Run** `uv run pytest -q tests/test_data_layer.py`. Expected: 3 FAIL.
- [ ] **Step 3: Implement.**
  - `load_bom` raises `ValueError("BOM csv is deprecated (see data/Data/BOM_SOURCE_OF_TRUTH.md); pass allow_deprecated=True for sensitivity only")` when `version == "csv"` and the flag is off.
  - `correct_epoxy` multiplies `EPX_ECH` by 10 in month `2026-05` only, copies the frame, and records `{"month": "2026-05", "column": "EPX_ECH", "factor": 10, "reason": "decimal typo, resin left as delivered and flagged"}` in `attrs["corrections"]`. `EPOXY_CORRECTIONS` holds the same list.
  - `load_epoxy` renames the five delivered columns in file order to `EPX_*`, applies `correct_epoxy` when `correct=True`, and keeps the month-start index it already builds.
  - Delete `PUBLICATION_LAG` (it has no users).
  - Update the two `load_bom("csv")` calls in the notebook build script and notebook source to pass `allow_deprecated=True`.
  - Add `pyyaml`, `scipy` and `requests` to `[project].dependencies`, then run `uv lock`.
  - `tests/conftest.py` defines `needs_data` as in the File structure note.
  - Spec 5.1 one-off copy: create `src/setup_data.py`. Running it with `uv run python -m src.setup_data` copies the workbooks from `data/Data/` into `data/raw/`, and the panel CSVs from `data/Data/data_zip/**/` into `data/raw/panel/`. It only copies files missing at the destination, never overwrites or deletes, and prints what it copied. This machine already has `data/raw/` filled, so running it should print "nothing to copy".
- [ ] **Step 4: Run** `uv run pytest -q tests`. Expected: all pass, including the old `tests/test_testbench.py`.
- [ ] **Step 5: Commit** (after the user's go-ahead): `git add pyproject.toml uv.lock src/data.py notebooks/build_01_eda.py notebooks/01_eda_julia.ipynb tests/conftest.py tests/test_data_layer.py`, message `Data layer: deprecate csv BOM, correct epoxy ECH typo, declare deps`. Check `git diff --cached notebooks/01_eda_julia.ipynb` first: it must contain no outputs.

---

### Task 2: Licensed CRU/MEPS loader

**Files:**
- Create: `src/licensed.py`, `tests/test_licensed.py`

**Interfaces:**
- Produces: `LICENSED_FILE = RAW / "Copia de Europe Steel INDEXES.xlsx"`, `LIC_COLUMNS = ("LIC_MEPS_CR_DE", "LIC_MEPS_HDG_DE", "LIC_MEPS_CR_FR", "LIC_CRU_SS304_DE")`, and `load_licensed(path=LICENSED_FILE) -> pd.DataFrame`. The return is EUR/t, month-start index, columns exactly `LIC_COLUMNS`. `LIC_CRU_SS304_DE` = BASE + Alloy.

Workbook layout (structure only):
- Sheet `MEPS`:
  - row 1 holds country labels in the first column of each block (forward-fill across the block);
  - row 5 holds product labels (`HR coil`, `CR coil`, `HDG coil`, `EG coil`), with `Month` in column A;
  - data starts at row 6.
- Sheet `CRU Stainless`:
  - row 1 holds the country (forward-fill), row 2 the grade (`Grade 304`, forward-fill), row 5 `BASE` / `Alloy`;
  - data starts at row 6, dated on the 15th.
- Locate columns by these labels, never by fixed letters.

- [ ] **Step 1: Write the failing tests.** A fixture builds a synthetic workbook with openpyxl in `tmp_path`:
  - It reproduces both layouts with Germany and France blocks.
  - It has 30 months from 2012-11, with values `600 + i` (CR DE), `700 + i` (HDG DE), `610 + i` (CR FR).
  - Stainless BASE `2000 + i` and Alloy `1000 + i` start 2012-11-15.
  - It leaves one blank cell and one `"-"` cell in the stainless block.

```python
def test_columns_units_and_month_start(wb_path):
    df = load_licensed(wb_path)
    assert list(df.columns) == list(LIC_COLUMNS)
    assert df.index.min() == pd.Timestamp("2012-11-01") and (df.index.day == 1).all()
    assert df.at[pd.Timestamp("2012-12-01"), "LIC_MEPS_CR_DE"] == 601
    assert df.at[pd.Timestamp("2012-12-01"), "LIC_CRU_SS304_DE"] == 2001 + 1001

def test_dash_and_blank_become_nan_not_zero(wb_path):
    df = load_licensed(wb_path)
    assert df["LIC_CRU_SS304_DE"].isna().sum() == 2 and (df.fillna(1) > 0).all().all()

def test_missing_label_raises(tmp_path):  # a workbook whose MEPS sheet lacks "HDG coil"
    with pytest.raises(ValueError, match="HDG coil"):
        load_licensed(make_wb(tmp_path, drop="HDG coil"))
```

- [ ] **Step 2: Run** `uv run pytest -q tests/test_licensed.py`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement `load_licensed`.**
  - Use openpyxl, `read_only=True, data_only=True`.
  - Convert non-numeric cells (`"-"`, `None`) to NaN.
  - Map the date to the month start with `to_period("M").to_timestamp()`.
  - If a required label is missing, raise `ValueError` naming it.
- [ ] **Step 4: Run** the tests. Expected: PASS. Then check the real file locally:

  ```
  uv run python -X utf8 -c "from src.licensed import load_licensed as f; x=f(); print(x.notna().sum().to_dict(), x.index.min(), x.index.max())"
  ```

  Expected: four non-zero counts; stainless ends one month before the MEPS columns. Do not print values.
- [ ] **Step 5: Commit** `src/licensed.py tests/test_licensed.py`: `Add licensed CRU/MEPS loader (LIC_ inputs only)`.

---

### Task 3: Availability table, ragged edge, information set

**Files:**
- Create: `config/availability.yaml`, `src/testbench/availability.py`, `tests/test_availability.py`
- Modify: `src/testbench/vintage.py:16-55`, `src/external/signals.py:42-45`, `tests/test_testbench.py:47-70`

**Interfaces:**
- Produces:
  - `AVAILABILITY = ROOT / "config" / "availability.yaml"`.
  - `load_availability(path=AVAILABILITY) -> pd.DataFrame`: index `series_id`; columns `provider, formation, realtime_lag, revision_months, vintages, licence`.
  - `realtime_lags(av=None) -> dict[str, int]`.
  - `ragged_edge_lags(panel, live_origin, exempt) -> dict[str, int]`.
  - `check_coverage(columns, av=None) -> None`: raises `KeyError` listing the missing ids.
  - `information_set(panel, origin, mode="submission", lags=None)`: `lags` masks the last `lag` months of each column in either mode. In realtime mode `lags=None` means `realtime_lags()`; in submission mode `lags=None` means no mask.
- Removed: `vintage.REALTIME_LAG_MONTHS`, `signals.EXTERNAL_LAGS`.

YAML row schema: `S09: {provider: Eurostat, formation: monthly index, realtime_lag: 2, revision_months: 2, vintages: false, licence: CC BY 4.0}`. Lags follow the providers' release calendars, read as "a buyer deciding mid-month t sees reference month t − lag":

| Series | Provider | realtime_lag | revision_months |
|---|---|---|---|
| S01–S05, E01–E03, E05–E08, E22, X_IMF_METALS, X_IMF_ALLCOMMOD | IMF via FRED, monthly average | 1 | 0 |
| S06, E23, E24 | ECB monthly average FX | 1 | 0 |
| D01, D02 | derived S01/S02 ÷ S06 | 1 | 0 |
| S07–S11, E13–E16 | Eurostat PPI | 2 | 2 |
| E20, E21, ED06 | Eurostat trade volumes | 3 | 2 |
| S12, S13, E17, E18, X_US_PPI_*, X_US_STEEL_SCRAP, X_FREIGHT | BLS PPI | 1 | 4 |
| E04, E19 | Bank of Japan CGPI | 1 | 1 |
| E09–E12 | Ember / JEPX monthly power | 1 | 0 |
| ED01–ED05 | derived from FRED/ECB | 1 | 0 |
| X_US_INDPRO, X_US_IP_METALS, X_US_CAPUTIL, X_US_HOUSING | Fed G.17 / Census | 1 | 3 |
| X_US_NEW_ORDERS | Census M3 | 2 | 2 |
| X_VIX, X_OVX, X_USD_BROAD, X_US_BREAKEVEN, X_US_10Y | daily market, monthly average | 1 | 0 |
| X_GLOBAL_EPU | EPU | 2 | 0 |
| X_GSCPI | NY Fed | 1 | 1 |
| X_CU_TERM, X_AL_TERM, X_CU_STOCKS, X_AL_STOCKS | LME file (challenge) | 1 | 0 |
| LIC_* | CRU/MEPS (licensed) | 1 | 0 |
| EPX_* | Schneider epoxy file | 1 | 0 |
| RM6_NE_IQI_433520, RMAirSeT_IIQ_24kV_IEC | derived BOM index | 2 | 2 |

(Daily-market averages move from lag 0 to lag 1 here, because the month's average is incomplete at a mid-month decision. Note this in `docs/testbench.md`.)

- [ ] **Step 1: Write the failing tests.**

```python
def test_every_known_column_has_a_row():
    from src.external.signals import EXTERNAL
    cols = (d.RM_TARGETS + ["S06"] + [f"E{i:02d}" for i in range(1, 25)] + [f"ED0{i}" for i in range(1, 7)]
            + list(EXTERNAL.values()) + ["X_GSCPI", "X_CU_TERM", "X_AL_TERM", "X_CU_STOCKS", "X_AL_STOCKS"]
            + list(LIC_COLUMNS) + ["EPX_BENZENE", "EPX_ACETONE", "EPX_ECH", "EPX_OTHER", "EPX_RESIN"]
            + ["RM6_NE_IQI_433520", "RMAirSeT_IIQ_24kV_IEC"])
    check_coverage(cols)

def test_missing_row_raises():
    with pytest.raises(KeyError, match="NEW_SERIES"):
        check_coverage(["S01", "NEW_SERIES"])

def test_ragged_edge_mirrors_live_gap(panel):   # B ends 2 months before the last row, A is a target
    edge = ragged_edge_lags(panel, live_origin=panel.index[-1], exempt={"A"})
    assert edge == {"B": 2}
    info = information_set(panel, "2019-06-01", lags=edge)
    assert info["B"].last_valid_index() == pd.Timestamp("2019-04-01")
    assert info["A"].last_valid_index() == pd.Timestamp("2019-06-01")

def test_series_running_past_live_origin_is_cut_not_negative(panel):
    edge = ragged_edge_lags(panel, live_origin=panel.index[-5], exempt=set())
    assert min(edge.values()) == 0

@needs_data
def test_real_panel_fully_covered():
    from src.testbench.panel import build_full_panel
    check_coverage(build_full_panel(use_licensed=True).columns)
```

  Then edit `test_perturbing_the_future_changes_nothing` to pass `lags={"A": 2}` through `run_backtest(..., lags=lags)` (Task 4 adds the parameter) instead of patching the module dict.
- [ ] **Step 2: Run** `uv run pytest -q tests/test_availability.py`. Expected: FAIL. (`test_real_panel_fully_covered` fails or skips until Task 4.)
- [ ] **Step 3: Implement** `availability.py` and the YAML.
  - `ragged_edge_lags` returns `max(0, months_between(last_valid, live_origin))` for every non-exempt column that has data, and leaves out all-NaN columns.
  - Rewrite `information_set`.
  - In `signals.py`, delete `EXTERNAL_LAGS` and fix any import of it (grep).
- [ ] **Step 4: Run** `uv run pytest -q tests`. Expected: all pass except the two that wait on Task 4 (`run_backtest(..., lags=)` and the real-panel test). Mark both `xfail(strict=True, reason="Task 4")` and remove the mark in Task 4.
- [ ] **Step 5: Commit** `config/availability.yaml src/testbench/availability.py src/testbench/vintage.py src/external/signals.py tests/`: `Single availability table drives realtime lags and live ragged edge`.

---

### Task 4: Panel builder, backtest config, run_backtest passthrough

**Files:**
- Create: `src/testbench/panel.py`, `tests/test_panel.py`
- Modify: `src/testbench/backtest.py:22-71`, `src/external/signals.py:111-118`, `config/assumptions.example.yaml` (append section), local `config/assumptions.yaml` (same section, never committed)

**Interfaces:**
- Consumes: `load_licensed`, `load_epoxy`, `extend_history`, `load_external`, `information_set(..., lags)`.
- Produces:
  - `load_backtest_config(assumptions: dict | None = None) -> dict`, with keys `use_licensed: bool`, `live_origin: str`, `team_id: str`, `availability_file: str`.
  - `build_full_panel(use_licensed: bool, long: bool = True, external: bool = True) -> pd.DataFrame`:
    - challenge rm + energy, with all-NaN columns (L01–L08) dropped;
    - `extend_history` applied when `long`;
    - `load_external()` added when `external`;
    - `EPX_*` joined;
    - `LIC_*` joined only when `use_licensed`.
  - `run_backtest(..., lags: dict[str, int] | None = None)` passes `lags` to `information_set`.
  - A model with attribute `targets: tuple[str, ...]` is run only on those series.
  - `signals.build_panel(long, external)` returns `build_full_panel(use_licensed=False, long=long, external=external)`.

Example YAML section (committed):

```yaml
backtest:
  use_licensed: false            # true only on machines with the licensed CRU/MEPS workbook
  live_origin: "2026-09"
  team_id: "team_TBD"
  availability_file: config/availability.yaml
```

- [ ] **Step 1: Write the failing tests.**
  - `test_missing_backtest_section_message`: passing a dict without `backtest` raises `KeyError` whose message contains `"backtest"` and `"assumptions.example.yaml"`.
  - `test_model_targets_filter`: a Spy model with `targets = ("A",)` produces rows only for `A`.
  - `test_lags_passthrough`: `run_backtest(panel, ["A"], [Spy()], origins=[o], lags={"A": 2})` gives `last_obs == o - 2 months`.
  - `@needs_data test_build_full_panel`:
    - no column starts with `L0`;
    - the `LIC_` columns are present if and only if `use_licensed`;
    - `EPX_ECH` is present;
    - the index starts at `1992-01-01`.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.** Before the real-data test, fetch the public series once:

  ```
  uv run python -m src.external.fetch_public
  ```

  This sends only public series codes and writes to `data/raw/external/`. Then add the `backtest:` section to the local `config/assumptions.yaml`, with `use_licensed: true`.
- [ ] **Step 4: Run** `uv run pytest -q tests`. Expected: all pass. Remove the Task 3 xfail marks.
- [ ] **Step 5: Commit** `src/testbench/panel.py src/testbench/backtest.py src/external/signals.py config/assumptions.example.yaml tests/`: `One panel builder; lags passthrough; per-model target filter`.

---

### Task 5: Causal BOM index deriver

**Files:**
- Modify: `src/testbench/targets.py:22-128`, `src/testbench/run.py:37-39` (`bom_deriver`), `src/testbench/audit.py:32`, `src/external/screen.py:30-31`, `src/external/run_champion.py:32`, `config/assumptions.example.yaml`, local `config/assumptions.yaml`
- Test: `tests/test_bom_causal.py`

**Interfaces:**
- Consumes: `load_backtest_config`, `LIC_COLUMNS`.
- Produces:
  - `make_bom_deriver(assumptions=None, use_licensed=False)`. The `panel_full` parameter is removed: it was the leak.
  - The return is `derive(values_df, bases_df, origin) -> DataFrame`, with `derive.weights_at(bases_df, origin)` as before.
  - `LICENSED_STEEL = {"carbon_steel": "LIC_MEPS_CR_DE", "galvanised_steel": "LIC_MEPS_HDG_DE", "stainless_steel": "LIC_CRU_SS304_DE"}`.
  - `anchor_values(panel) -> dict`: a helper that prints the numbers to paste into the local config; its output is never written to tracked files.
  - `run.bom_deriver(cfg: dict)` replaces `bom_deriver(panel)`. Update its three callers.

Rule (spec 6.1):
- EUR/kg at origin t:
  - copper and aluminium: `D01 / D02 ÷ 1000`, last known value ≤ t;
  - steel with `use_licensed` and its `LIC_` column known ≤ t: `LIC ÷ 1000`, last known;
  - otherwise (plastics, or steel without licensed data): `k_i × index_i(last known ≤ t)`, where `k_i = anchor eur_per_kg ÷ anchor series_value` and both are **config constants**.
- The deriver therefore never reads a panel value dated after the origin.
- Declared limit, to be written into the preregistration: the anchor constants are dated 2016-11. For origins before 2016-11 they are a level calibration fixed later in time. They apply identically to forecast and truth and carry no price-move information. The graded window is unaffected.

Example YAML (committed):

```yaml
bom_anchor:
  month: "2016-11"
  carbon_steel:     {eur_per_kg: null, series_value: null}   # FILL_LOCALLY: MEPS DE CR /1000 and S09 at month
  stainless_steel:  {eur_per_kg: null, series_value: null}   # FILL_LOCALLY: CRU DE 304 base+alloy /1000, S09
  galvanised_steel: {eur_per_kg: null, series_value: null}   # FILL_LOCALLY: MEPS DE HDG /1000, S10
  plastics:         {eur_per_kg: 2.50, series_value: null}   # ASSUMED level; FILL_LOCALLY S11 at month
```

- [ ] **Step 1: Write the failing tests.** All of them use a synthetic panel (A as copper-like EUR/t, S as an index, LIC as EUR/t from month 40) and a synthetic assumptions dict.

```python
def test_deriver_identity_on_cut_panel(toy):            # physically cut == information_set of full
    der = make_bom_deriver(toy.assumptions, use_licensed=True)
    for o in toy.panel.index[[20, 45, 80]]:
        cut = toy.panel.loc[:o]
        pd.testing.assert_frame_equal(der(cut, cut, o), der(toy.panel, toy.panel.loc[:o], o).loc[:o])

def test_index_is_100_at_origin(toy):
    o = toy.panel.index[60]
    der = make_bom_deriver(toy.assumptions, use_licensed=True)
    assert (der(toy.panel.loc[:o], toy.panel.loc[:o], o).loc[o] == 100).all()

def test_licensed_level_used_when_known_else_anchor(toy):
    der = make_bom_deriver(toy.assumptions, use_licensed=True)
    w_late = der.weights_at(toy.panel, toy.panel.index[60]).set_index("material")
    w_early = der.weights_at(toy.panel, toy.panel.index[10]).set_index("material")
    assert w_late.at["carbon_steel", "eur_per_kg"] == toy.panel["LIC_MEPS_CR_DE"].iloc[60] / 1000
    assert w_early.at["carbon_steel", "eur_per_kg"] == pytest.approx(toy.k_carbon * toy.panel["S"].iloc[10])

def test_unfilled_anchor_raises():
    with pytest.raises(ValueError, match="bom_anchor"):
        make_bom_deriver(assumptions_with_null_anchor())
```

  `weights_at` gains an `eur_per_kg` column.
- [ ] **Step 2: Run** `uv run pytest -q tests/test_bom_causal.py`. Expected: FAIL.
- [ ] **Step 3: Implement**, then fill the local config:

  ```
  uv run python -X utf8 -c "from src.testbench.targets import anchor_values; from src.testbench.panel import build_full_panel; print(anchor_values(build_full_panel(True)))"
  ```

  Paste the output into the local `config/assumptions.yaml` only.
- [ ] **Step 4: Run** `uv run pytest -q tests`. Expected: all pass. Then:

  ```
  uv run python -X utf8 -c "from src.testbench.targets import make_bom_deriver; from src.testbench.panel import build_full_panel as b; p=b(True); der=make_bom_deriver(use_licensed=True); print(der(p.loc[:'2026-09'], p.loc[:'2026-09'], '2026-09').iloc[-1])"
  ```

  Expected: both products equal 100.
- [ ] **Step 5: Commit** `src/testbench/targets.py src/testbench/run.py src/testbench/audit.py src/external/screen.py src/external/run_champion.py config/assumptions.example.yaml tests/test_bom_causal.py`: `Causal BOM weights: no post-origin panel value in k_i`.

---

### Task 6: Leakage canaries (a) to (d)

**Files:**
- Create: `src/testbench/canaries.py`, `tests/test_canaries.py`

**Interfaces:**
- Consumes: `run_backtest(..., lags)`, `metrics.score_rows`, `metrics.summarize`.
- Produces:
  - `planted_leak_prepare(targets, shift=3) -> Callable[[pd.DataFrame], pd.DataFrame]`: adds `LEAK_<s> = s.shift(-shift)`.
  - `class PlantedLeak` (`name="planted_leak"`): p50 = `exog["LEAK_"+y.name]` last value when finite, else the last y; bands are ±1%.
  - `off_by_one_prepare(col) -> Callable`: adds `<col>_NEXT = col.shift(-1)`.
  - `class OffByOneModel(col)`: p50 = `last × exp(log(exog[col_NEXT].iloc[-1] / exog[col].iloc[-1]))`, falling back to `last` when the input is not finite.
  - `permute_targets(panel, cols, seed=0) -> pd.DataFrame`: i.i.d. permutation of monthly log changes, re-cumulated from the first value. Block length 1 is used so that within-block momentum cannot survive; the reason goes in a code comment.
  - `truncation_identity(raw, prepare, targets, models, origins, mode, lags, derive=None) -> pd.DataFrame`: rows `origin, model, n, max_abs_diff, PASS`. It compares `run_backtest(prepare(raw))` with `run_backtest(prepare(raw.loc[:o]))` at each origin, exact equality on p10/p50/p90 and identical row sets.
  - `perturbation_audit(raw, prepare, targets, models, origins, mode, lags, derive=None) -> pd.DataFrame`: Julia's audit, generalised to `prepare` and to every column, including `LIC_` and `X_`.
  - `run_canaries(raw, prepare, targets, models, derive, lags_by_mode, origins, out_dir) -> dict`. It writes `out_dir / f"{git_head()}.json"` with `{"head", "dirty", "results": {...}, "PASS": bool}`.
  - `git_head() -> str`, `git_dirty() -> bool`.

PASS rules:
- (a) `planted_leak` pooled rel_QL vs naive < 0.2 at h3, **and** `perturbation_audit` flags `planted_leak` (max_abs_diff > 0).
- (b) on permuted targets, every model's pooled rel_QL ≥ 0.97.
- (c) `truncation_identity` passes for every real model.
- (d) `truncation_identity` fails for `OffByOneModel`.

The report PASSes only if all four hold.

- [ ] **Step 1: Write the failing tests** on a synthetic panel (three series of 240 months, one AR(1) in log changes so that AR1 has real skill before permutation):
  - `test_planted_leak_has_absurd_skill_and_is_flagged`.
  - `test_permutation_kills_skill`: `Conformal(AR1Point())` has rel_QL < 0.97 on the AR series before permutation and ≥ 0.97 after.
  - `test_truncation_identity_passes_for_baselines_and_deriver`: uses `_toy_derive` from `test_testbench.py`; move it to `conftest.py`.
  - `test_off_by_one_is_caught`.
  - `test_report_written_and_keyed_by_head(tmp_path)`.
- [ ] **Step 2: Run** `uv run pytest -q tests/test_canaries.py`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q tests`. Expected: all pass. Record the runtime of `test_canaries.py`; it must stay under 60 s for CI.
- [ ] **Step 5: Commit** `src/testbench/canaries.py tests/`: `Leakage canaries: planted leak, permuted target, truncation identity, off-by-one`.

---

### Task 7: Statistics

**Files:**
- Create: `src/testbench/stats.py`, `tests/test_stats.py`
- Modify: `src/testbench/metrics.py:45-60`. Keep `dm_test` and add an explicit `origin` sort in `summarize` before calling it.

**Interfaces:**
- Produces:
  - `GROUPS = {"metals": ["S01","S02","S03","S04","S05","D01","D02","S07","S08"], "steel": ["S09","S10","S12"], "plastics": ["S11","S13"], "bom": ["RM6_NE_IQI_433520","RMAirSeT_IIQ_24kV_IEC"]}`.
  - `paired(scored, model, ref, series=None, horizon=None, loss="ql") -> pd.DataFrame`: columns `origin, d`, sorted by origin, inner-joined on `(series, origin, horizon)`.
  - `dm_ordered(scored, model, ref, series, horizon, extra_lag=0, loss="ql") -> tuple[float, float]`: HAC with `horizon + extra_lag - 1` lags, HLN correction, t(n−1), two-sided.
  - `clark_west(scored, model, ref, series, horizon, extra_lag=0) -> tuple[float, float]`. It uses the log point errors `e = log y − log p50`: `f = e_ref² − (e_mod² − (p_ref − p_mod)²)`, HAC t-stat, one-sided p (model better).
  - `pooled_dm(scored, model, ref, group, horizon, extra_lag=0, loss="ql") -> tuple[float, float, int]`: mean loss differential across the group's series per origin, then HAC. Returns `(stat, p_one_sided_better, n_origins)`.
  - `holm(pvals: pd.Series) -> pd.Series`: adjusted p-values.
  - `block_bootstrap_ci(scored, model, ref, horizon, stat in {"rel_mae","rel_ql","coverage"}, block=None, n_boot=2000, seed=0, level=0.95) -> tuple[float, float, float]`: moving blocks of origins, block length = horizon by default.
  - `placebo_rows(base_rows_naive, sigma, seed) -> pd.DataFrame` (model `placebo_<sigma>_<seed>`): p10/p50/p90 × `exp(N(0, sigma))`, one draw per row.
  - `mde_table(scored_naive, sigmas=(0.005,0.01,0.02,0.04,0.08), seeds=range(20), alpha=0.05, power=0.8) -> pd.DataFrame` per group × horizon. The MDE is the smallest mean relative-QL gap `QL_placebo/QL_naive − 1` at which `pooled_dm(naive vs placebo)` rejects in ≥ 80% of seeds.

- [ ] **Step 1: Write the failing tests.**

```python
def test_loss_ordering_perfect_naive_anti():          # spec 5.7 sanity
    # synthetic rows: perfect (p50=y_true), naive, anti (p50 = 2*naive - y_true)
    assert ql(perfect) < ql(naive) < ql(anti)

def test_dm_ordered_sorts_by_origin():
    shuffled = scored.sample(frac=1, random_state=0)
    assert dm_ordered(shuffled, "m", "naive", "A", 3) == dm_ordered(scored, "m", "naive", "A", 3)

def test_clark_west_detects_nested_signal_dm_misses():
    # model = naive + small true signal (rho=0.15) on 150 origins; CW p < 0.05
    stat, p = clark_west(scored_signal, "m", "naive", "A", 3)
    assert p < 0.05

def test_holm_matches_hand_computation():
    assert holm(pd.Series([0.01, 0.04, 0.03])).round(3).tolist() == [0.03, 0.06, 0.06]

def test_pooled_dm_handles_missing_series_at_some_origins():
    stat, p, n = pooled_dm(scored_with_gap, "m", "naive", "steel", 3)
    assert np.isfinite(stat) and n == expected_n_origins

def test_bootstrap_ci_contains_point_and_is_seeded():
    lo, pt, hi = block_bootstrap_ci(scored, "m", "naive", 3, "rel_ql", seed=0)
    assert lo <= pt <= hi and (lo, pt, hi) == block_bootstrap_ci(scored, "m", "naive", 3, "rel_ql", seed=0)

def test_mde_monotone_in_sigma():
    t = mde_table(scored_naive, seeds=range(10))
    assert (t["mde_rel_ql"] > 0).all()
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Reuse `metrics.dm_test`'s HAC + HLN core by refactoring it into `_hac_t(d, lags) -> (stat, n)`. Keep `dm_test`'s public signature unchanged.
- [ ] **Step 4: Run** `uv run pytest -q tests`. Expected: all pass.
- [ ] **Step 5: Commit** `src/testbench/stats.py src/testbench/metrics.py tests/test_stats.py`: `Ordered DM, Clark-West, pooled DM + Holm, block bootstrap, placebo MDE`.

---

### Task 8: Strategy A meta pass

**Files:**
- Create: `src/testbench/meta.py`, `tests/test_meta.py`

**Interfaces:**
- Consumes: base forecasts (long format from `run_backtest`, with `y_true`), `realtime_lags()`.
- Produces:
  - `@dataclass MetaConfig`:
    - `window: int = 36` (trailing origin-months of realised rows per series × horizon);
    - `min_rows: int = 12`;
    - `power: float = 2.0` (weights ∝ trailing QL^−power);
    - `shortlist: tuple[str, ...] = ()`. Empty means `wf_median` uses every candidate with a row at t. `run.py` passes `SHORTLIST` from Task 11;
    - `fallback: str = "naive"`.
  - `realised(base, t, mode, lags) -> pd.Series[bool]`: `target_month <= t` in submission mode; `target_month <= t − lags[series]` in realtime mode. Raises `KeyError` if a series has no lag in realtime mode.
  - `meta_pass(base, cfg, mode, lags=None, origins=None) -> tuple[pd.DataFrame, pd.DataFrame]`:
    - The first frame holds rows in the base schema, with model ∈ {`wf_best`, `wf_blend`, `wf_median`}, `last_obs` and `step` copied from the fallback row, and `y_true` copied.
    - The second frame, `diag`, has columns `origin, series, horizon, n_realised, pick, weights (json str), fallback_used`.
  - `pick_stability(diag) -> pd.DataFrame`: switch rate per series × horizon, share each model is picked, and the agreement rate between `wf_best` and the argmax weight of `wf_blend`.

Algorithm per (origin t, series, horizon):
1. `R` = base rows with `realised` true and origin in `(t − window, t]`, scored with `metrics.score_rows`.
2. Eligible candidates are those with ≥ `min_rows` rows in `R`. Losses are compared on the **common** rows of eligible candidates only. If none are eligible, use the fallback.
3. `wf_best` = the argmin of the loss.
4. `wf_blend`:
   - weights `w ∝ L^−power`, normalised;
   - combine each quantile geometrically: `exp(Σ w log q)`;
   - then sort the three quantiles.
5. `wf_median`: per-quantile median over shortlist ∩ candidates that have a row at t, then sort.

- [ ] **Step 1: Write the failing tests** (synthetic base frames built by hand):

```python
def test_only_realised_rows_used():          # perturb y_true and forecasts of rows with target_month > t
    a, _ = meta_pass(base, cfg, "submission", origins=[t])
    b, _ = meta_pass(perturb_unrealised(base, t), cfg, "submission", origins=[t])
    pd.testing.assert_frame_equal(a, b)

def test_realtime_uses_series_lag():
    r = realised(base, t, "realtime", {"A": 2})
    assert base.loc[r, "target_month"].max() == t - pd.DateOffset(months=2)

def test_realtime_missing_lag_raises():
    with pytest.raises(KeyError, match="B"):
        realised(base_with_B, t, "realtime", {"A": 2})

def test_best_picks_lower_loss_and_blend_weights_follow_power():
    # model "good" has QL half of "bad" on all realised rows; power=2 -> weights 0.8 / 0.2
    _, diag = meta_pass(base_good_bad, MetaConfig(power=2.0, shortlist=("good", "bad")), "submission", origins=[t])
    assert diag.iloc[0]["pick"] == "good"
    assert json.loads(diag.iloc[0]["weights"]) == pytest.approx({"good": 0.8, "bad": 0.2})

def test_ineligible_candidate_dropped_and_fallback_when_none():
    # "sparse" has 5 realised rows < min_rows -> absent from weights; at an origin with 0 eligible -> naive copy
    ...

def test_quantiles_sorted_after_blend():
    out, _ = meta_pass(crossing_base, cfg, "submission")
    assert ((out.p10 <= out.p50) & (out.p50 <= out.p90)).all()
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Vectorise per (series, horizon) and loop over origins. On the dev base pass the meta pass must take < 60 s; measure it in Task 12.
- [ ] **Step 4: Run** `uv run pytest -q tests`. Expected: all pass. Add the meta pass to `truncation_identity`'s coverage by running `test_only_realised_rows_used` inside `run_canaries` as check (c-meta).
- [ ] **Step 5: Commit** `src/testbench/meta.py src/testbench/canaries.py tests/test_meta.py`: `Strategy A: walk-forward best / blend / median meta pass`.

---

### Task 9: Strategy B pooled steel chain

**Files:**
- Create: `src/testbench/steel_chain.py`, `tests/test_steel_chain.py`

**Interfaces:**
- Consumes: the `_frame` and `NaiveVolScaled` contract.
- Produces: `@dataclass SteelChain` with:
  - `name="steel_chain"`, `targets=("S09","S10","S12")`;
  - `pools = {"EU": {"S09": "LIC_MEPS_CR_DE", "S10": "LIC_MEPS_HDG_DE"}, "US": {"S12": ("X_US_STEEL_SCRAP", "X_US_PPI_STEEL")}}`;
  - `alpha_per_obs=0.5`, `window=120`, `min_rows=48`, `n_cal=48`, `min_cal=18`, `halflife=3.0`, `clip=0.5`.

Model:
- Features at month τ, for each pooled series s:
  - `own_d1`, `own_d3`: log changes of s at τ;
  - `drv_d1`, `drv_d3`: log changes of each driver at τ−1 (lagged one month).
- Target: `log s(τ+st) − log s(τ)`, direct per `st` in `steps`.
- Rows from every series of the pool are stacked over the last `window` months of exog.
- Fit: no intercept and no centring. Scale each feature by its pooled standard deviation. Ridge penalty `alpha_per_obs × n`. Clip the prediction to ±`clip`.
- Bands are conformal with volatility scaling:
  - At pseudo-origins k = the last `n_cal` months with a known target, refit on data ≤ k.
  - Residual `e_k` = realised minus predicted, divided by `EWMA_vol(k) × sqrt(st)`.
  - Bands = `p50 × exp(quantile(z, .1|.9) × cur_vol × sqrt(st))`, where `cur_vol` uses `NaiveVolScaled`'s floor rule.
  - With fewer than `min_cal` residuals, use `NaiveVolScaled` bands.
- Fallback to `NaiveVolScaled().forecast` when:
  - the series has no pool;
  - a pool driver column is missing (`use_licensed: false`, so no `LIC_` columns);
  - the driver's last value is NaN;
  - there are fewer than `min_rows` rows.
- Constants are fixed here a priori and never tuned on scores (spec 7).

- [ ] **Step 1: Write the failing tests** on a synthetic panel where `S09` follows the driver `LIC_MEPS_CR_DE` with a one-month lag (true coefficient 0.6) plus noise:
  - `test_learns_lagged_driver`: p50 direction matches the driver's last move in > 80% of 50 origins.
  - `test_falls_back_to_naive_vol_without_licensed`: drop the `LIC_` columns; the result equals `NaiveVolScaled().forecast` exactly.
  - `test_falls_back_before_driver_starts`: driver NaN before month 100, origin at month 90; result equals the naive_vol output.
  - `test_contract`: index equals steps, columns `p10, p50, p90`, ordered, finite.
  - `test_truncation_identity`: use `canaries.truncation_identity` with `SteelChain` and pass.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q tests`. Expected: all pass.
- [ ] **Step 5: Commit** `src/testbench/steel_chain.py tests/test_steel_chain.py`: `Strategy B: pooled steel chain with lagged CRU/MEPS drivers`.

---

### Task 10: Walk-forward signal screen and ChampionWF

**Files:**
- Modify: `src/external/screen.py` (add `screen_at`; keep `screen` and label it "exploration only, uses the inspected 2024 split" in its docstring), `src/external/models.py` (add `ChampionWF`)
- Test: `tests/test_screen_wf.py`

**Interfaces:**
- Produces:
  - `screen_at(info: pd.DataFrame, target: str, horizons=(3,6,9), max_keep=6, min_rows=60) -> dict[tuple[str,int], list[str]]`.
    - Uses only pairs (feature at τ, change τ→τ+h) where τ+h ≤ the last valid month of `info[target]`.
    - Stability halves = the first and second halves of those rows.
    - Keeps the same rule as `screen`: same sign in both halves and |corr| > 2/√(n/h). Ranks by |corr|.
  - `@dataclass ChampionWF`, `name="champion_wf"`, `min_signals=3`, `max_keep=6`. `forecast(y, steps, exog)` calls `screen_at(exog, y.name)` and delegates to `Champion(selected=...)`. The screen runs at every origin; the cadence knob from spec 7 is not used unless Task 12 shows the runtime budget is exceeded.

- [ ] **Step 1: Write the failing tests.**
  - `test_screen_at_ignores_future`: `screen_at(info.loc[:t], ...)` equals `screen_at(info_full_masked_after_t, ...)`, and perturbing values after t changes nothing.
  - `test_screen_finds_planted_signal`: a synthetic feature that leads the target at h=3 with corr 0.5 over 240 months is kept, and pure noise is not kept.
  - `test_champion_wf_truncation_identity`: use the canaries helper.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Refactor the correlation body of `screen` into a private function shared by both.
- [ ] **Step 4: Run** `uv run pytest -q tests`. Expected: all pass.
- [ ] **Step 5: Commit** `src/external/screen.py src/external/models.py tests/test_screen_wf.py`: `Walk-forward signal screen; ChampionWF`.

---

### Task 11: Candidate set and shortlist

**Files:**
- Create: `src/testbench/candidates.py`, `tests/test_candidates.py`

**Interfaces:**
- Produces:
  - `candidate_models(use_licensed: bool) -> list`, in this order:
    1. `Naive()`, `NaiveGaussian()`;
    2. `NaiveVolScaled(halflife=3, name="naive_vol_hl3")`, `NaiveVolScaled(halflife=6, name="naive_vol_hl6")`, `NaiveVolScaled(halflife=12, name="naive_vol_hl12")`;
    3. `Conformal(DriftPoint())`, `Conformal(AR1Point())`, `Conformal(MeanReversionPoint())`;
    4. `DriverRidge()`, `SteelChain()`, `ChampionWF()`.
  - `use_licensed` has no effect on the list. `SteelChain` falls back by itself; the flag is passed only for the manifest.
  - `SHORTLIST = ("naive", "naive_vol_hl3", "ar1_conf", "driver_ridge", "champion_wf", "steel_chain")`.
- [ ] **Step 1: Write the failing test.** Names are unique, `naive` is first, `SHORTLIST ⊂ names`, and every model satisfies the contract on a 200-month synthetic series with exog.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** `NaiveVolScaled` already accepts `name`; set it explicitly.
- [ ] **Step 4: Run** `uv run pytest -q tests`. Expected: all pass.
- [ ] **Step 5: Commit** `src/testbench/candidates.py tests/test_candidates.py`: `Pre-registered candidate set and wf_median shortlist`.

---

### Task 12: Protocol, manifest, CLI

**Files:**
- Create: `src/testbench/protocol.py`, `tests/test_protocol.py`
- Modify: `src/testbench/run.py` (rewrite `main`; keep `build_panel`, `DEFAULT_TARGETS` and `bom_deriver` for importers), `src/testbench/audit.py` (it becomes `python -m src.testbench.audit` = `run_canaries` on real data), `docs/testbench.md`

**Interfaces:**
- Consumes everything above.
- Produces:
  - Constants `DEV = ("2010-01-01", "2021-12-01")`, `GRADED = ("2022-01-01", "2026-09-01")`, `PURGE = pd.Timestamp("2022-01-01")`.
  - `dev_view(scored) -> pd.DataFrame`: keeps rows with origin in DEV and `target_month < PURGE`. It asserts that no row with origin ≥ PURGE survives.
  - `assert_can_grade(root=ROOT, canary_dir=OUT/"canaries") -> str`. It returns the tag and raises `RuntimeError` naming the failed check, in this order:
    1. HEAD has an exact tag (`git describe --exact-match --tags HEAD`);
    2. `git status --porcelain -- src config tests pyproject.toml uv.lock` is empty;
    3. the canary report for HEAD exists with `"dirty": false` and `"PASS": true`;
    4. `OUT/"backtest"/"graded"/<tag>` does not exist yet ("scored once").
  - `write_manifest(out_dir, mode, origins, runtime_s, inputs: list[Path], configs: list[Path], use_licensed) -> Path`. It writes `manifest.json` with `git_head, dirty, tag (or null), config_sha256 {path: hash}, input_sha256 {path: hash}, mode, origins [first, last, n], runtime_s, python, packages {numpy, pandas, scipy, sklearn, statsmodels}, use_licensed, created_utc`.
- CLI `uv run python -m src.testbench.run`:
  - `--dev` (default): base pass on DEV origins in both modes, meta pass, dev_view scores, stats tables. Output goes to `data/outputs/backtest/dev/<mode>/`.
  - `--canaries`: `run_canaries` on real data, with origins `2012-06`, `2016-03`, `2020-06`, both modes, all candidates, plus the BOM deriver and `prepare = build_full_panel` columns.
  - `--graded`: `assert_can_grade`, then the base pass on `2010-01..2026-09` in both modes, the meta pass, full scoring and stats. Output goes to `data/outputs/backtest/graded/<tag>/<mode>/`.
  - `--submit`: reads the graded submission-mode forecasts for the current HEAD tag (error if missing). It takes `wf_blend` at origin `2026-09` and writes `data/outputs/backtest/graded/<tag>/<team_id>_forecast_202609.csv` via `from_forecasts` + `validate_submission`. It asserts that no `target_id` starts with `LIC_`, `EPX_` or `X_`.
  - `--mode {submission,realtime,both}` (default both), `--no-licensed` (overrides config).
  - Every run computes the submission-mode lags with `ragged_edge_lags(build_full_panel(...), live_origin, exempt=set(DEFAULT_TARGETS))` once, before any truncation.
- Outputs per mode folder:
  - `forecasts.parquet`, `meta_diag.parquet`;
  - `scores.csv` (model × series × horizon), `overview.csv`;
  - `pooled_tests.csv` (group × horizon × model vs naive: rel_QL, pooled DM stat/p, Holm p, CW p for point models, bootstrap CI). In realtime mode every DM / CW / pooled call gets `extra_lag = max realtime lag of the series involved` (spec 5.7: step = horizon + publication lag). Holm runs across groups × horizons within each model and mode;
  - `pick_stability.csv`, `compute_footprint.csv`, `manifest.json`.
  - `holdout_selection.csv` is no longer produced (2024 split was inspected).

- [ ] **Step 1: Write the failing tests** (git operations against a temporary repo created in `tmp_path` with `git init`):
  - `test_dev_view_purges`: rows with origin 2021-11 and h=3 (target 2022-02) are dropped; origin 2021-06 h=3 is kept; origin 2022-01 raises.
  - `test_grade_refused_without_tag`, `test_grade_refused_dirty`, `test_grade_refused_without_passing_canaries`, `test_grade_refused_second_time`.
  - `test_manifest_fields_and_hashes(tmp_path)`.
  - `test_submit_rejects_licensed_ids`: a forecast frame with series `LIC_MEPS_CR_DE` raises `ValueError`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement**, and rewrite `docs/testbench.md` (commands, outputs, modes, lags table pointer, the lag-0 → lag-1 change for daily-market averages, the declared anchor limit).
- [ ] **Step 4: Run** `uv run pytest -q tests`. Expected: all pass. Then on real data:

  ```
  uv run python -X utf8 -m src.testbench.run --canaries
  ```

  Expected: a report written and `PASS: True`. Then `uv run python -X utf8 -m src.testbench.run --dev --mode submission`. Expected: it completes, and the runtime printed in `compute_footprint.csv` is < 15 min. If it is over, set the `ChampionWF` screen cadence to "first origin of each year" (recomputed from `exog.loc[:Jan]` of the current call so that the identity holds). Re-run `--canaries` and note the change in `docs/testbench.md`.
- [ ] **Step 5: Commit** `src/testbench/protocol.py src/testbench/run.py src/testbench/audit.py docs/testbench.md tests/test_protocol.py`: `Protocol: dev/graded split, grade guard, manifest, CLI`.

---

### Task 13: Dev-window decisions, MDE, preregistration

**Files:**
- Create: `docs/backtest/preregistration.md`
- Modify (only if a dev choice changes a default): `src/testbench/meta.py` (`MetaConfig.power`)

This task uses the **dev window only**. No graded origin is run.

- [ ] **Step 1:** Run `uv run python -X utf8 -m src.testbench.run --dev`. Both modes must complete.
- [ ] **Step 2:** Choose `MetaConfig.power` ∈ {1, 2, 4} by pooled dev QL of `wf_blend` in submission mode. Ties go to the lower power. Keep `window=36` and `min_rows=12` (fixed a priori). Record the dev table in `data/outputs/backtest/dev/` and only the choice in the preregistration.
- [ ] **Step 3:** Run the MDE: `stats.mde_table` on dev naive rows, both modes. Save it to `data/outputs/backtest/dev/mde.csv`. The preregistration gets the relative MDE per group × horizon; these are percentages, with no price values. Before writing them into the tracked file, confirm with the user that relative percentages are acceptable there.
- [ ] **Step 4: Write `docs/backtest/preregistration.md`** with:
  - candidate set and SHORTLIST (Task 11);
  - MetaConfig values;
  - submitted model `wf_blend`;
  - metrics: QL primary, rel_MAE, coverage 80% target, below/above;
  - PASS rule: `wf_blend` beats naive on pooled QL in a group × horizon iff the one-sided pooled DM Holm-adjusted p < 0.05 **in both modes**; anything else is reported as "not detectable" with the MDE;
  - no per-series significance claim at h = 9;
  - MDE table;
  - declared prior looks: Julia's 2024–26 inspections (half-life, hist_floor, champion rule, recenter test) and our EDA correlations over the full sample;
  - declared limits: latest-vintage truth, the BOM anchor constants dated 2016-11, `DriverRidge` alpha hand-set before this work, epoxy correction;
  - the graded command and its single-use guard.
- [ ] **Step 5: Commit and tag, only with the user's word.** Commit `docs/backtest/preregistration.md` (plus `meta.py` if changed): `Pre-registration for graded backtest`. Re-run `--canaries` on the clean committed HEAD (report `dirty: false`). Then propose the tag `backtest-freeze-v1` to the user; they create it or approve it.

---

### Task 14: Graded run and submission (after the user's go)

- [ ] **Step 1:** `uv run python -X utf8 -m src.testbench.run --graded`. Expected: guard passes, both modes run once, outputs and manifests are in `data/outputs/backtest/graded/backtest-freeze-v1/`.
- [ ] **Step 2:** `uv run python -X utf8 -m src.testbench.run --submit`. Expected: `validate_submission` returns no issues, and the CSV has 16 targets × 3 horizons = 48 rows with no `LIC_`, `EPX_` or `X_` ids.
- [ ] **Step 3:** Report to the user:
  - `pooled_tests.csv` per group × horizon for `wf_blend`, `wf_best`, `wf_median`, `steel_chain` and the baselines;
  - coverage;
  - pick stability;
  - PASS/"not detectable" per the preregistration;
  - realtime vs submission differences.
  - Numbers stay in `data/outputs`; the chat gets the summary.
- [ ] **Step 4:** Use superpowers:requesting-code-review on the branch, then finishing-a-development-branch only if the user asks.
