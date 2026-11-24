# AI track: plan and to-do list

*This file contains methods and relative metrics only, so it is safe to push under the NDA. Prices, BOM data and supplier data stay in `data/` (git-ignored).*

## 1. Goal

The kick-off deck (slide 16) offers a bonus for an **"AI-heavy approach: agents, pretrained models, LLM features, always disclosed and validated"**. It also asks us to compare our approach with pretrained forecasting models, list every model or API with its licence, report compute and CO₂, and disclose AI tools (the CSV's `ai_tools` column).

The AI track is a **standalone package** (`src/ai/`) with its own one-command run and its own CSV. Its models also implement the team test-bench interface (`forecast(y, steps, exog) → p10/p50/p90`), so they can be scored inside `champion/v2` without changes.

What we expect: market prices are close to a random walk, so the AI models should not beat naive much on central forecasts. Their value is in **better ranges**, **extra signals**, and an AI process that is **properly disclosed and validated**.

## 2. Ground rules (every component)

| Rule | How |
|---|---|
| **NDA** | Pretrained time-series models run **locally**, so they may use licensed series. Every LLM API call goes through `guard.public_only()`: public series and public text only, checked against the `public/scan_leaks.py` deny-list. Every prompt and response is logged. |
| **No look-ahead** | The model input is cut at the forecast date. Drivers are shifted by their publication lag (`src/features.py`). The perturbation audit (scramble the future → forecast unchanged) must pass for every model. |
| **Training contamination** | A pretrained model or LLM may have seen the history. Each model is also scored on its **clean window**: target months after its release (`config.MODELS[...]["clean_after"]`). |
| **Pre-registered selection** | Choose per price × horizon on targets **before 2024**. Report only **origins from 2024**: a forecast made in 2023 for a 2024 month is in neither set. |
| **Disclosure** | Licence table, `ai_tools` column, prompt log, runtime and token count → compute and CO₂ estimate. |

## 3. Components

| # | Component | Status |
|---|---|---|
| 1 | **Pretrained time-series models**: Chronos-Bolt small/base, and Chronos-2 with the economic drivers as inputs. Zero-shot, local, Apache-2.0, versions pinned. | ✅ done |
| 2 | **Agent-proposed features**: an LLM proposes features in a small, restricted formula language, from public metadata only. The local screening decides which to keep (notebook 03's procedure, on pre-2024 data). Runs on the local **Qwen3-14B** (Ollama) by default, or Claude. | ✅ code done; first result negative, **to re-run** (look-ahead fixes of 2026-11-24) |
| 3 | **LLM news / regime signal**: public dated headlines (GDELT) scored by an LLM for supply/demand tension, per material and month. Must beat the GDELT-tone baseline. | 🔜 (local Qwen or Claude) |
| 4 | **LLM analyst forecast**: P10/P50/P90 from public series only. Usable only for the live forecast plus a check on months after the LLM's training cutoff, and flagged as such. | 🔜 (local Qwen or Claude) |
| 5 | **Packaging**: documentation, public results page, `make ai`, CI, merge with the team. | 🔜 next |

## 4. Phase 1 results (rolling backtest, 12 prices, h = 3/6/9; re-run of 2026-11-24)

"Error" is MAE and "range quality" is the P10/P90 pinball loss, both relative to a benchmark (below 1 = better). Coverage is the share of outcomes inside P10–P90 (target 0.80). Naive gets online conformal ranges. `naive_vol` is the team's volatility-scaled naive (`champion/v2`), ported in `baselines.py`.

**Clean window: target months after the model's release.** The only check the pretrained models cannot have seen. Combinations and the pick start after the latest release (Chronos-2).

| Model | n | Error vs naive | Range quality vs naive | Range quality vs naive_vol | Coverage |
|---|---|---|---|---|---|
| naive_vol (team benchmark) | 399 | 1.00 | 0.75 | 1 | 0.82 |
| Chronos-Bolt base | 795 <!-- nda-ok: count of forecasts, not a price --> | 1.04 | 0.81 | 1.14 | 0.88 |
| Chronos-2 with drivers | 399 | 1.01 | 0.96 | 1.28 | 0.73 |
| Chronos-2 + classical average | 399 | **0.94** | 0.98 | 1.30 | 0.93 |
| Chronos-2 P50 + naive_vol fan | 399 | 1.01 | 0.80 | 1.07 | 0.82 |
| Chronos-2 ~ naive_vol quantile average | 399 | 1.00 | 0.82 | 1.09 | 0.81 |
| Pre-registered pick | 399 | 0.97 | 0.79 | 1.06 | 0.85 |

The clean window for Chronos-2 is 12 target months only: overlapping forecasts, so few independent observations.

**Origins from 2024 on (n = 975): possible training contamination.** <!-- nda-ok: count of forecasts -->

| Model | Error vs naive | Range quality vs naive | Range quality vs naive_vol | Coverage |
|---|---|---|---|---|
| naive_vol (team benchmark) | 1.00 | 0.73 | 1 | 0.85 |
| Classical ensemble (refit here) | 1.04 | 1.00 | 1.38 | 0.96 |
| Chronos-Bolt base | 1.06 | 0.83 | 1.14 | 0.88 |
| Chronos-2 with drivers | 0.97 | 0.82 | 1.12 | 0.86 |
| Chronos-2 P50 + naive_vol fan | 0.97 | 0.75 | 1.02 | 0.88 |
| Chronos-2 ~ naive_vol quantile average | 0.97 | 0.75 | 1.03 | 0.89 |
| Pre-registered pick | **0.95** | 0.77 | 1.06 | 0.90 |

**Correction (2026-11-24).** The first version reported every *target month* from 2024 (n = 1191). 216 of those forecasts were made in 2023, so the pick used outcomes not known at their origin. Reporting now starts at *origins* from 2024; the choice rule (targets before 2024) and the choices are unchanged. Other fixes: pooled bands and the pooled LightGBM only use published outcomes; live bands only use outcomes up to the official origin.

| Error vs naive / range quality vs naive_vol | Before | After (origins from 2024) | Clean window |
|---|---|---|---|
| Chronos-2 with drivers | 0.99 / 1.07 | 0.97 / 1.12 | 1.01 / 1.28 |
| Chronos-2 ~ naive_vol quantile average | 0.96 / 1.01 | 0.97 / 1.03 | 1.00 / 1.09 |
| Pre-registered pick | 0.97 / 1.04 | 0.95 / 1.06 | 0.97 / 1.06 |

**Reading:**
- **naive_vol is the range model to beat, and no AI model beats it** in either window. Against plain naive the AI ranges look better; most of that comes from naive's fixed, shock-inflated bands.
- **Chronos-2's own central-forecast gain does not survive the clean window** (0.97 on all origins from 2024, 1.01 on months after its release) and its own ranges are too narrow there (coverage 0.73). Its quantile average with naive_vol ties naive (1.00).
- **The pick (0.97) and the Chronos-2 + classical average (0.94) stay below 1 in the clean window**, on 12 target months and with no significance test on that window: encouraging, not proven.
- **Caveat on the combinations:** the two naive_vol combinations were added after seeing that naive_vol had the best bands (the selection rule still only uses targets before 2024, but the candidate list was informed by the aggregate result). Report them as exploratory.
- **The earlier conclusion that the fixed quantile average beats the pick no longer holds on the central forecast** (0.95 vs 0.97); it still has slightly better ranges (1.03 vs 1.06).
- **CQR dropped:** it widened the AI ranges too much in the first run (coverage 0.95+, range quality > 1 vs naive); not re-tested.
- **Compute:** about 5 s of inference for the full backtest; negligible energy and CO₂.

## 5. How to run

```bash
uv sync --group ai                                             # torch + chronos-forecasting
make ai                                                        # backtest + live forecast (~1 min; needs the NDA data)
uv run --group ai --with pytest python -m pytest -q tests/       # tests (guard tests skip without the NDA data)
cp .env.example .env    # then fill ANTHROPIC_API_KEY (phases 2-4); .env is git-ignored
uv run --group ai python -m src.ai.llm                           # connectivity check (a few tokens)
```

Outputs (private, `data/outputs/ai/`): `ai_backtest.parquet`, `ai_scores_*.csv`, `ai_selection.csv`, `ai_live_forecast.parquet`, `team_xhec_forecast_ai.csv`, `ai_licences.csv`, `ai_compute_footprint.csv`.

## 6. Files

| File | Role |
|---|---|
| `config.py` | Disclosure registry: model ids, pinned revisions, licences, release dates, clean windows, settings (incl. `LIVE_ORIGIN`) |
| `baselines.py` | `naive_vol`, ported from the team test bench (range benchmark and candidate) |
| `foundation.py` | Inputs cut at the forecast date, batched local inference, adapter to the team test bench |
| `backtest.py` | AI rows in the `src/backtest.py` format; CQR, averaging, hybrid / quantile-average combinations, comparison, pre-registered selection |
| `run.py` | One command: backtest → selection → live forecast → CSV, licences, compute |
| `guard.py` | NDA gate: public-series allow-list + deny-list text check (shared with `public/scan_leaks.py`) |
| `feature_dsl.py` | Formula language for LLM-proposed features: parser, leak-free compiler, notebook-03 screen |
| `agent_features.py` | Phase 2 loop: Claude proposes formulas, the local screen decides, feedback is coarse buckets only |
| `llm.py` | Only LLM entry point (local Qwen via Ollama, or Claude): gate → cache → budget cap → call → JSONL log |
| `agent_eval.py` | dynreg with vs without the agent's accepted features, scored on 2024–26 |
| `../../tests/test_ai_guard.py` | The gate refuses licensed series, client / source words, private numbers and terms; `ask` refuses before any network call |
| `../../tests/test_ai_leakage.py` | One test per look-ahead leak fixed on 2026-11-24 (selection embargo, live cutoff, publication lags, formula language, LLM prompts) |
| `../../tests/test_ai_foundation.py` | No look-ahead (AI inputs and naive_vol), publication lags, CQR, selection (synthetic data) |

## 7. To-do list

### Phase 1: pretrained models ✅
- [x] Chronos-Bolt small/base and Chronos-2 (with drivers) running locally, versions pinned, Apache-2.0 confirmed on the model cards
- [x] Adapter to the team test bench (`FoundationModel.forecast`)
- [x] Backtest rows in the `src/backtest.py` format, scored with the same helpers
- [x] Perturbation audit on the real models (Chronos-Bolt, Chronos-2 with drivers)
- [x] Clean-window scoring after each model's release
- [x] Pre-registered selection (targets before 2024 → report origins from 2024)
- [x] Live forecast + CSV in Schneider's format with `ai_tools`
- [x] Licence table + compute / CO₂ estimate
- [x] Synthetic-data tests passing (7)

### Phase 1 follow-ups
- [x] Drop CQR variants from the selection candidates
- [x] Add the team's `naive_vol` as the range benchmark (and as a candidate)
- [x] Combinations: Chronos-2 P50 + naive_vol fan, and Chronos-2 ~ naive_vol quantile average
- [x] Align the live origin to 2026-09 (team convention: target = origin + h; lagged series forecast h + gap steps ahead; classical models refit at the origin, no more dependency on notebook 02's output)
- [ ] Decide the submitted AI model: a single fixed rule (quantile average) instead of the per-series pick, since the pick did worse
- [ ] Check the classical live bands for lagged series (residuals at h are used for a step of h + 1)
- [ ] Score the AI models inside `champion/v2` (`src/testbench`) through `FoundationModel`
- [ ] Optional: TimesFM (check licence and release date first)
- [ ] Add the RM6 / AirSeT BOM cost indices as targets

### Shared LLM plumbing ✅
- [x] `.env` (git-ignored) + `.env.example` for `ANTHROPIC_API_KEY`, `AI_MODEL` (default `claude-opus-5-5`), `AI_BUDGET_USD`
- [x] `guard.py`: NDA gate (public-series allow-list + deny-list text check) + 9 tests
- [x] `llm.py`: response cache, budget cap, JSONL prompt/response log, server-side refusal fallback, JSON-schema outputs
- [x] Local provider: `qwen3:14b` through Ollama (`AI_PROVIDER=ollama`, default; same NDA gate, $0, ~90 s per call with reasoning on); registered in the licence table (Apache-2.0)
- [x] `public/scan_leaks.py` refactored: `compile_patterns()` / `find_leaks()` shared by the file scan and the gate
- [x] Connectivity check with Qwen (local). Claude API key set, account still needs credits

### Phase 2: agent-proposed features (needs `ANTHROPIC_API_KEY`)
- [x] Formula language (`dlog`, `lag`, `ratio`, `spread`, `dev_ma`, `zscore`, `vol`, `TARGET`) + safe `ast` compiler with publication lags (`feature_dsl.py`, 21 tests incl. scramble-the-future)
- [x] Screening from notebook 03 on pre-2024 targets only (n ≥ 60, |ρ| ≥ 0.10, sign-stable in ≥ 2 regimes, not a duplicate of an existing feature)
- [x] Agent loop (`agent_features.py`): public metadata in, pass/fail + signal bucket back, 3 rounds × 8 formulas per family, acceptance = passes at ≥ 2 of 3 horizons; tested with a fake LLM (prompts pass the gate, licensed series and code injection refused, no numbers in the feedback)
- [x] Full run with Qwen (15 calls, ~28 min, $0): 93 formulas proposed, 0 rejected by the parser after the language fix, 107 (formula, target) pairs accepted (47 distinct formulas)
- [x] Evaluation script `agent_eval.py`: dynreg with vs without the accepted features, 2024–26 only
- [ ] **To re-run** after the look-ahead fixes (`dev_ma`, origins from 2024, post-release window). First run, on target months from 2024: **dynreg gets worse with the agent's features** (error vs naive 1.09 → 1.18, range quality 1.09 → 1.15, coverage 0.95 → 0.87). Small central-forecast gains on the EU steel PPIs (S09 0.88 → 0.80, S10 0.74 → 0.68), copper and aluminium, but with worse bands. Cause: the one-feature-at-a-time screen let up to 19 extra drivers in per target for ~100 training months (data mining). Not fed to Chronos-2.
- [ ] Possible follow-up, to be pre-registered before looking at 2024–26 again: cap at the 2 best features per target by pre-2024 rank, and require each to improve a pre-2024 rolling backtest, not just the screen
- [ ] Optional: same loop with Claude once credits are added, to compare the two LLMs

### Phase 3: LLM news / regime signal (needs `ANTHROPIC_API_KEY`)
- Note (2026-11-24): years are removed from headlines and the running month is skipped; the GDELT download is incomplete and nothing has been scored with the fixed prompt yet.
- [ ] GDELT headline fetch per material and month (2016 →)
- [ ] Baseline without an LLM: GDELT tone
- [ ] LLM tension score (−2…+2), date-free prompt, batch calls
- [ ] Commit the JSON cache (public-text only) so the run reproduces without a key
- [ ] Evaluate before and after the LLM's training cutoff separately
- [ ] Keep / reject rule: must beat both baselines after the cutoff

### Phase 4: LLM analyst forecast (needs `ANTHROPIC_API_KEY`)
- Note (2026-11-24): every series now stops at its own publication date; any earlier run must be redone. Claude's release date is not verified, so it has no clean window until it is.
- [ ] Prompt with public series only (+ news index + model forecasts)
- [ ] JSON P10/P50/P90 + short rationale per series and horizon
- [ ] Check on months after the cutoff only
- [ ] Separate, flagged `model_name` in the CSV
- [ ] Pre-registered inclusion rule (must not lose to naive after the cutoff)

### Phase 5: packaging and disclosure
- [ ] `docs/ai_track.md` (approach, licence table, leakage protocol, results, limits)
- [ ] Public results page in `public/ai/`, checked with `public/scan_leaks.py`
- [x] Licence table and AI-tool disclosure in the main README
- [x] `make ai` target
- [ ] GitHub Actions: run the synthetic AI tests
- [ ] Commit and open a PR from `ai-track` (only when agreed)

### Open questions
- [x] Claude API key slot and budget cap (`.env`); key still to be filled
- [ ] Team agreement on merging into `champion/v2`
- [ ] Which CSV the AI track submits: a separate file, or rows merged into the champion CSV
