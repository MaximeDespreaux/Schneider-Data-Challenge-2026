# AI track: plan and to-do list

*Branch `ai-track`. This file contains methods and relative metrics only, so it is safe to push under the NDA. Prices, BOM data and supplier data stay in `data/` (git-ignored).*

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
| **Pre-registered selection** | Choose per price × horizon on targets **before 2024**. Report only on **2024–26**. |
| **Disclosure** | Licence table, `ai_tools` column, prompt log, runtime and token count → compute and CO₂ estimate. |

## 3. Components

| # | Component | Status |
|---|---|---|
| 1 | **Pretrained time-series models**: Chronos-Bolt small/base, and Chronos-2 with the economic drivers as inputs. Zero-shot, local, Apache-2.0, versions pinned. | ✅ done |
| 2 | **Agent-proposed features**: an LLM proposes features in a small, restricted formula language, from public metadata only. The local screening decides which to keep (notebook 03's procedure, on pre-2024 data). | ⏳ blocked: API key |
| 3 | **LLM news / regime signal**: public dated headlines (GDELT) scored by an LLM for supply/demand tension, per material and month. Must beat the GDELT-tone baseline. | ⏳ blocked: API key |
| 4 | **LLM analyst forecast**: P10/P50/P90 from public series only. Usable only for the live forecast plus a check on months after the LLM's training cutoff, and flagged as such. | ⏳ blocked: API key |
| 5 | **Packaging**: documentation, public results page, `make ai`, CI, merge with the team. | 🔜 next |

## 4. Phase 1 results (rolling backtest, 12 prices, h = 3/6/9)

"Error" is MAE and "range quality" is the P10/P90 pinball loss, both relative to a benchmark (below 1 = better). Coverage is the share of outcomes inside P10–P90 (target 0.80). Naive gets online conformal ranges. `naive_vol` is the team's volatility-scaled naive (`champion/v2`), ported in `baselines.py`.

| Model (targets 2024 on, n = 1191) | Error vs naive | Range quality vs naive | Range quality vs naive_vol | Coverage |
|---|---|---|---|---|
| naive_vol (team benchmark) | 1.00 | 0.75 | 1 | 0.87 |
| Classical ensemble (notebook 02) | 1.07 | 1.01 | 1.35 | 0.95 |
| Chronos-Bolt base | 1.10 | 0.83 | 1.11 | 0.90 |
| Chronos-2 with drivers | 0.99 | 0.80 | 1.07 | 0.87 |
| Chronos-2 P50 + naive_vol fan | 0.99 | 0.76 | 1.02 | 0.89 |
| Chronos-2 ~ naive_vol quantile average | **0.96** | 0.75 | **1.01** | 0.91 |
| Pre-registered pick | 0.97 | 0.78 | 1.04 | 0.91 |

Clean windows (target months after each model's release, vs naive / vs naive_vol): Chronos-Bolt base 0.81 / 1.14 (n = 795) <!-- nda-ok: count of forecasts, not a price -->; Chronos-2 0.96 / 1.28 (n = 399, 12 target months only).

**Reading:**
- **naive_vol is the range model to beat, and no AI model beats it.** Against plain naive the AI ranges looked 15–20% better; against naive_vol they are 7–14% worse. Most of the earlier gain came from naive's fixed, shock-inflated bands.
- **Chronos-2 adds a little on the central forecast.** The quantile average with naive_vol is the best central forecast in the table (error 0.96 vs naive) and ties naive_vol on ranges (1.01). That is the honest "AI contribution" so far: small, and on the P50 only.
- **Caveat on the combinations:** the two naive_vol combinations were added after seeing that naive_vol had the best bands (the selection rule still only uses targets before 2024, but the candidate list was informed by the aggregate result). Report them as exploratory.
- **The pre-registered pick (0.97 / 1.04) is worse than simply using the quantile average.** Choosing per series × horizon on 4 years of history overfits; a single fixed rule is safer.
- **CQR dropped:** it widened the AI ranges too much (coverage 0.95+, range quality > 1 vs naive).
- **Compute:** about 4 s of inference for the full backtest; negligible energy and CO₂.

## 5. How to run

```bash
uv sync --group ai                                             # torch + chronos-forecasting
uv run --group ai python -m src.ai.run                         # backtest + live forecast (~15 s; needs the NDA data)
uv run --group ai --with pytest python -m pytest -q tests/test_ai_foundation.py   # synthetic-data tests
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
| `../../tests/test_ai_foundation.py` | No look-ahead (AI inputs and naive_vol), publication lags, CQR, selection (synthetic data) |

## 7. To-do list

### Phase 1: pretrained models ✅
- [x] Chronos-Bolt small/base and Chronos-2 (with drivers) running locally, versions pinned, Apache-2.0 confirmed on the model cards
- [x] Adapter to the team test bench (`FoundationModel.forecast`)
- [x] Backtest rows in the `src/backtest.py` format, scored with the same helpers
- [x] Perturbation audit on the real models (Chronos-Bolt, Chronos-2 with drivers)
- [x] Clean-window scoring after each model's release
- [x] Pre-registered selection (before 2024 → report 2024–26)
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

### Phase 2: agent-proposed features (needs `ANTHROPIC_API_KEY`)
- [ ] `guard.py`: NDA gate + prompt/response log + tests (refuses licensed series and deny-list terms)
- [ ] Formula-language spec (`dlog`, `ratio`, `spread`, `dev_ma`, `zscore`, `lag`) + compiler with publication lags
- [ ] Agent loop: public metadata in, pass/fail and a |ρ| bucket back, 2–3 rounds, cost cap
- [ ] Screening reused from notebook 03 on pre-2024 targets only
- [ ] Feed accepted features to dynreg and to Chronos-2 as drivers
- [ ] Evaluate on 2024–26 against the same models without the new features

### Phase 3: LLM news / regime signal (needs `ANTHROPIC_API_KEY`)
- [ ] GDELT headline fetch per material and month (2016 →)
- [ ] Baseline without an LLM: GDELT tone
- [ ] LLM tension score (−2…+2), date-free prompt, batch calls
- [ ] Commit the JSON cache (public-text only) so the run reproduces without a key
- [ ] Evaluate before and after the LLM's training cutoff separately
- [ ] Keep / reject rule: must beat both baselines after the cutoff

### Phase 4: LLM analyst forecast (needs `ANTHROPIC_API_KEY`)
- [ ] Prompt with public series only (+ news index + model forecasts)
- [ ] JSON P10/P50/P90 + short rationale per series and horizon
- [ ] Check on months after the cutoff only
- [ ] Separate, flagged `model_name` in the CSV
- [ ] Pre-registered inclusion rule (must not lose to naive after the cutoff)

### Phase 5: packaging and disclosure
- [ ] `docs/ai_track.md` (approach, licence table, leakage protocol, results, limits)
- [ ] Public results page in `public/ai/`, checked with `public/scan_leaks.py`
- [ ] Licence table and AI-tool disclosure in the main README
- [ ] `make ai` target
- [ ] GitHub Actions: run the synthetic AI tests
- [ ] Commit and open a PR from `ai-track` (only when agreed)

### Open questions
- [ ] Claude API key and budget cap for phases 2–4
- [ ] Team agreement on merging into `champion/v2`
- [ ] Which CSV the AI track submits: a separate file, or rows merged into the champion CSV
