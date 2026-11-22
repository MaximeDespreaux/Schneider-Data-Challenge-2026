# Pre-registration: graded backtest and margin-protection replay

Written before the graded run. Committed and tagged together with the code that runs it. No price values in this file.

## What is graded

- Graded origins run from 2022-01 to 2026-09. Targets go up to 2026-09 (latest vintage), and the window is scored once from the tagged commit. The command refuses to run without a tag, a clean tree and passing canaries for that commit.
- All tuning used the dev window: origins 2010-01 to 2021-12, scored only where the target month is before 2022-01.
- There are two modes:
  - submission: the live ragged edge of every input is mirrored at every origin;
  - realtime: publication lags come from `config/availability.yaml`.

## Candidates and the submitted model

- Base candidates:
  - `naive`, `naive_gauss`;
  - `naive_vol_hl3`, `naive_vol_hl6`, `naive_vol_hl12`;
  - `drift_conf`, `ar1_conf`, `ma12_conf`;
  - `driver_ridge`, `ets_damped`, `champion_wf` (signal screen re-run at every origin on realised targets only).
- Meta models (Strategy A), built from rows already realised at each origin:
  - `wf_best`;
  - `wf_blend`, with weights proportional to the trailing pinball loss to the power −`power`;
  - `wf_median`, over the shortlist `naive`, `naive_vol_hl3`, `ar1_conf`, `driver_ridge`, `champion_wf`.
- Meta settings:
  - trailing window 36 origin-months;
  - at least 12 realised rows, otherwise fall back to `naive`;
  - `power` = 4, chosen on the dev window among {1, 2, 4} by the lowest pooled dev pinball loss (0.984, 0.981 and 0.978 of naive).
- **Submitted model: `wf_blend`.** The others are reported next to it and never used to switch after grading.
- Targets:
  - CSV: Schneider's 14 series and the two BOM indices.
  - Scored internally, never in the CSV: the licensed MEPS/CRU steel prices L01–L03 and the epoxy resin price EP01.

## Metrics and pass rules

- The primary metric is pinball loss averaged over P10/P50/P90, in logs, relative to `naive` on the same rows. Also reported: relative MAE of the P50, coverage of P10–P90 (target 80%), and below/above rates.
- **Pass rule.** `wf_blend` beats naive in a group × horizon only if both hold:
  - the one-sided pooled Diebold-Mariano test is significant (average loss gap across the group's series at each origin, Newey-West with step − 1 lags, HLN correction);
  - the Holm-adjusted p < 0.05, in both modes.

  Groups are metals, steel, plastics, BOM indices, licensed steel and other. Anything else is reported as "not detectable" together with the minimum detectable effect (MDE) from placebo runs on the dev window (naive + noise, 20 seeds, power 80%, alpha 5%). The MDE table is kept with the confidential outputs (`data/outputs/backtest/dev/mde_submission.csv`, sha256 `e23120dc498e7b07a50a2fd6207a407cc2dd3546023cc5e41e1cb6774db1e0bb`); it ranges from about 2% to 11% of naive's loss depending on group and horizon.
- There are no per-series significance claims at h = 9, which has about 5 independent windows in the graded period.
- Calibration:
  - Kupiec test on the 80% band, on non-overlapping origins;
  - block-bootstrap 95% CI for the relative pinball loss.

## Margin-protection replay (Goal A)

- **Setup.** Each month t is a quote for one box at a fixed price, and the materials are bought at t + h.
  - The realised cost change is that of the product's BOM basket, with weights fixed at t.
  - Material share of price is 0.20 (config), with sensitivities at 0.40 and 0.68.
  - Target margins are 15% for RM6 and 10% for AirSeT; floors are 8% and 5%.
- **Policies:**
  - spot;
  - P50 or P90 of the `wf_blend` forecast;
  - historical P90 (naive);
  - two-way index clause at ±3, ±5 and ±10%;
  - copper bought forward at LME 3M;
  - copper forward plus a ±5% clause.
- **Decision rule, fixed in code before any graded number was seen (commit c2ef724).** Among policies with a breach rate of at most 5% and a 5%-worst margin at or above the floor, pick the lowest mean markup, with ties going to fewer price revisions. The rule is applied on the dev window only; the graded window is reported as it comes.

## Prior looks, declared

- Julia's bench was inspected on 2024–26 when choosing the naive_vol half-life, hist_floor and the recenter test. Her champion rule was chosen with a global screen on targets before 2024, and the champion v1/v2 choice used the 2024–26 holdout.
- The teammates' EDA correlations and Report 3's feature selection used the full sample. These inform which candidates exist, not their parameters. `champion_wf` re-screens walk-forward.
- The margin replay above was first run on 2026-11-22, with the decision rule already committed.
- Dev-window results (both modes) were inspected on 2026-11-22 to choose `power`; nothing else was tuned on them.

## Declared limits

- Truth is the latest vintage; there are no first releases.
- The BOM is v2 (Schneider, about 20 Nov). Material shares are provisional (Schneider is checking), and "Other" (about a tenth of AirSeT's mass) cannot be priced.
- Plastics and electronics are valued at the config EUR/kg assumption at every origin.
- The epoxy May-2026 ECH typo is corrected ×10; the resin price is left as delivered.
- `driver_ridge` alpha was hand-set before this work.
- The copper forward is the LME 3M/cash spread scaled to h/3.
- Selling prices, margins and material share are assumptions until Schneider confirms them.

## Amendment before grading (2026-11-22, freeze v2)

- **Band calibration tested and not adopted.** A walk-forward conformal shift of each tail (`calibration.py`) was evaluated on the dev window. The rule was written in the ledger before the evaluation: submit `wf_blend_cal` only if its pooled dev pinball loss is lower than `wf_blend`'s. It was worse (1.017 against 0.978 of naive), and coverage dropped at 9 months. The submitted model stays `wf_blend`.
- **Fixes from the independent code review**, which found no look-ahead path:
  - In submission mode, the licensed and extra inputs (L01–L03, EP01, EL01) now keep their live publication gap at every origin. Only Schneider's CSV series are exempt.
  - The copper forward in the margin replay is a USD forward, so the EUR/USD move stays with Schneider.
  - The replay reports the effective markup including clause revisions.
  - The graded run always scores both modes, and the report refuses a single mode.
  - `driver_ridge` no longer loses its whole training sample when a driver column has no value yet (licensed steel before 2012). This changes a candidate after the dev choice of `power`; the power choice is not revisited.
  - The bootstrap block length is h + publication lag.
- **Declared limits added:**
  - In realtime mode, a masked BOM component is carried flat to the origin, which makes BOM-index bands slightly narrow (conservative, not leaky).
  - The Kupiec test pools hits across a group's series, so its p-values are too confident; it is reported only.
  - The canaries run on the assembled panel (identity `prepare`), so panel construction itself is covered by the truncation tests of the loaders and the BOM deriver, not by the canaries.

## Amendment 2 before grading (2026-11-22, freeze v3): submitted model

- **Submitted model: `champion_wf2`, not `wf_blend`.** This is the user's decision, taken on dev-window evidence before any graded number. The walk-forward champion beat the blend on most groups at 3 months; the blend diluted it with weaker models. `wf_blend`, `wf_best` and `wf_median` are still computed and reported, never used to switch after grading.
- **`champion_wf2` = `champion_wf` with extra candidate signals for its walk-forward screen:**
  - the licensed market price vs index gap (and its deviation from the 24-month mean), Report 3's strongest signal;
  - regime-gated momentum (3-month change while not overextended) and mean reversion (distance to the 24-month mean once overextended).

  The rule was written in the ledger before the evaluation: adopt v2 if its pooled dev pinball loss is lower than v1's. It was, by a negligible margin (0.95874 against 0.95884 of naive, 7,851 common forecasts), so in substance this is a tie.
- In the candidate set, `champion_wf2` replaces `champion_wf` (runtime), and the `wf_median` shortlist uses `champion_wf2`. The pass rule, metrics and windows are unchanged and now apply to `champion_wf2`.
- An exploratory regime model (momentum vs mean reversion with gates) was tried on the dev window. It is not a candidate: it trails the champion on most index-type series. It lives on branch `wip/regime`, not here.
