# Business modules (Goal A pricing, Goal B, decarbonisation)

> This page describes code only. It contains no licensed prices and no challenge data. All outputs are written to
> `data/outputs/business/` (gitignored); figures to `data/outputs/business/figures/` (PNG, 150 dpi).

Settings live in two files:
- `config/business.yaml` (tracked, no licensed values): seed, origin, horizons, shock sizes, lag pairs, electronics
  slider, competitor groups, TCO ranges with sources, supplier peer settings and the supplier list.
- `config/assumptions.yaml` (local, gitignored): prices, margins, margin floors, quote rules, competitor and customer
  levers. The modules read these values; none are hard-coded.

## Commands

```bash
uv run python -m src.business.run_all          # every module, in order, + REPORT.md and compute footprint
uv run python -m src.business.<module>         # one module, e.g. src.business.tco
uv run pytest -q                                # tests (synthetic data only, run in CI)
```

`run_all` runs: forecasts, convergence, shocks, lags, eaton, electronics, pricing_levers, tco, suppliers. A module
that fails is marked FAILED and the others still run. It writes `REPORT.md` (status, runtime and headline figures of
each module; EUR per box, % and kg only) and `compute_footprint.csv`.

## 1. Forecasts at the origin (`forecasts.py`)
- **Question:** what are the P10 / P50 / P90 of the BOM indices and of the material prices 3, 6 and 9 months after
  the origin (2026-09)?
- **Inputs:** the panel (`src.testbench.run.build_panel`), the BOM deriver.
- **Method:** uses the champion forecasts if `data/outputs/external/champion_holdout.parquet` has them for every
  series at the origin; otherwise runs ONE single-origin backtest of `NaiveVolScaled`. The result is cached.
  `last_value` is the last known value at the origin (100 for the BOM indices). `pct_change(df, series, h, q)` gives
  `p_q / last_value − 1`.
- **Outputs:** `forecasts_origin_202609.parquet`.
- **Cache:** reused unless `champion_holdout.parquet` is newer than the cache, or the cache holds naive_vol forecasts
  while usable champion forecasts now exist; the console says which source is used and why. Delete the cache to
  force a recompute.
- **Caveats:** stainless steel (L02) is not yet published for the origin month, so
  the BOM index's last full month is one month earlier; its P50 is therefore slightly below 100.

## 2. Convergence (`convergence.py`)
- **Question:** how did the raw-material cost of one RM6 and one AirSeT evolve in EUR, and how exposed is each to
  copper?
- **Inputs:** `bom_weights()` (BOM xlsx_v2, global config: electronics excluded, like the submitted index), the panel
  (licensed steel included), the Task 1 forecasts.
- **Method:** constant quantities: cost(t) = Σ kg × EUR/kg at base × P(t) / P(base). A series not yet published at the
  origin carries its last value forward up to the origin month only (column `carried_to_base` says which); any other
  missing month stays empty. Forward view: base cost × BOM index P10/P50/P90 / 100.
- **Electronics:** not in the main lines. Shown as a flat sensitivity: a shaded band on the chart and extra summary
  columns with the cost per box plus the low / mid / high EUR per box of the electronics slider.
- **Outputs:** `convergence_monthly.csv`, `convergence_summary.csv`, `convergence_cost_per_box.png`,
  `convergence_value_shares.png`.

## 3a. Shock table (`shocks.py`)
- **Question:** if material X moves by Y %, how many EUR more per box, and how many margin points?
- **Inputs:** `bom_weights()`, selling prices, `quote_rules.indexation_trigger_pct`, `material_share_of_price`.
- **Method:** delta = base value of the material (or combo) × shock; also as % of material cost, % of price, margin
  points, and whether it crosses the indexation trigger. Repeated with the top-down cost base (share of price,
  columns `_ms`). Materials absent from a product's BOM (e.g. electronics, brass in AirSeT) are skipped.
- **Outputs:** `shock_table.csv` (long, with the lag columns of 3b), `shock_table_wide.csv`, `shock_copper_steel.png`.

## 3b. Pass-through lag (`lags.py`)
- **Question:** how many months after a market move does it show up in the slower price?
- **Inputs:** the long-history panel (falls back to the base panel if public downloads are missing); epoxy file.
- **Method:** on monthly log changes, cross-correlations corr(dy_t, dx_{t−k}) for k = 0..6 and a distributed-lag
  regression dy_t = a + Σ b_k dx_{t−k} (numpy least squares). Cumulative pass-through C_k = Σ b_j; `lag_50` / `lag_80`
  = first month where C_k reaches 50 % / 80 % of the total. Full overlap, before 2024 and from 2024. Epoxy: EP01
  against each of its ingredients.
- **How to read it:** our BOM cost index uses market prices, so in the model a shock hits immediately. The lag table
  tells sales how long real purchase prices — which often follow producer-price indices or contract resets — take
  to catch up, i.e. how long a quote stays protected after a market move.
- **Epoxy:** ingredient pass-through into epoxy is weak (total pass-through C_max < 0.45 for benzene, acetone and
  epichlorohydrin), so the epoxy pairs stay in `lag_table.csv` but are left out of the figure, the REPORT and the slides.
- **Outputs:** `lag_table.csv`, `lag_cumulative_passthrough.png`; `lag_50` / `lag_80` joined onto `shock_table.csv`
  (copper → copper pair; carbon steel and all steel → MEPS CR coil pair; galvanised → HDG pair; stainless → nickel
  pair; others empty).

## 4. Eaton price gap (`eaton.py`)
- **Question:** is the SF6-free brand pricing differently from the SF6 brands?
- **Inputs:** the RMU price benchmark (yearly P50 per brand) and the competitor teardowns.
- **Method:** each brand's P50 relative to Schneider's and its year-on-year change; a brand is SF6-free if its
  teardown has no SF6 line above 0.1 kg (broken teardowns: unknown). Eaton vs the average of the SF6 brands; first
  year Eaton's P50 falls below Schneider's.
- **Outputs:** `eaton_price_gap.csv`, `eaton_vs_sf6_brands.png`.
- **Caveat:** the benchmark is synthetic, and Eaton's 2026 move may also reflect its product mix — a signal, not proof.

## 5. Electronics slider (`electronics.py`)
- **Question:** how much would electronics weigh in the BOM index at different EUR per box?
- **Method:** the only module that sets `extras.electronics.include = true`, on a copy of the assumptions. For each
  EUR per box: total BOM value, electronics value share, cost change for EL01 +10 %, copper share of the index, and
  the BOM index P90 change at 6 months recomputed with these weights (brass = 63 % copper + 37 % zinc changes;
  materials without a forecast count as no change).
- **Outputs:** `electronics_slider.csv`, `electronics_slider.png`.
- **Caveat:** adding up marginal P90s overstates the joint P90. Electronics is not in Schneider's BOM and stays out
  of the submitted index.

## 6. Pricing levers (`pricing_levers.py`)
- **Question:** for a given deal, by how much should the price move, and when should we index?
- **6a Margin at risk:** per product × horizon × scenario, cost change of the BOM index applied to both cost bases
  (priced BOM value, or share of price); margin points lost, margin after, below floor.
- **6b Base quote rules:** price adjustment (P50), risk buffer (P90 − P50 in EUR per box), band width
  (P90 − P10) / P50, validity (narrow / wide band), indexation recommended, trigger.
- **6c Competitor lever (RM6):** teardown masses × EUR/kg at the origin give each brand's value shares; the brand's
  cost change = Σ share × material change, for each scenario (P10, P50, P90). Teardowns underestimate copper, so the
  gap is measured **competitor change − Schneider-teardown change** (same method on both sides), never against our
  official BOM. The band and pass-through use the gap of `competitor_lever.band_scenario` in `config/business.yaml`
  (default P90, the risk scenario: with a naive forecast the P50 change is zero by construction). A backward-looking
  check applies the same weights to realised prices over the 12 months to the origin (`gap_realised_12m_pp`). The
  strategy switch picks full pass-through (protect margin) or the gain-share pass-through when competitors are hit
  harder.
- **6d Customer lever:** margin points we may give up = tier allowance + deal-size extra.
- **6e `recommend(...)`:** applies the levers in `combination_order`: cost delta → competitor pass-through → customer
  give-up → margin floor → indexation check, with the competitor band from `band_scenario`. The full grid (both cost
  bases, column `cost_basis`) is in `pricing_recommendations.csv`; bars at 0 % in the example figure mean the
  customer give-up absorbed the whole increase.
- **Outputs:** `margin_at_risk.csv`, `quote_rules.csv`, `competitor_lever.csv`, `competitor_value_shares.csv`,
  `pricing_recommendations.csv`, `pricing_levers_example.png`.
- **How it fits with the pricing-options backtest:** Julia's pricing-options backtest
  (`src/proposals/part4_pricing_options.py`) picks *which rule* protects margin on past data; these levers adjust the
  price *for a given deal*. They meet only in the final recommendation.
- **Caveats:** the competitor average at P90 is sensitive to one brand's mix (e.g. a teardown with a large epoxy
  share reacts strongly to the wide epoxy P90); look at the brand rows, not only the average. The bottom-up
  (`bom_value`) and top-down (`material_share`) cost bases give very different margin impacts; both are reported.
  The AirSeT price is assumed (RM6 + premium) until Schneider
  shares it.

## 7. TCO Monte Carlo (`tco.py`)
- **Question:** over its life, is an SF6-free AirSeT cheaper to own than an SF6 RM6, and at what price premium do
  they break even?
- **Inputs:** `tco` block of `config/business.yaml` (each range with its source), RM6 price as capex.
- **Method:** 10,000 vectorised draws (fixed seed): SF6 leakage and end-of-life losses in t CO2e, their carbon cost,
  maintenance and SF6 admin, end-of-life recovery, all discounted. Break-even premium = premium at which both TCOs
  are equal. One-at-a-time tornado around the mid values.
- **Outputs:** `tco_draws.parquet`, `tco_summary.csv` (overall and by carbon price), `tco_tornado.csv`,
  `tco_diff_hist.png`, `tco_tornado.png`.
- **Caveat:** several inputs are assumptions (premium, maintenance, discount rate, end-of-life loss); the tornado
  shows which ones matter.

## 8. Supplier emissions (`suppliers.py`)
- **Question:** roughly how much do our suppliers emit (Scope 1+2, and Scope 3 purchased goods)?
- **Inputs:** the raw PitchBook files in `data/raw/suppliers/` (revenue is parsed, never typed) and the supplier list
  in `config/business.yaml` (country / sector with their confirmation status).
- **Revenue rules:** (a) Overview "Total Revenue", newest period; else (b) Income Statement "Revenue" (or Business +
  Other Revenue), newest period; else (c) total assets × asset-turnover proxy, flagged as PROXY. Amounts must be in
  thousands of EUR. File names are matched to `file_keyword` ignoring punctuation and accents.
- **Method:** a synthetic peer pool (sector, country, size; electricity and gas use; Scope 3 reported by 40 %); the
  estimate takes the 15 peers closest in size in the same sector and country (sector only if fewer than 5), and
  scales their intensity by revenue (P10 / P50 / P90). Leave-one-out validation. Spend-based total as a cross-check.
- **Outputs:** `supplier_financials.csv`, `peers_synthetic.csv`, `supplier_validation.csv`,
  `supplier_estimates.csv`, `supplier_estimates.png`.
- **Three rules:** (a) peer S1+S2 and spend-based figures are never added — spend-based already includes upstream;
  (b) the S3 estimate stacks estimates on estimates (peers' own S3 is often estimated), so its band is wide and it is
  a demo of the pipeline; (c) all figures are for the whole supplier, not Schneider's share, which needs Schneider's
  spend per supplier.

## Caveats to say on the slides
1. **Convergence is moderate.** Both products' material costs are driven up by copper, RM6 only slightly faster. Frame
   it as RM6 being more exposed to copper, not as "the gap is closing fast".
2. **Teardowns underestimate copper** (Schneider's own teardown: [NDA] kg vs 0 kg in the official BOM), so
   competitors are compared relative to Schneider's teardown, never in absolute terms.
3. **Bands during trends.** During the 2025–26 rally the volatility-scaled bands were too narrow in a short check
   (50–57 % coverage at 6 months for the BOM indices vs the 80 % target), so calibration during trends is a known
   limitation and Julia's full backtest is the reference.
