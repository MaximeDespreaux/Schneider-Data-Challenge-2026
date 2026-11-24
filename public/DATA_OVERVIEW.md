# Data overview & first analysis (public version)

*Schneider Electric × X-HEC Data Challenge 2026 · data snapshot of 16-Nov-2026, BOM update of 22-Nov-2026*
*Public, NDA-safe copy. Licensed CRU / MEPS / epoxy prices and Schneider-internal figures (BOM masses, € product costs, should-costs, part-mapping counts, supplier names and financials) are removed or shown only as indices, % changes or statistics. Companion notebooks: [`notebooks/`](notebooks/). Figures: [`reports/figures/`](reports/figures/).*

---

## 1. Executive summary

**The task.** Forecast the raw-material (RM) prices that drive two medium-voltage Ring Main Units at horizons of 3, 6 and 9 months. Then turn the forecasts into a sourcing or business recommendation. The two products:

- **RM AirSeT**. Mostly carbon and stainless steel.
- **RM6 NE-IQI**. Steel plus copper.

**What we have.**

- A clean, well-documented core panel: 15 raw-material series and 30 energy series, monthly from Nov-2016 to Oct-2026.
- A Schneider BOM. A **latest version (v1)** was received on 22-Nov-2026 and is now the reference.
- Four "bonus" sources that are more valuable than they look:
  1. a **MEPS/CRU steel-price workbook**, which fills the licensed gap the organisers left on purpose;
  2. an LME copper and aluminium workbook;
  3. an epoxy cost build-up;
  4. a competitor should-cost teardown.
- Financial statements for 9 suppliers.

**Six things a modeller must know before writing a line of code:**

1. **The licensed gap can be closed.** The core set leaves L01–L08 (CR coil, 304L, HDG, plastics, Rotterdam premia) empty. The steel workbook provides real CRU and MEPS prices for CR, HDG and HR coil and for 304 stainless, by country. Our free proxy for steel is the Eurostat PPI (S09/S10). It is a **smoothed, roughly 1-month-lagged** version of the market price:
   - correlation of monthly changes is 0.56 at lag 0 and 0.71 at lag 1;
   - it overshoots after shocks: rebased to Nov-2016 = 100, the PPI still sits 13–21 points above the CRU price in Sep-2026.

   **Recommendation:** forecast the CRU/MEPS series as targets, and use the PPI as a feature, not as the target.
2. **The two families of series behave differently.**
   - Producer-price indices (S08–S11) are sticky. The lag-1 autocorrelation of their monthly changes is 0.67–0.79, so short-horizon forecasting is easy. Naive MAPE at 3 months is about 4.5%.
   - Exchange-traded metals (copper, aluminium, nickel, iron ore) are close to random walks: lag-1 autocorrelation is 0.1–0.3, and naive MAPE at 3 months is 7–14%.

   **Model them separately.**
3. **Everything is I(1), nothing is seasonal.** ADF tests do not reject a unit root for any log price level. Log-differences are stationary. STL seasonal strength is ≤ 0.16 for all series, and seasonal-naive is the worst baseline everywhere. **Model log-returns. Do not spend effort on seasonality.**
4. **Copper is the risk to watch.** Copper in EUR is at an **all-time high** (Sep-2026, +36% YoY, +149% since 2016, public LME/IMF data). In RM6 it is about **60% of the priced material cost and 40% of its variance**. In the data.zip BOM the AirSeT has no copper and its exposure is split evenly between carbon and stainless steel. The latest BOM (v1) adds a small copper share to the AirSeT.
5. **Forecasting reality check** (§6). Traded prices are close to efficient: nothing beats the random walk by more than about 5%. The Eurostat PPIs, however, *are* forecastable from the market prices that lead them, with 30–38% lower error. The value lies in calibrated risk ranges and in index-linked contract design.
6. **Data-quality traps.** Each needs handling before modelling:
   - several BOM versions (now resolved: v1 is the reference);
   - the RM6 carbon-steel mass was missing in the first versions;
   - one competitor's should-cost is clearly broken;
   - the supplier data provider's "Debt to Equity" ratio is 0 for almost every supplier;
   - the price benchmark file was DRM-locked at first; an unlocked copy arrived on 18 Nov 2026.

---

## 2. File-by-file review

### 2.1 `data.zip`: core challenge set (★★★★★ primary input)
Extracted to `data/raw/data/`. Loaders: `load_rm_wide`, `load_energy_wide`, `load_bom`, `load_catalogs` in [`src/data_loading.py`](../src/data_loading.py).

| File | Grain | Content |
|---|---|---|
| `rm_prices_monthly_wide.csv` / `_long.csv` | month × series | **S01–S13** free proxies from IMF and FRED (LME Cu, Al, Ni, Zn, iron ore), ECB (EUR/USD), Eurostat PPIs (copper, aluminium, steel, galvanised, plastics) and US BLS PPIs. **D01/D02** are Cu and Al converted to EUR. **L01–L08** are licensed series and are 100% empty. |
| `energy_prices_monthly_wide.csv` / `_long.csv` | month × series | **E01–E24**: Brent, WTI, TTF, Henry Hub, Asia LNG, coal, EU wholesale power (DE/FR/HU), JEPX, energy PPIs, Russian crude flows and FX. **ED01–ED06** are the same data harmonised to EUR/MWh or EUR/bbl. |
| `bom_materials.csv` | product × material | Material share, mass and mapping to price series for both RMUs (superseded by BOM v1). |
| `source_catalog.csv`, `energy_source_catalog.csv`, `data_dictionary.csv` | series | Provenance, units, URLs and field rules. |
| Excel masters | — | Redundant with the CSVs. |

**Quality.**
- 120 months, Nov-2016 to Oct-2026, with **no internal gaps**. All missing values are at the ragged right edge (publication lags): 32 series end in Sep-2026, 10 in Oct-2026 (FX, power, JEPX), 3 in Jul-2026 (Eurostat crude imports).
- So the "imputation challenge" is really a **nowcasting** problem: fill the last 1–3 months.
- D01/D02 match the stand-alone LME workbook to within 0.06% on average (max gap 1.1%).

**Caveats.** S09 is a single PPI used for both carbon and stainless steel, which is too coarse for stainless. S12 and S13 are US indices. Russia is represented only by import volumes.

### 2.2 Steel index workbook: MEPS & CRU steel prices (★★★★★, licensed)
Real market prices from the Schneider procurement team. Loaders: `load_meps`, `load_cru_flat`, `load_cru_stainless`, `load_steel_mapping`.

| Sheet | What | Coverage |
|---|---|---|
| `MEPS` | Flat coil (HR, CR, HDG, EG) prices for DE, FR, IT, ES, UK, Belgium, plus Nordic/Sweden fragments | 2012 → 2026-10 |
| `CRU Flat` | CRU weekly and monthly-average flat coil prices by country | 2012 → 2026-10/11 |
| `CRU Long` | Beams, merchant bar, rebar, wire rod | 1997 → |
| `CRU Stainless` | 304, 430 and 316 CR-2mm base price + **alloy surcharge** | 2008 → 2026 |
| `F▲`, `L▲`, `SS▲` sheets | Month-on-month and quarter-on-quarter changes used in price-adjustment clauses | 2016 → |
| `Reference to Index Mapping` | Schneider part references across several plants, each mapped to a country × grade index, with a **1-month lead time** in every clause | — |
| Snapshot sheets | Historical negotiation and allocation records (not used) | — |

**Findings.**
- MEPS and CRU are interchangeable: CR coil DE correlates 1.00 in levels and 0.94 in monthly changes.
- The Eurostat PPIs trail the market by about one month and are heavily smoothed:

  | Market price | Proxy | Corr. of Δlog at lag 0 | Best lag | Corr. at best lag |
  |---|---|---|---|---|
  | CRU CR DE | S09 | 0.56 | market leads by 1 month | 0.71 |
  | CRU HDG DE | S10 | 0.56 | market leads by 1 month | 0.62 |
  | CRU SS304 alloy surcharge | S03 nickel | 0.03 | nickel leads by 1 month | 0.66 |
  | CRU CR DE | S04 iron ore | 0.27 | iron ore leads by 1 month | 0.43 |

- **Most useful mapping:** most references are indexed on HDG Benelux, CR France, CR Spain and HDG Germany. These are the series to forecast to speak the purchasing team's language.

**Caveats.** Custom parser needed (multi-row headers, separators, placeholders up to 2028). Licensed data: never publish raw values.

### 2.3 LME copper & aluminium workbook (★★★☆☆)
Cash and 3-month prices, cash–3M spreads, LME stocks and FX, Jan-2016 → Nov-2026 month-to-date (provisional). It mostly duplicates D01/D02, but adds one extra (partial) month for nowcasting, **the futures curve** (contango/backwardation) and **inventories**: classic predictors for metal returns.

### 2.4 Epoxy cost build-up (★★★☆☆)
Monthly benzene, acetone, epichlorohydrin (ECH), "other costs" and epoxy resin prices, Jan-2018 → Oct-2026. Epoxy is a quasi-deterministic cost build-up: a linear "recipe" fit has **R² = 0.976**. **Forecast the feedstocks, then derive the resin price.** Epoxy is in the latest BOM (v1) for both products.

### 2.5 Competitor teardown should-cost (★★★☆☆)
A teardown should-cost of the **Schneider RM6 3-function module** against Ormazabal, SGC, ABB, Siemens and Eaton, marked *provisional*.

| Product | Material share of should-cost | Comment |
|---|---|---|
| Schneider RM6 (ref.) | 80% | Lightest unit |
| ABB SafePlus | 82% | |
| Siemens 8DJH | 81% | Single web list price found |
| Ormazabal CGMcosmos | 86% | |
| Eaton Xiria | 84% | **SF6-free** (air/vacuum); the cost looks low |
| SGC DR-6 | — | **Broken** (total orders of magnitude too low): exclude |

- Masses are mostly inferred. Treat as order-of-magnitude.
- **Use:** materials are about 80% of should-cost in every unit, so RM forecasts translate almost one-for-one into cost competitiveness.

### 2.6 Supplier financials (★★☆☆☆)
Nine suppliers, statements in kEUR. Names and figures are confidential; anonymised overlay (A = largest by revenue):

| Supplier | EBITDA margin | Flag |
|---|---|---|
| A | thin (< 5%) | ⚠️ highly leveraged, interest not covered |
| B, D, F | moderate (5–12%) | F: P&L out of date |
| C, E, G | healthy (≥ 12%) | G: current ratio below 1, single year |
| H | negative | ⚠️ negative EBITDA, current ratio below 1 |
| I | — | Balance sheet only |

**The provider's own "Debt to Equity" ratio is 0 almost everywhere: do not use it.** Recompute as Total Debt / Total Equity; a debt of 0 may mean "not reported". **Use:** a supplier-risk overlay for the recommendation. Coverage is too uneven for statistical modelling.

### 2.7 BOM versions (⚠️ resolved)
- `data.zip` BOM and the first root workbook disagreed on AirSeT masses (carbon steel, stainless, "Other"), and both lacked the RM6 carbon-steel mass.
- **BOM v1 (22-Nov-2026) is the reference** (organiser guidance). Both products are complete: it gives the RM6 carbon-steel mass, adds copper and epoxy to the AirSeT, and adds epoxy, brass and PP to the RM6. Discrepancies with earlier versions are documented as a data-quality assumption.
- "Other" / unidentified material stays non-priceable; run a sensitivity scenario if it matters. Brass has no series: priced as 65% copper + 35% zinc.

### 2.8 RMU IQI EU price benchmark (✅ unlocked copy, 18 Nov 2026)
- Yearly P10 / P50 / P90 selling prices (EUR ex-VAT, labelled "AI/ML", **synthetic**) for 6 brands, 2017–2026. No monthly series, no 1F–4F configurations.
- **2026 jump:** SF6 units +32–40% (Schneider +40%); Eaton, the only SF6-free unit, +8%. This lines up with the EU SF6 ban for new switchgear up to 24 kV from 2026.
- **Material share of price:** RM6 priced material is 6–9% of the Schneider median price every year (6.6% in 2026); the teardown should-cost is about 25%. A 20% copper move is about 0.8% of price.

### 2.9 Kick-off presentation
- ETO contracts lock prices for 3, 6 or 9 months; RM AirSeT has a higher cost structure and thinner margin.
- Scope: forecast **all** raw-material inputs (copper, carbon steel, insulating materials = epoxy, electronics: no series given) and carry them into **2027 RMU unit pricing**.
- Non-regulated markets for question 3: APAC, Middle East, Americas. Forecast accuracy and the non-regulated-market strategy count equally.
- It contains personal data: never commit it.

---

## 3. Key EDA findings

| # | Finding | Evidence | Implication |
|---|---|---|---|
| 1 | Prices roughly doubled in the 2021-22 shock, then partly reverted. Copper has since broken out to new highs. | Fig. [02](reports/figures/02_rm_indexed.png) | Regime changes: use robust losses and include the shock in the training window. |
| 2 | Volatility hierarchy: iron ore 32% > nickel 25% > zinc 20% > aluminium 16% > copper 14% annualised; EU PPIs 6–7% | notebook 01 §3 | Uncertainty bands must be series-specific. |
| 3 | All series are I(1) in logs; none is seasonal | notebook 01 §4 | Model Δlog. Skip seasonal terms. |
| 4 | PPIs have strong momentum (AC1 0.67–0.79); LME metals don't (0.1–0.3) | Fig. [04](reports/figures/04_momentum.png) | ARIMA/ETS should beat naive on PPIs; metals need exogenous drivers. |
| 5 | Oil leads plastics (Brent → S11, lag 1, r = 0.60). Energy drives aluminium (lag 0, r = 0.44). Gas and power lead steel only weakly. | Fig. [06](reports/figures/06_energy_ccf.png) | Lagged energy features: strongest for plastics and aluminium. |
| 6 | Steel market prices (CRU/MEPS) lead the PPI by about 1 month and are ~2.5× more volatile | Fig. [07](reports/figures/07_steel_market_vs_proxy.png) | Use CRU/MEPS as targets. |
| 7 | RM6 priced material cost doubled from Jun-20 to today and is ~12% below its Apr-22 peak, because of copper. AirSeT is +56% vs Jun-20 and ~41% below its peak (data.zip BOM). | Fig. [08](reports/figures/08_bom_cost_index.png) (cost index, latest BOM) | RM6 is exposed to copper *now*; AirSeT mostly to steel. |

### First baselines: the bar to beat
Naive (random-walk) MAPE (%) from an expanding-window backtest with monthly origins from Jan-2020 onward. Seasonal-naive is worse everywhere (11–25%), and drift is no better than naive.

| Series | h = 3 | h = 6 | h = 9 |
|---|---|---|---|
| S11 EU plastics PPI | 4.3 | 7.4 | 9.9 |
| S10 EU galvanised PPI | 4.5 | 8.5 | 12.1 |
| S09 EU steel PPI | 4.7 | 8.7 | 12.1 |
| D01 Copper EUR | 7.1 | 10.5 | 12.9 |
| D02 Aluminium EUR | 8.9 | 12.6 | 15.3 |
| S05 Zinc | 9.0 | 13.3 | 17.0 |
| CRU SS304 DE | 10.0 | 16.2 | 21.5 |
| S03 Nickel | 11.2 | 15.0 | 17.1 |
| CRU HDG DE | 11.4 | 15.8 | 16.9 |
| CRU CR DE | 12.0 | 16.4 | 17.8 |
| S04 Iron ore | 14.4 | 19.8 | 19.1 |

---

## 4. Risks & open questions

1. **BOM:** v1 is the reference; earlier versions are documented as a data-quality assumption. "Other" is non-priceable: sensitivity scenario.
2. **Ragged edge:** 1–3 trailing months are missing per series. Nowcast with the LME month-to-date, FX and power to Oct-2026; flag provisional values.
3. **Licensing / NDA:** CRU and MEPS data is licensed and the Schneider files are under NDA. Never publish raw values (see [`README.md`](README.md)).
4. **Plastics:** there is no EUR/t price. Our cost model assumes a 2021 price moved with the S11 PPI. Replace with a real quote if available.
5. **The benchmark is provisional.** Exclude SGC, treat Eaton with caution.
6. **Cost overviews:** ask for the two documents cited in the kick-off deck.

---

## 5. Recommended roadmap

1. **Targets.** CRU CR and HDG (DE/FR, matching the mapping sheet), CRU SS304 (base and alloy separately), D01 copper, D02 aluminium, epoxy, S11 plastics; roll them up into a **cost-weighted BOM cost index per product** (price × mass, one stated base date).
2. **Features.** Lagged own Δlog (momentum); lagged energy (Brent, TTF, DE power, E13); nickel → SS304 surcharge (lag 1); iron ore → CR coil (lag 1); EUR/USD; LME cash–3M spread and stocks; PPIs as slow-moving anchors.
3. **Models**, in increasing complexity: ARIMA/ETS on Δlog → ARIMAX / dynamic regression with lagged drivers → gradient boosting with direct multi-horizon training → a probabilistic layer (quantile regression or conformal intervals).
4. **Validation.** Expanding-window backtest, MAPE and directional accuracy at h = 3/6/9, shock window shown separately.
5. **The business layer.** Hedging/timing (copper for RM6), should-cost competitiveness (§2.5), supplier-risk overlay (§2.6), and the offer price reference (LME cash vs 3-month) for 3/6/9-month quotes.

---

## 6. Modelling results: which forecasting approach is worth pursuing?

*Source: [`notebooks/02_forecasting_benchmark.ipynb`](notebooks/02_forecasting_benchmark.ipynb). Code: [`src/features.py`](../src/features.py), [`src/models.py`](../src/models.py), [`src/backtest.py`](../src/backtest.py).*

**Setup.** 12 targets; the cumulative log-change for each horizon (h = 3/6/9), one model per horizon; monthly origins Jan-2020 → Jul-2026 (~21,600 forecasts); every driver lagged by its publication delay, with an automated no-leakage test.

### 6.1 Leaderboard: relative MAE vs naive (< 1 = better), mean over the 12 series

| Model | h = 3 | h = 6 | h = 9 |
|---|---|---|---|
| **Ensemble** (ETS, ARIMA, Theta, dynamic regression, LightGBM) | **0.95** | 1.01 | 1.04 |
| ARIMA(1,1,0) + drift | 0.96 | 1.03 | 1.06 |
| Naive (random walk) | 1.00 | 1.00 | 1.00 |
| Global LightGBM | 1.01 | 1.10 | 1.33 |
| Dynamic regression (ridge) | 1.01 | 1.07 | 1.05 |
| Theta / drift / ETS damped | 1.01–1.02 | 1.02–1.09 | 1.04–1.10 |

Per-series results: Fig. [12](reports/figures/12_leaderboard_heatmap.png).

### 6.2 What the backtest says

1. **Market prices are close to efficient.** No model beats naive by more than about 5%, and only at h = 3. The LME 3-month futures price does no better (relative MAE 0.995 copper, 0.966 aluminium).
2. **The Eurostat PPIs are forecastable** because market prices lead them. Dynamic regression on S09/S10: relative MAE **0.62–0.69 at h = 3** and **0.66–0.69 at h = 9**; out of sample 0.64–0.76. Diebold–Mariano p-values 0.05–0.25: consistent, not significant at 5%.
3. **Models help in trending regimes and hurt in calm ones** (ensemble 0.91 at h = 3 in 2021-22; 1.15 at h = 9 in calm periods).
4. **Global LightGBM overfits** (1.33 at h = 9; bands under-cover at 56–64% vs 80%).
5. **The uncertainty is the more valuable output.** Ensemble conformal intervals cover 77% / 74% / 71% of the nominal 80% band at h = 3/6/9 (Fig. [13](reports/figures/13_interval_coverage.png)); today's bands are wide and skewed upward (Fig. [15](reports/figures/15_forecast_fans_pct.png)).
6. **Business test: a forward-buy rule** (buy ahead when the h = 3 forecast is above +2%, carry 0.4%/month):

   | Price paid | Rule saving vs always-spot | Perfect-foresight saving | Share captured |
   |---|---|---|---|
   | S09 / S10 (PPI-indexed) | 1.4–2.1% | 2.5–2.7% | **~55–80%** |
   | CRU CR, SS304, copper (market) | −0.3 to +1.4% | 4.4–6.6% | ≤ ~20% |

**BOM view** (joint, correlation-preserving simulation; change in priced material cost from today; data.zip BOM, to be re-run with v1):

| Product | P50 at 9 months | P90 at 9 months |
|---|---|---|
| RM6 | +4% | +54% |
| AirSeT | +3% | +58% |

The upside tail is far larger than the drift: the business case is **risk sizing and contract design**, not point prediction.

### 6.3 Recommended approach going forward

1. **The right tool per family.** Market prices: naive or futures as the point forecast, ARIMA momentum only at h = 3, effort on calibrated intervals and scenarios. PPI-indexed contracts: dynamic regression on leading market prices.
2. **Next explorations:** regime-aware models; more forward-looking data (full futures curves, steel futures); longer public history; a zero-shot time-series foundation model; conditional conformal intervals.
3. **Frame the recommendation around risk:** P90 cost exposure per product; hedge or forward-buy policies for copper (RM6) and steel (AirSeT); indexation clauses where forecasting works (PPI-based).

---

## Reproducing

```bash
make install
# private, with the NDA data in data/ (outputs must never be pushed)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_forecasting_benchmark.ipynb
# public figures + reports, then the leak check
uv run python public/reports/make_public_figures.py
uv run python public/reports/build_public_reports.py
uv run python public/scan_leaks.py
```
