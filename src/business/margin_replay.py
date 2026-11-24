"""Margin-protection replay: which quote policy would have protected the margin best, on past data?

Set-up (Goal A, organiser request: "test and justify the reference that best protects margin while remaining commercially
usable"). Every past month t is a quote for one box. The price is fixed at t; the materials are bought at t + h
(h = 3, 6, 9). Everything is in units of the material cost at quote (= 1):

    price at spot basis   P0 = 1 / material_share
    non-material cost     F  = P0 * (1 - target_margin) - 1          (fixed over the offer)
    quote basis b         P  = (F + b) / (1 - target_margin)         (b = 1 at spot, > 1 with a buffer)
    realised margin       m  = 1 - (F + C) / P_final                 (C = realised material cost ratio)

C is the true change of the product's BOM cost basket over the offer (the backtest's ``y_true / 100``, weights
fixed at t). All inputs of a policy at t are known at t: model quantiles come from the walk-forward backtest,
``hist_p90`` from past h-month changes only, the copper forward from the LME 3-month/cash spread at t
(a USD forward: the EUR/USD move over the offer stays with Schneider).

Policies
- spot        price on today's cost, no protection
- p50_model   price on the forecast median
- p90_model   price on the forecast P90 (risk buffer = P90 - spot)
- p90_hist    price on the historical P90 of h-month changes (no model)
- clause      two-way index clause: beyond +/- k, the material part of the price follows the index
- hedge_cu    copper share bought forward at t (LME 3M); the rest floats
- hedge_cu_clause  copper hedge + two-way clause on the rest
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Economics:
    material_share: float  # material cost / selling price at quote
    target_margin: float
    floor: float

    @property
    def p0(self) -> float:
        return 1.0 / self.material_share

    @property
    def fixed(self) -> float:
        return self.p0 * (1 - self.target_margin) - 1.0


def _price(eco: Economics, basis) -> np.ndarray:
    return (eco.fixed + np.asarray(basis, dtype=float)) / (1 - eco.target_margin)


def _clause(ratio, k: float) -> np.ndarray:
    r = np.asarray(ratio, dtype=float) - 1.0
    return np.sign(r) * np.maximum(np.abs(r) - k, 0.0)


def policy_outcomes(q: pd.DataFrame, eco: Economics, policy: str, k: float = 0.05) -> pd.DataFrame:
    """One row per quote: price multiple, realised margin, price revision. ``q`` columns: ratio, w_cu, cu_ratio,
    cu_fwd_ratio, p50, p90, hist_p90 (cost ratios relative to the cost at quote)."""
    ratio = q["ratio"].to_numpy(dtype=float)
    basis = {"spot": 1.0, "p50_model": q["p50"], "p90_model": q["p90"], "p90_hist": q["hist_p90"],
             "clause": 1.0, "hedge_cu": 1.0, "hedge_cu_clause": 1.0}[policy]
    price = _price(eco, np.broadcast_to(np.asarray(basis, dtype=float), ratio.shape))
    cost = ratio.copy()
    w = q["w_cu"].to_numpy(dtype=float)
    non_cu = np.where(w < 1, (ratio - w * q["cu_ratio"].to_numpy(dtype=float)) / np.where(w < 1, 1 - w, 1), 1.0)
    if policy.startswith("hedge_cu"):
        cost = w * q["cu_fwd_ratio"].to_numpy(dtype=float) + (1 - w) * non_cu
    revision = np.zeros_like(ratio)
    if policy == "clause":
        revision = _clause(ratio, k)                      # in units of the material basis (= 1)
    elif policy == "hedge_cu_clause":
        revision = (1 - w) * _clause(non_cu, k)           # clause on the unhedged basket only
    final = price + revision
    margin = 1 - (eco.fixed + cost) / final
    return pd.DataFrame({"origin": q["origin"].to_numpy(), "horizon": q["horizon"].to_numpy(), "policy": policy,
                         "price": price, "markup": price / eco.p0 - 1, "price_change": revision / eco.p0,
                         "margin": margin, "breach": margin < eco.floor})


DEFAULT_POLICIES = {"spot": {}, "p50_model": {}, "p90_model": {}, "p90_hist": {}, "clause_3": {"k": 0.03},
                    "clause_5": {"k": 0.05}, "clause_10": {"k": 0.10}, "hedge_cu": {},
                    "hedge_cu_clause_5": {"k": 0.05}}


def _base_policy(name: str) -> str:
    if name.startswith("hedge_cu_clause"):
        return "hedge_cu_clause"
    return "clause" if name.startswith("clause") else name


def summarize_policies(q: pd.DataFrame, eco: Economics, policies: dict | None = None) -> pd.DataFrame:
    """Per policy: breach rate (margin < floor), 5% worst margin, mean margin, mean markup vs spot price,
    share of contracts with a price revision and mean absolute revision (% of price)."""
    rows = {}
    for name, kw in (policies or DEFAULT_POLICIES).items():
        o = policy_outcomes(q, eco, _base_policy(name), **kw)
        rows[name] = {"n": len(o), "breach_rate": o["breach"].mean(), "p5_margin": o["margin"].quantile(0.05),
                      "mean_margin": o["margin"].mean(), "mean_markup": o["markup"].mean(),
                      "mean_final_markup": (o["markup"] + o["price_change"]).mean(),
                      "revision_rate": (o["price_change"] != 0).mean(),
                      "mean_abs_revision": o["price_change"].abs().mean()}
    return pd.DataFrame(rows).T.astype(float)


def choose_policy(tab: pd.DataFrame, max_breach: float = 0.05, floor: float | None = None) -> str:
    """Pre-registered rule: among policies with breach rate <= max_breach (and 5% worst margin >= floor when
    given), the lowest mean markup (most competitive quote); ties -> fewer price revisions. If none qualifies,
    the lowest breach rate."""
    ok = tab[tab["breach_rate"] <= max_breach]
    if floor is not None:
        ok = ok[ok["p5_margin"] >= floor]
    if ok.empty:
        return tab["breach_rate"].idxmin()
    return ok.sort_values(["mean_markup", "revision_rate"]).index[0]


def build_quotes(fc: pd.DataFrame, panel: pd.DataFrame, series: str, model: str, w_cu,
                 hist_model: str = "naive") -> pd.DataFrame:
    """Quote table for one product from walk-forward backtest rows (BOM index = 100 at each origin).

    ``w_cu(product_series, origin)`` returns the copper share of the basket known at the origin. The copper forward
    is the LME 3M/cash log spread at the origin (``X_CU_TERM``) scaled to h/3 months (0 when unavailable)."""
    f = fc[(fc.series == series) & fc.y_true.notna()]
    m = f[f.model == model].set_index(["origin", "horizon"])
    h = f[f.model == hist_model].set_index(["origin", "horizon"])["p90"].rename("hist")
    m = m.join(h, how="inner").reset_index()
    cu = panel["D01"]
    term = panel["X_CU_TERM"] if "X_CU_TERM" in panel else pd.Series(0.0, index=panel.index)
    fx = panel["S06"] if "S06" in panel else pd.Series(1.0, index=panel.index)  # USD per EUR
    rows = []
    for r in m.itertuples():
        cu0 = cu.loc[:r.origin].dropna()
        cu_t = cu.get(r.target_month, np.nan)
        spread = term.loc[:r.origin].dropna()
        fx0, fx_t = fx.loc[:r.origin].dropna(), fx.get(r.target_month, np.nan)
        fx_move = float(fx0.iloc[-1]) / fx_t if len(fx0) and np.isfinite(fx_t) else 1.0  # USD forward keeps FX risk
        rows.append({"origin": r.origin, "horizon": r.horizon, "ratio": r.y_true / 100, "p50": r.p50 / 100,
                     "p90": r.p90 / 100, "hist_p90": r.hist / 100, "w_cu": float(w_cu(series, r.origin)),
                     "cu_ratio": cu_t / float(cu0.iloc[-1]) if len(cu0) else np.nan,
                     "cu_fwd_ratio": (float(np.exp(spread.iloc[-1] * r.horizon / 3)) if len(spread) else 1.0) * fx_move})
    return pd.DataFrame(rows).dropna(subset=["ratio", "cu_ratio"])
