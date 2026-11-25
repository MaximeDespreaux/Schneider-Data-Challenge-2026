"""Scenario engine of the local app (``src/app/streamlit_app.py``). Pure functions, no file access.

Everything uses data known at the forecast origin (2026-09): the graded forecasts of the submitted model, the BOM cost
weights at the origin, and the historical h-month log changes up to the origin.

- Confidence band: a forecast's P10-P90 band is rescaled in log space around P50 by z((1+c)/2) / z(0.9).
- Shock: a move of the driver's h-month log price, in units of its historical standard deviation (sigma), or its
  worst historical h-month move.
- Co-movement: other series follow with beta = cov / var of h-month log changes (crash preset: change windows ending
  in the stress periods only, because correlations rise in crises). Energy reaches materials with a lag of 0-6 months
  (the lag with the strongest correlation); within an h-month horizon only the share (h - lag) / h has passed through.
- BOM: shocked BOM median = forecast median x sum_i w_i r_i e^{m_i} / sum_i w_i r_i (r_i = component median ratio,
  m_i = its log move). The band keeps the BOM forecast's relative width.
- Margin: ``margin_replay.policy_outcomes`` on the scenario cost ratios, plus an optional direct plant-energy cost line
  (share of price, an assumption).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.business.margin_replay import DEFAULT_POLICIES, Economics, _base_policy, policy_outcomes

Z80 = float(norm.ppf(0.9))
STRESS = (("2008-01-01", "2009-12-01"), ("2020-01-01", "2020-12-01"), ("2021-01-01", "2022-12-01"))
MARKET = "__market__"  # driver value: every series moves by the same shock in its own units
MAX_ENERGY_LAG = 6


@dataclass(frozen=True)
class Shock:
    driver: str        # a series id, or MARKET
    kind: str          # "sigma" (k standard deviations) or "worst" (worst historical h-month move)
    k: float = 0.0
    up: bool = True    # prices up (cost shock) or down


def scale_band(p10, p50, p90, level: float):
    r = norm.ppf(0.5 + level / 2) / Z80
    return p50 * (p10 / p50) ** r, p50 * (p90 / p50) ** r


def log_changes(s: pd.Series, h: int, origin) -> pd.Series:
    """h-month log changes of the monthly series, windows ending at or before the origin."""
    x = np.log(s.loc[: pd.Timestamp(origin)].dropna()).asfreq("MS")
    return (x - x.shift(h)).dropna()


def sigma(s: pd.Series, h: int, origin) -> float:
    return float(log_changes(s, h, origin).std())


def worst_move(s: pd.Series, h: int, origin, up: bool = True) -> float:
    d = log_changes(s, h, origin)
    return float(d.max() if up else d.min())


def in_stress(idx: pd.DatetimeIndex) -> np.ndarray:
    m = np.zeros(len(idx), dtype=bool)
    for a, b in STRESS:
        m |= (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
    return m


def pass_through(y: pd.Series, x: pd.Series, h: int, origin, max_lag: int = 0, stress: bool = False,
                 min_n: int = 24) -> dict:
    """Beta of y's h-month log change on x's h-month log change ``lag`` months earlier, for the lag in 0..max_lag with
    the largest |correlation|. ``stress``: only windows ending in STRESS (full sample, flagged, if too few)."""
    dy, dx = log_changes(y, h, origin), log_changes(x, h, origin)
    best = None
    for lag in range(max_lag + 1):
        df = pd.concat([dy, dx.shift(lag, freq="MS")], axis=1, keys=["y", "x"], sort=True).dropna()
        sample = "full"
        if stress:
            m = in_stress(df.index)
            if m.sum() >= min_n:
                df, sample = df[m], "stress"
            else:
                sample = "full (too few stress months)"
        if len(df) < min_n or df.x.var() == 0:
            continue
        c = float(df.corr().iloc[0, 1])
        if best is None or abs(c) > abs(best["corr"]):
            best = {"beta": float(df.cov().iloc[0, 1] / df.x.var()), "lag": lag, "corr": c, "n": len(df),
                    "sample": sample}
    return best or {"beta": 0.0, "lag": 0, "corr": np.nan, "n": 0, "sample": "not enough data"}


def own_move(s: pd.Series, h: int, origin, shock: Shock) -> float:
    if shock.kind == "worst":
        return worst_move(s, h, origin, up=shock.up)
    return (1 if shock.up else -1) * shock.k * sigma(s, h, origin)


def scenario_moves(panel: pd.DataFrame, series: list[str], h: int, origin, shock: Shock, follow: bool,
                   energy: set[str] = frozenset()) -> pd.DataFrame:
    """Log move of each series over the horizon, with how it was obtained (beta, lag, sample)."""
    rows = {}
    if shock.driver == MARKET:
        for s in series:
            rows[s] = {"move": own_move(panel[s], h, origin, shock), "beta": 1.0, "lag": 0, "sample": "own"}
        return pd.DataFrame.from_dict(rows, orient="index")
    d = shock.driver
    md = own_move(panel[d], h, origin, shock)
    for s in series:
        if s == d:
            rows[s] = {"move": md, "beta": 1.0, "lag": 0, "sample": "driver"}
        elif not follow:
            rows[s] = {"move": 0.0, "beta": 0.0, "lag": 0, "sample": "held"}
        else:
            pt = pass_through(panel[s], panel[d], h, origin, max_lag=MAX_ENERGY_LAG if d in energy else 0,
                              stress=shock.kind == "worst")
            rows[s] = {"move": pt["beta"] * md * max(0, h - pt["lag"]) / h, "beta": pt["beta"], "lag": pt["lag"],
                       "sample": pt["sample"]}
    return pd.DataFrame.from_dict(rows, orient="index")


def bom_band(bom_q, comps: pd.DataFrame, moves: pd.Series, level: float):
    """(low, median, high) BOM cost ratio vs the cost at the origin. ``bom_q`` = BOM index P10/P50/P90 (origin = 100);
    ``comps`` indexed by series with columns weight, ratio50."""
    w, r = comps["weight"], comps["ratio50"]
    m = moves.reindex(comps.index).fillna(0.0)
    f = float((w * r * np.exp(m)).sum() / (w * r).sum())
    p10, p50, p90 = bom_q
    lo, hi = scale_band(p10, p50, p90, level)
    return lo / 100 * f, p50 / 100 * f, hi / 100 * f


def margin_table(eco: Economics, h: int, points: dict, basis: dict, w_cu: float, cu_fwd_ratio: float,
                 energy_share: float = 0.0, energy_ratio: dict | None = None,
                 policies: dict | None = None) -> pd.DataFrame:
    """Margin per quote policy and scenario point. ``points``: {label: (BOM cost ratio, copper ratio)};
    ``basis``: quote bases known at the origin (p50, p90 of the BOM forecast, hist_p90), as ratios.
    The direct plant-energy line costs ``energy_share`` of the price at quote and scales with ``energy_ratio``."""
    q = pd.DataFrame([{"origin": lab, "horizon": h, "ratio": r, "cu_ratio": c, "w_cu": w_cu,
                       "cu_fwd_ratio": cu_fwd_ratio, **basis} for lab, (r, c) in points.items()])
    er = q["origin"].map(energy_ratio or {}).fillna(1.0).to_numpy(dtype=float)
    out = []
    for name, kw in (policies or DEFAULT_POLICIES).items():
        o = policy_outcomes(q, eco, _base_policy(name), **kw)
        final = o["price"] + o["price_change"] * eco.p0
        margin = o["margin"] - energy_share * eco.p0 * (er - 1) / final
        out.append(pd.DataFrame({"policy": name, "point": o["origin"], "markup": o["markup"],
                                 "price_change": o["price_change"], "margin": margin, "breach": margin < eco.floor}))
    return pd.concat(out, ignore_index=True)
