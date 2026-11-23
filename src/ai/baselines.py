"""Volatility-scaled naive benchmark (``naive_vol``), ported from the team test bench.

Source: ``champion/v2`` ``src/testbench/models.py::NaiveVolScaled`` (same defaults: EWMA half-life 3 months,
current-vol floor 0.5 × long vol, historical-vol floor 0.5 × long vol, re-centred bands). Ported rather than
imported so the AI track stays standalone; the maths is unchanged.

P50 = today's price. P10/P90 = past h-month log-changes standardised by the EWMA volatility of their time,
re-scaled to the current volatility. It is the tougher range benchmark: bands tighten in calm markets and
widen as soon as volatility rises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import backtest as bt

from . import config as C

Z90 = 1.2815515655446004


def naive_vol(y: pd.Series, steps, halflife: float = 3.0, floor: float = 0.5,
              hist_floor: float = 0.5) -> dict[int, tuple[float, float]]:
    """P10/P90 as log-change vs the last value of ``y`` (y ends at the origin, no NaN). P50 = 0."""
    ly = np.log(y.astype(float))
    d1 = ly.diff().dropna()
    vol = np.sqrt((d1 ** 2).ewm(halflife=halflife, min_periods=6).mean())
    long_vol = float(d1.std(ddof=1))
    cur = max(float(vol.iloc[-1]), floor * long_vol)
    out = {}
    for h in steps:
        dh = (ly.shift(-h) - ly).dropna()  # only changes fully realised by the origin
        v = vol.reindex(dh.index).dropna().clip(lower=hist_floor * long_vol)
        z = (dh.reindex(v.index) / (v * np.sqrt(h))).to_numpy()
        z = z[np.isfinite(z)]
        if len(z) < 12:
            z = np.array([-Z90, 0.0, Z90])
        med = np.median(z)
        out[h] = ((np.quantile(z, 0.1) - med) * cur * np.sqrt(h), (np.quantile(z, 0.9) - med) * cur * np.sqrt(h))
    return out


def run_naive_vol(targets: pd.DataFrame, start: str = C.BACKTEST_START, live: bool = False,
                  horizons=C.HORIZONS) -> pd.DataFrame:
    """Rows in the ``src/backtest.py`` format (log-change vs origin), model ``naive_vol``."""
    from .backtest import _pairs

    rows = []
    for s, t in _pairs(targets, start, live):
        y = targets[s].loc[:t].dropna()
        for h, (lo, hi) in naive_vol(y, horizons).items():
            rows.append(dict(series=s, origin=t, h=h, model="naive_vol", pred=0.0, q10=lo, q50=0.0, q90=hi))
    res = pd.DataFrame(rows)
    if live:
        res["target_month"] = res.origin + res.h.apply(lambda h: pd.DateOffset(months=int(h)))
        res["y_origin"] = [targets.at[t, s] for s, t in zip(res.series, res.origin)]
        return res
    return bt.attach_truth(res, targets)
