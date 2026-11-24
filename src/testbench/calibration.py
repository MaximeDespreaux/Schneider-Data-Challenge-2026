"""Walk-forward band calibration (conformalised quantile regression, per tail).

The dev window shows the P10-P90 bands of the blend under-cover at 6 and 9 months. For margin protection the
worst case (P90 of cost) must be reliable, so each tail is shifted from the model's OWN past misses, using only
outcomes already realised at the origin (same rule as the meta pass):

    c_lo = q_{1-tau}( log p10 - log y )      over realised rows of the trailing window
    c_hi = q_{1-tau}( log y - log p90 )
    p10' = p10 * exp(-c_lo),  p90' = p90 * exp(c_hi),  p50 unchanged

c > 0 widens a tail that missed too often; c < 0 narrows a tail that was too wide. Finite-sample conformal level
ceil((n+1)(1-tau))/n. Fewer than ``min_rows`` realised rows: bands unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .vintage import month_add


def _level(n: int, tau: float) -> float:
    return min(1.0, np.ceil((n + 1) * (1 - tau)) / n)


def calibrate_bands(fc: pd.DataFrame, model: str = "wf_blend", name: str | None = None, mode: str = "submission",
                    lags: dict[str, int] | None = None, window: int = 60, min_rows: int = 24,
                    tau: float = 0.10) -> pd.DataFrame:
    name = name or f"{model}_cal"
    src = fc[fc.model == model].copy()
    if mode == "realtime":
        if lags is None:
            from .availability import realtime_lags
            lags = realtime_lags()
        missing = sorted(set(src.series) - set(lags))
        if missing:
            raise KeyError(f"no realtime lag for series {missing}")
    out = []
    for (s, h), g in src.groupby(["series", "horizon"]):
        g = g.sort_values("origin")
        lag = int((lags or {}).get(s, 0)) if mode == "submission" else int(lags[s])
        known = g.dropna(subset=["y_true"])
        d_lo = np.log(known.p10) - np.log(known.y_true)
        d_hi = np.log(known.y_true) - np.log(known.p90)
        for r in g.itertuples(index=False):
            t = r.origin
            m = (known.target_month <= month_add(t, -lag)) & (known.origin > month_add(t, -window))
            row = r._asdict()
            n = int(m.sum())
            if n >= min_rows:
                lv = _level(n, tau)
                c_lo, c_hi = float(np.quantile(d_lo[m], lv)), float(np.quantile(d_hi[m], lv))
                row["p10"] = min(r.p10 * np.exp(-c_lo), r.p50)
                row["p90"] = max(r.p90 * np.exp(c_hi), r.p50)
            row["model"] = name
            out.append(row)
    return pd.DataFrame(out, columns=fc.columns)
