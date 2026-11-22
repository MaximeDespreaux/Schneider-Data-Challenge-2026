"""Exploratory model (not pre-registered, not submitted): momentum vs mean reversion with regime gates.

Direct h-step ridge (no intercept, scaled but not centred, so no signal -> no change) of the h-month log change on:
- mom3          own 3-month log change (momentum)
- dev24         log distance to the 24-month average (mean reversion)
- mom3 x calm   momentum only while the price is NOT overextended
- dev24 x hot   reversion only once overextended (|dev24| above its own 80th percentile in the training window)
- mom3 x hivol  momentum in high-volatility regimes (3-month vol / 36-month vol > 1.5)
Bands: volatility-scaled naive bands re-centred on the P50 (as ``DampedTrendETS``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import NaiveVolScaled, _frame


def _features(ly: pd.Series, hot_q: float | None = None) -> tuple[pd.DataFrame, float]:
    mom3 = ly.diff(3)
    dev24 = ly - ly.rolling(24, min_periods=18).mean()
    vol = ly.diff().rolling(3).std() / ly.diff().rolling(36, min_periods=24).std()
    q = float(dev24.abs().quantile(0.8)) if hot_q is None else hot_q
    hot = (dev24.abs() > q).astype(float)
    hivol = (vol > 1.5).astype(float)
    f = pd.DataFrame({"mom3": mom3, "dev24": dev24, "mom_calm": mom3 * (1 - hot), "dev_hot": dev24 * hot,
                      "mom_hivol": mom3 * hivol})
    return f, q


@dataclass
class RegimeMomRev:
    name: str = "regime_momrev"
    alpha_per_obs: float = 1.0
    window: int = 180
    min_rows: int = 48
    clip: float = 0.5

    def forecast(self, y, steps, exog=None):
        bands = NaiveVolScaled().forecast(y, steps)
        last = float(y.iloc[-1])
        ly = np.log(y.astype(float))
        if len(y) < self.min_rows + max(steps) + 24:
            return bands
        f, q = _features(ly.iloc[-(self.window + max(steps) + 36):])
        x_now = f.iloc[-1]
        if not np.isfinite(x_now.to_numpy()).all():
            return bands
        p50 = []
        for h in steps:
            tgt = (ly.shift(-h) - ly).reindex(f.index)
            df = f.join(tgt.rename("tgt")).dropna().iloc[-self.window:]
            if len(df) < self.min_rows:
                p50.append(last)
                continue
            X, Y = df[f.columns].to_numpy(), df["tgt"].to_numpy()
            sd = X.std(0) + 1e-12
            Z = X / sd
            beta = np.linalg.solve(Z.T @ Z + self.alpha_per_obs * len(Y) * np.eye(Z.shape[1]), Z.T @ Y)
            pred = float(np.clip((x_now.to_numpy() / sd) @ beta, -self.clip, self.clip))
            p50.append(last * np.exp(pred))
        p50 = np.asarray(p50)
        return _frame(steps, p50 * bands["p10"].to_numpy() / last, p50, p50 * bands["p90"].to_numpy() / last)
