"""Exploratory regime model: momentum vs mean reversion, gated by overextension and volatility."""

import numpy as np
import pandas as pd

from src.testbench.regime import RegimeMomRev


def _series(kind, n=240, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 0.02, n)
    d = np.zeros(n)
    for t in range(1, n):
        d[t] = (0.6 if kind == "momentum" else -0.4) * d[t - 1] + e[t]
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    return pd.Series(100 * np.exp(np.cumsum(d)), index=idx, name="S09")


def test_contract():
    out = RegimeMomRev().forecast(_series("momentum"), [3, 6, 9])
    assert list(out.columns) == ["p10", "p50", "p90"] and list(out.index) == [3, 6, 9]
    assert (out.p10 <= out.p50).all() and (out.p50 <= out.p90).all() and np.isfinite(out.to_numpy()).all()


def test_beats_naive_on_momentum_series():
    y = _series("momentum", seed=1)
    m = RegimeMomRev()
    err_m, err_n = [], []
    for k in range(150, 237, 3):
        h = y.iloc[: k + 1]
        p = m.forecast(h, [3]).p50.iloc[0]
        err_m.append(abs(np.log(y.iloc[k + 3] / p)))
        err_n.append(abs(np.log(y.iloc[k + 3] / h.iloc[-1])))
    assert np.mean(err_m) < np.mean(err_n)


def test_short_history_is_naive_vol():
    y = _series("momentum").iloc[:30]
    out = RegimeMomRev().forecast(y, [3])
    assert np.isclose(out.p50.iloc[0], y.iloc[-1])
