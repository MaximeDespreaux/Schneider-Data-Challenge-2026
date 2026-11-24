import numpy as np
import pandas as pd

from src.testbench.models import DampedTrendETS


def _y(trend, n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    return pd.Series(100 * np.exp(np.cumsum(trend + rng.normal(0, 0.01, n))), index=idx, name="A")


def test_ets_contract_and_trend():
    m = DampedTrendETS()
    out = m.forecast(_y(0.01), [3, 6, 9])
    assert list(out.columns) == ["p10", "p50", "p90"] and list(out.index) == [3, 6, 9]
    assert (out.p10 <= out.p50).all() and (out.p50 <= out.p90).all()
    last = _y(0.01).iloc[-1]
    assert out.p50.iloc[0] > last and out.p50.is_monotonic_increasing      # follows the up-trend
    assert m.name == "ets_damped"


def test_ets_short_history_falls_back_to_naive_vol():
    y = _y(0.0, n=20)
    out = DampedTrendETS().forecast(y, [3])
    assert np.isclose(out.p50.iloc[0], y.iloc[-1])
