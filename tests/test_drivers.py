"""DriverRidge on SYNTHETIC data only (no real file is read)."""

import numpy as np
import pandas as pd

from src.testbench.driver_models import DriverRidge


def _toy_panel():
    rng = np.random.default_rng(3)
    n = 150
    idx = pd.date_range("2012-01-01", periods=n, freq="MS")
    x = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.04, n)))
    y = np.empty(n)
    y[0] = 50.0
    y[1:] = 0.5 * x[:-1] * np.exp(rng.normal(0, 0.005, n - 1))  # Y follows X with a 1-month lag
    z = 80 * np.exp(np.cumsum(rng.normal(0.0, 0.03, n)))
    return pd.DataFrame({"X": x, "Y": y, "Z": z}, index=idx)


def _check(out):
    assert list(out.index) == [3, 6, 9] and list(out.columns) == ["p10", "p50", "p90"]
    assert np.isfinite(out.to_numpy()).all()
    assert (out["p10"] <= out["p50"]).all() and (out["p50"] <= out["p90"]).all()


def test_driver_ridge_with_driver_and_without():
    panel = _toy_panel()
    model = DriverRidge(drivers={"Y": ["X"]})
    _check(model.forecast(panel["Y"], [3, 6, 9], exog=panel))
    feats = model._features(panel["Y"], panel, "Y")
    assert {"X_d1", "X_d3"} <= set(feats.columns)  # the driver is used
    # a target absent from DRIVERS still gets forecasts, from its own features only
    _check(model.forecast(panel["Z"], [3, 6, 9], exog=panel))
    assert set(model._features(panel["Z"], panel, "Z").columns) == {"own_d1", "own_d3", "own_dev12"}


def test_driver_ridge_trains_before_a_driver_exists():
    import numpy as np
    import pandas as pd
    from src.testbench.driver_models import DriverRidge
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-01", periods=180, freq="MS")
    y = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, .02, 180))), index=idx, name="S09")
    exog = pd.DataFrame({"S09": y, "S04": 50 * np.exp(np.cumsum(rng.normal(0, .05, 180))),
                         "L01": np.nan}, index=idx)
    # L01 is a column of the panel but has no value yet at this origin (licensed steel before 2012)
    m = DriverRidge(drivers={"S09": ["S04", "L01"]})
    out = m.forecast(y, [3], exog=exog)
    assert not np.isclose(out.p50.iloc[0], y.iloc[-1])  # a model was fitted (not the no-data fallback)
