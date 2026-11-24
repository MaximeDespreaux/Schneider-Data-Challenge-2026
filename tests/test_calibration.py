"""Walk-forward band calibration (conformalised quantiles) on synthetic forecasts."""

import numpy as np
import pandas as pd
import pytest

from src.testbench.calibration import calibrate_bands

ORIGINS = pd.date_range("2005-01-01", periods=160, freq="MS")


def _fc(half_width, sigma=0.10, h=6, seed=0, model="wf_blend"):
    """Forecasts with P50 = truth noise centre; band +/- half_width (log); true errors N(0, sigma)."""
    rng = np.random.default_rng(seed)
    rows = []
    for o in ORIGINS:
        y = 100 * np.exp(rng.normal(0, sigma))
        rows.append((model, "A", o, h, o + pd.DateOffset(months=h), o, h, 100 * np.exp(-half_width), 100.0,
                     100 * np.exp(half_width), y))
    return pd.DataFrame(rows, columns=["model", "series", "origin", "horizon", "target_month", "last_obs", "step",
                                       "p10", "p50", "p90", "y_true"])


def _tails(out, start=60):
    o = out[out.origin >= ORIGINS[start]]
    return (o.y_true < o.p10).mean(), (o.y_true > o.p90).mean()


def test_too_narrow_bands_are_widened_to_target():
    fc = _fc(half_width=0.03)                       # true 10/90% is +/-0.128: raw bands miss ~38% per side
    out = calibrate_bands(fc, mode="submission")
    lo, hi = _tails(out)
    assert 0.03 < lo < 0.18 and 0.03 < hi < 0.18
    assert set(out.model) == {"wf_blend_cal"}


def test_too_wide_bands_are_narrowed():
    fc = _fc(half_width=0.40)
    out = calibrate_bands(fc, mode="submission")
    later = out[out.origin >= ORIGINS[60]]
    assert (np.log(later.p90 / later.p10) < 0.8).all()           # narrower than the raw 0.80 log width
    lo, hi = _tails(out)
    assert lo > 0.03 and hi > 0.03


def test_median_kept_and_quantiles_ordered():
    fc = _fc(half_width=0.03)
    out = calibrate_bands(fc, mode="submission").set_index("origin")
    src = fc.set_index("origin")
    assert np.allclose(out.p50, src.p50)
    assert ((out.p10 <= out.p50) & (out.p50 <= out.p90)).all()


def test_too_few_realised_rows_leaves_bands_unchanged():
    fc = _fc(half_width=0.03)
    out = calibrate_bands(fc, mode="submission", min_rows=24)
    first = out[out.origin == ORIGINS[5]].iloc[0]
    raw = fc[fc.origin == ORIGINS[5]].iloc[0]
    assert first.p10 == raw.p10 and first.p90 == raw.p90


def test_only_realised_rows_used():
    fc = _fc(half_width=0.03)
    t = ORIGINS[100]
    a = calibrate_bands(fc, mode="submission")
    pert = fc.copy()
    pert.loc[pert.target_month > t, "y_true"] *= 5.0
    b = calibrate_bands(pert, mode="submission")
    ka, kb = a[a.origin == t].iloc[0], b[b.origin == t].iloc[0]
    assert (ka.p10, ka.p90) == (kb.p10, kb.p90)


def test_realtime_needs_lags():
    fc = _fc(half_width=0.03)
    with pytest.raises(KeyError, match="A"):
        calibrate_bands(fc, mode="realtime", lags={})
