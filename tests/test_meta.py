"""Strategy A meta pass (synthetic base forecasts)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.testbench.meta import MetaConfig, meta_pass, pick_stability, realised

ORIGINS = pd.date_range("2015-01-01", periods=60, freq="MS")


def _base(models=("good", "bad", "naive"), series=("A",), horizon=3, seed=0, err=None):
    rng = np.random.default_rng(seed)
    err = err or {"good": 0.02, "bad": 0.04, "naive": 0.03}
    rows = []
    for s in series:
        for o in ORIGINS:
            tm = o + pd.DateOffset(months=horizon)
            y = 100 * np.exp(rng.normal(0, 0.05))
            for m in models:
                p50 = y * np.exp(err[m] * (1 if rng.random() < .5 else -1))
                rows.append((m, s, o, horizon, tm, o, horizon, p50 * .95, p50, p50 * 1.05, y))
    return pd.DataFrame(rows, columns=["model", "series", "origin", "horizon", "target_month", "last_obs", "step",
                                       "p10", "p50", "p90", "y_true"])


CFG = MetaConfig(window=36, min_rows=12, power=2.0, shortlist=("good", "bad"))


def test_only_realised_rows_used():
    base = _base()
    t = ORIGINS[40]
    a, _ = meta_pass(base, CFG, "submission", origins=[t])
    pert = base.copy()
    fut = pert["target_month"] > t
    pert.loc[fut, "y_true"] *= 3.0                                    # outcomes not yet known at t
    pert.loc[fut & (pert.origin < t), ["p10", "p50", "p90"]] *= 3.0   # past forecasts still unscored at t
    b, _ = meta_pass(pert, CFG, "submission", origins=[t])
    cols = ["model", "series", "horizon", "p10", "p50", "p90"]
    pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True))


def test_realtime_uses_series_lag():
    base = _base()
    t = ORIGINS[40]
    r = realised(base, t, "realtime", {"A": 2})
    assert base.loc[r, "target_month"].max() == t - pd.DateOffset(months=2)


def test_realtime_missing_lag_raises():
    base = _base(series=("A", "B"))
    with pytest.raises(KeyError, match="B"):
        realised(base, ORIGINS[40], "realtime", {"A": 2})


def test_best_picks_lower_loss_and_blend_weights_follow_power():
    base = _base(models=("good", "bad"), err={"good": 0.02, "bad": 0.04})
    base = base.assign(p10=base.p50 * 0.999, p90=base.p50 * 1.001)   # near-point forecasts: QL ~ |err|/2
    cfg = MetaConfig(window=36, min_rows=12, power=2.0, shortlist=("good", "bad"), fallback="good")
    _, diag = meta_pass(base, cfg, "submission", origins=[ORIGINS[50]])
    row = diag.iloc[0]
    assert row["pick"] == "good"
    w = json.loads(row["weights"])
    assert w["good"] == pytest.approx(0.8, abs=0.03) and w["bad"] == pytest.approx(0.2, abs=0.03)


def test_ineligible_candidate_dropped_and_fallback_when_none():
    base = _base()
    sparse = base[(base.model != "bad") | (base.origin >= ORIGINS[45])]   # "bad" has few realised rows at t
    _, diag = meta_pass(sparse, CFG, "submission", origins=[ORIGINS[50]])
    assert "bad" not in json.loads(diag.iloc[0]["weights"])
    out, diag0 = meta_pass(base, CFG, "submission", origins=[ORIGINS[5]])  # nothing realised yet
    assert bool(diag0.iloc[0]["fallback_used"])
    nv = base[(base.model == "naive") & (base.origin == ORIGINS[5])]
    for m in ("wf_best", "wf_blend", "wf_median"):
        assert out.loc[out.model == m, "p50"].iloc[0] == nv["p50"].iloc[0]


def test_quantiles_sorted_after_blend():
    base = _base()
    base.loc[base.model == "bad", ["p10", "p90"]] = base.loc[base.model == "bad", ["p90", "p10"]].to_numpy()
    out, _ = meta_pass(base, CFG, "submission")
    assert ((out.p10 <= out.p50 + 1e-12) & (out.p50 <= out.p90 + 1e-12)).all()


def test_pick_stability_columns():
    base = _base()
    _, diag = meta_pass(base, CFG, "submission")
    st = pick_stability(diag)
    assert {"switch_rate", "agree_best_blend"} <= set(st.columns)
