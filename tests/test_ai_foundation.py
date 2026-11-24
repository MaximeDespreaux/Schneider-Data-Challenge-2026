"""AI track tests on synthetic data only (no NDA data, no model download)."""

import numpy as np
import pandas as pd
import pytest

from src import features as fe
from src.ai import backtest as ab
from src.ai.foundation import build_input


@pytest.fixture
def panel():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2012-01-01", "2026-10-01", freq="MS")
    walk = lambda: 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(idx))))  # noqa: E731
    targets = pd.DataFrame({s: walk() for s in fe.TARGETS}, index=idx)
    exog = pd.DataFrame({d: walk() for d in ["S04", "S05", "S06", "ED01", "ED02", "E09", "E13", "E22", "S09"]}, index=idx)
    exog["cu_spread"] = rng.normal(0, 0.01, len(idx))
    targets.iloc[:5, 0] = np.nan  # a series that starts later
    return targets, exog


@pytest.mark.parametrize("series", ["CRU_CR_DE", "S09", "D01"])
def test_inputs_ignore_the_future(panel, series):
    """Scrambling every value after the origin must not change the model input (target and covariates)."""
    targets, exog = panel
    o = pd.Timestamp("2021-06-01")
    rng = np.random.default_rng(0)
    t2, x2 = targets.copy(), exog.copy()
    for df in (t2, x2):
        fut = df.index > o
        df.loc[fut] = df.loc[fut] * rng.uniform(0.5, 1.5, size=df.loc[fut].shape)
    a = build_input(targets, exog, series, o, covariates=True)
    b = build_input(t2, x2, series, o, covariates=True)
    np.testing.assert_array_equal(a["target"], b["target"])
    assert a.get("past_covariates", {}).keys() == b.get("past_covariates", {}).keys()
    for k in a.get("past_covariates", {}):
        np.testing.assert_array_equal(a["past_covariates"][k], b["past_covariates"][k])
    assert a["target"][-1] == targets.at[o, series] and not np.isnan(a["target"][0])


def test_covariates_are_publication_lagged(panel):
    """S09 (Eurostat, lag 2) uses CRU_CR_DE (lag 0): shift is max(0, 0-2)=0; D01 (lag 1) uses S06 (lag 0): 0.
    A lag-2 driver for a lag-0 target must be shifted by 2 months."""
    targets, exog = panel
    o = pd.Timestamp("2021-06-01")
    inp = build_input(targets, exog, "CRU_CR_DE", o, covariates=True)  # CRU_CR_DE drivers include gap_steel (S09 lag 2)
    X = fe._driver_frame(exog, targets)
    expected = X["gap_steel"].shift(2).loc[:o].iloc[-1]
    assert inp["past_covariates"]["gap_steel"][-1] == pytest.approx(expected)


def _toy_results(n_orig=60, bias=0.0, width=0.05, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m, w in [("naive", width), ("model", width / 3)]:
        for t in pd.date_range("2019-01-01", periods=n_orig, freq="MS"):
            actual = rng.normal(0, 0.05)
            p = bias if m == "model" else 0.0
            rows.append(dict(series="X", origin=t, h=3, model=m, pred=p, q10=p - w, q90=p + w, actual=actual,
                             target_month=t + pd.DateOffset(months=3)))
    return pd.DataFrame(rows)


def test_cqr_repairs_a_too_narrow_band():
    res = _toy_results()
    before = res[res.model == "model"]
    out = ab.add_cqr(res, "model")
    after = out[(out.model == "model+cqr") & out.q10.notna()]
    cov = lambda d: ((d.actual >= d.q10) & (d.actual <= d.q90)).mean()  # noqa: E731
    assert cov(after) > cov(before) + 0.2


def test_preregistered_selection_only_reports_after_split():
    res = _toy_results(n_orig=72)
    out, choice = ab.select_preregistered(res, ["naive", "model"], split="2022-01-01")
    sel = out[out.model == "ai_selected"]
    assert (sel.target_month >= pd.Timestamp("2022-01-01")).all() and len(sel) > 0
    assert set(choice.chosen) <= {"naive", "model"}


def test_naive_vol_ignores_the_future_and_tracks_volatility(panel):
    from src.ai.baselines import naive_vol

    targets, _ = panel
    o = pd.Timestamp("2021-06-01")
    y = targets["CRU_CR_DE"]
    y2 = y.copy()
    y2[y2.index > o] *= 3.0
    assert naive_vol(y.loc[:o], (3, 6, 9)) == naive_vol(y2.loc[:o], (3, 6, 9))
    calm = y.loc[:o]
    idx = pd.date_range(o + pd.DateOffset(months=1), periods=4, freq="MS")
    shocked = pd.concat([calm, pd.Series(calm.iloc[-1] * np.exp(np.cumsum([0.15, -0.15, 0.15, -0.15])), index=idx)])
    w = lambda b: b[3][1] - b[3][0]  # noqa: E731
    assert w(naive_vol(shocked, (3,))) > w(naive_vol(calm, (3,)))
