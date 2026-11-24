"""Formula language tests on synthetic data (the public-series allow-list is patched, no NDA data needed)."""

import numpy as np
import pandas as pd
import pytest

from src import features as fe
from src.ai import feature_dsl as dsl
from src.ai import guard


@pytest.fixture(autouse=True)
def allow_list(monkeypatch):
    monkeypatch.setattr(guard, "public_series", lambda: frozenset({"S09", "E09", "ED02", "S06"}))


@pytest.fixture
def panel():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2012-01-01", "2026-09-01", freq="MS")
    walk = lambda s=0.03: 100 * np.exp(np.cumsum(rng.normal(0, s, len(idx))))  # noqa: E731
    exog = pd.DataFrame({c: walk() for c in ["S09", "E09", "ED02", "S06"]}, index=idx)
    targets = pd.DataFrame({"CRU_CR_DE": walk()}, index=idx)
    return targets, exog


@pytest.mark.parametrize("bad", [
    "__import__('os').system('ls')", "S09.values", "lambda: 1", "dlog(S09, 99)", "dlog(S09, -1)",
    "dlog(S09, 1.5)", "dlog(CRU_CR_DE, 3)", "MEPS_CR_DE", "foo(S09)", "spread(S09, 3)", "dlog(3, 3)",
    "S09 + E09", "zscore(S09, window=12)", "dlog(dlog(dlog(dlog(dlog(dlog(S09))))))",
])
def test_rejects_everything_outside_the_language(bad):
    with pytest.raises(dsl.FormulaError):
        dsl.parse(bad)


def test_accepts_valid_formulas():
    for f in ["dlog(S09, 3)", "zscore(spread(dlog(E09, 3), dlog(TARGET, 3)), 24)", "dev_ma(ratio(TARGET, S09), 12)",
              "vol(ED02, 6)", "lag(dlog(S06), 2)"]:
        dsl.parse(f)
    assert dsl.leaves("spread(dlog(E09, 3), dlog(TARGET, 3))") == {"E09", "TARGET"}


@pytest.mark.parametrize("formula", ["zscore(spread(dlog(E09, 3), dlog(TARGET, 3)), 24)", "dev_ma(ratio(TARGET, S09), 12)",
                                     "vol(ED02, 6)", "lag(dlog(S06), 2)"])
def test_compiled_features_ignore_the_future(panel, formula):
    targets, exog = panel
    o = pd.Timestamp("2020-06-01")
    a = dsl.compile_feature(formula, "CRU_CR_DE", targets, exog)
    t2, x2 = targets.copy(), exog.copy()
    rng = np.random.default_rng(0)
    for d in (t2, x2):
        d.loc[d.index > o] *= rng.uniform(0.5, 1.5, size=d.loc[d.index > o].shape)
    b = dsl.compile_feature(formula, "CRU_CR_DE", t2, x2)
    pd.testing.assert_series_equal(a.loc[:o], b.loc[:o])


def test_publication_lag_is_applied(panel):
    targets, exog = panel
    shift = fe._shift_for("S09", "CRU_CR_DE")
    f = dsl.compile_feature("lag(S09, 1)", "CRU_CR_DE", targets, exog)
    pd.testing.assert_series_equal(f, exog.S09.shift(1 + shift), check_names=False)
    assert shift >= 1  # Eurostat PPI is published later than the CRU price


def test_screen_separates_signal_from_noise(panel):
    targets, exog = panel
    ly = np.log(targets.CRU_CR_DE)
    rng = np.random.default_rng(5)
    # a driver whose last 3-month change predicts the next 3 months of the target
    exog["E09"] = np.exp((ly.shift(-3) - ly).fillna(0) + rng.normal(0, 0.02, len(ly)))
    good = dsl.compile_feature("E09", "CRU_CR_DE", targets, exog)  # level of the synthetic driver
    good = np.log(good)
    noise = pd.Series(rng.normal(size=len(ly)), index=ly.index, name="noise")
    s_good = dsl.screen(good.rename("E09"), "CRU_CR_DE", 3, targets)
    s_noise = dsl.screen(noise, "CRU_CR_DE", 3, targets)
    assert s_good.passed and abs(s_good.rho) > 0.3 and s_good.feedback()["signal"] in ("moderate", "strong")
    assert s_noise.passed is False
    assert set(s_good.feedback()) == {"formula", "horizon", "passed", "signal", "reason"}
    dup = dsl.screen(noise, "CRU_CR_DE", 3, targets, existing=noise.to_frame("same"))
    assert dup.max_corr_existing == pytest.approx(1.0)


def test_screen_only_uses_targets_before_cutoff(panel):
    targets, exog = panel
    f = dsl.compile_feature("dlog(S09, 3)", "CRU_CR_DE", targets, exog)
    a = dsl.screen(f, "CRU_CR_DE", 6, targets)
    t2 = targets.copy()
    t2.loc["2024-01-01":] *= 3.0  # change every target month from the cutoff on
    b = dsl.screen(f, "CRU_CR_DE", 6, t2)
    assert (a.rho, a.n) == (b.rho, b.n)


def test_dev_ma_works_on_spreads(panel):
    targets, exog = panel
    f = dsl.compile_feature("dev_ma(spread(dlog(S09), dlog(E09)), 12)", "CRU_CR_DE", targets, exog)
    assert f.notna().sum() > 100
