"""Tests du banc de test, sur données SYNTHÉTIQUES (aucune donnée Schneider) : ils tournent partout, y compris
en CI GitHub Actions où data/ est vide."""

import numpy as np
import pandas as pd
import pytest

from src.testbench import metrics
from src.testbench.backtest import run_backtest
from src.testbench.models import AR1Point, Conformal, DriftPoint, Naive, NaiveGaussian
from src.testbench.submission import COLUMNS, from_forecasts, validate_submission
from src.testbench.vintage import information_set, month_add, months_between

from tests.conftest import toy_derive as _toy_derive


class Spy:
    """Modèle espion : enregistre la dernière date vue et échoue s'il voit le futur."""
    name = "spy"

    def __init__(self):
        self.seen = []

    def forecast(self, y, steps, exog=None):
        self.seen.append((y.index[-1], exog.index[-1]))
        last = y.iloc[-1]
        return pd.DataFrame({"p10": last, "p50": last, "p90": last}, index=pd.Index(steps, name="step"))


@pytest.mark.xfail(strict=True, reason="Task 4 adds run_backtest(lags=)")
@pytest.mark.parametrize("mode", ["submission", "realtime"])
def test_perturbing_the_future_changes_nothing(panel, mode):
    """Test de perturbation : on remplace tout le futur (toutes séries) par du bruit -> prévisions identiques."""
    from src.testbench.models import NaiveVolScaled
    models = [Naive(), NaiveVolScaled(), Conformal(AR1Point()), Conformal(DriftPoint())]
    lags = {"A": 2} if mode == "realtime" else None
    for o in [pd.Timestamp("2019-06-01"), pd.Timestamp("2021-12-01")]:
        pert = panel.copy()
        fut = pert.index > o
        pert.loc[fut] = pert.loc[fut] * np.random.default_rng(7).uniform(0.2, 5, size=pert.loc[fut].shape)
        kw = dict(origins=[o], mode=mode, derive=_toy_derive, verbose=False, lags=lags)
        a = run_backtest(panel, ["A", "B", "IDX"], models, **kw)
        b = run_backtest(pert, ["A", "B", "IDX"], models, **kw)
        cols = ["p10", "p50", "p90"]
        assert len(a) == len(b) > 0
        np.testing.assert_array_equal(a[cols].to_numpy(), b[cols].to_numpy())


def test_month_helpers():
    assert month_add(pd.Timestamp("2026-09-01"), 3) == pd.Timestamp("2026-12-01")
    assert months_between(pd.Timestamp("2026-09-01"), pd.Timestamp("2027-06-01")) == 9


def test_information_set_never_contains_future(panel):
    o = pd.Timestamp("2020-06-01")
    info = information_set(panel, o)
    assert info.index.max() == o
    rt = information_set(panel, o, mode="realtime", lags={"A": 2})
    assert rt["A"].dropna().index.max() == pd.Timestamp("2020-04-01")
    assert rt["B"].dropna().index.max() == o


def test_backtest_has_no_leakage(panel):
    spy = Spy()
    origins = pd.date_range("2019-01-01", "2024-12-01", freq="MS")
    fc = run_backtest(panel, ["A", "B"], [spy], origins=origins, verbose=False)
    for (last_y, last_exog), origin in zip(spy.seen, np.repeat(origins, 2)):
        assert last_y <= origin and last_exog <= origin
    # B a 2 mois de retard en fin d'échantillon -> le pas réel est plus long que l'horizon
    late = fc[(fc.series == "B") & (fc.origin == pd.Timestamp("2024-12-01")) & (fc.horizon == 3)]
    assert late["step"].iloc[0] == 5


def test_naive_scores_zero_on_constant_series():
    idx = pd.date_range("2015-01-01", periods=80, freq="MS")
    df = pd.DataFrame({"C": 100.0 + np.arange(80) * 0}, index=idx)
    df["C"] = df["C"] * np.exp(np.r_[0, np.random.default_rng(1).normal(0, 1e-12, 79)])
    fc = run_backtest(df, ["C"], [Naive()], origins=pd.date_range("2019-01-01", "2021-01-01", freq="MS"), verbose=False)
    sc = metrics.score_rows(fc)
    assert sc["ae"].max() < 1e-9


@pytest.mark.parametrize("model", [Naive(), NaiveGaussian(), Conformal(DriftPoint()), Conformal(AR1Point())])
def test_models_respect_contract(panel, model):
    y = panel["A"].iloc[:60]
    out = model.forecast(y, [3, 6, 9])
    assert list(out.columns) == ["p10", "p50", "p90"] and list(out.index) == [3, 6, 9]
    assert (out.p10 <= out.p50).all() and (out.p50 <= out.p90).all()
    assert (out.p90 / out.p10).is_monotonic_increasing  # incertitude croissante avec l'horizon


def test_coverage_close_to_80pct_on_random_walk():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2000-01-01", periods=300, freq="MS")
    df = pd.DataFrame({"RW": 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 300)))}, index=idx)
    fc = run_backtest(df, ["RW"], [Naive()], origins=idx[60:-9], verbose=False)
    cov = metrics.score_rows(fc).groupby("horizon")["covered"].mean()
    assert ((cov > 0.70) & (cov < 0.90)).all(), cov


def test_pinball_and_dm():
    assert metrics.pinball(1.0, 0.0, 0.9) == pytest.approx(0.9)
    assert metrics.pinball(0.0, 1.0, 0.9) == pytest.approx(0.1)
    rng = np.random.default_rng(3)
    e_ref = rng.normal(0, 1, 200)
    stat, p = metrics.dm_test(e_ref * 0.5, e_ref, h=3)
    assert stat < 0 and p < 0.01
    stat, p = metrics.dm_test(e_ref + rng.normal(0, 1e-3, 200), e_ref, h=3)
    assert p > 0.05


def test_submission_roundtrip(panel):
    o = pd.Timestamp("2024-09-01")
    fc = run_backtest(panel, ["A", "B"], [Naive()], origins=[o], verbose=False)
    sub = from_forecasts(fc, "2024-09", "team_x", "naive", "claude", region={"A": "EU", "B": "WW"},
                         unit={"A": "EUR/t", "B": "index"})
    assert list(sub.columns) == COLUMNS and validate_submission(sub) == []
    bad = sub.copy()
    bad.loc[0, "p10"] = bad.loc[0, "p90"] + 1
    bad.loc[1, "target_month"] = "2030-01"
    msgs = " ".join(validate_submission(bad))
    assert "p10" in msgs and "target_month" in msgs
