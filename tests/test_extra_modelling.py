"""Tests (données synthétiques) : modèle LightGBM quantile, Ljung-Box, règle de remplacement du champion."""

import numpy as np
import pandas as pd

from src.testbench import metrics
from src.testbench.backtest import run_backtest
from src.testbench.ml_models import LGBMQuantile
from src.testbench.models import Naive


def _panel(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-01", periods=n, freq="MS")
    return pd.DataFrame({"A": 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))}, index=idx)


def test_lgbm_contract_and_no_leak():
    p = _panel()
    y = p["A"].iloc[:150].rename("A")
    out = LGBMQuantile().forecast(y, [3, 6, 9])
    assert list(out.index) == [3, 6, 9] and (out.p10 <= out.p50).all() and (out.p50 <= out.p90).all()
    # perturbation du futur : prévisions identiques
    o = p.index[150]
    pert = p.copy()
    pert.loc[pert.index > o] *= np.random.default_rng(1).uniform(0.2, 5, size=pert.loc[pert.index > o].shape)
    a = run_backtest(p, ["A"], [LGBMQuantile()], origins=[o], verbose=False)
    b = run_backtest(pert, ["A"], [LGBMQuantile()], origins=[o], verbose=False)
    np.testing.assert_array_equal(a[["p10", "p50", "p90"]].to_numpy(), b[["p10", "p50", "p90"]].to_numpy())


def test_lgbm_short_history_falls_back():
    y = _panel(50)["A"].rename("A")
    out = LGBMQuantile(min_obs=60).forecast(y, [3])
    assert np.isfinite(out.to_numpy()).all()


def test_ljung_box_table_flags_autocorrelated_errors():
    rng = np.random.default_rng(0)
    origins = pd.date_range("2000-01-01", periods=120, freq="MS")
    ar = np.zeros(120)
    for t in range(1, 120):
        ar[t] = 0.8 * ar[t - 1] + rng.normal(0, 1)
    scored = pd.DataFrame({"model": "m", "series": "S", "horizon": 1, "origin": origins, "err": ar})
    white = scored.assign(series="W", err=rng.normal(0, 1, 120))
    t = metrics.ljung_box_table(pd.concat([scored, white]), ["m"]).set_index("series")
    assert t.at["S", "p_value"] < 0.01 and t.at["W", "p_value"] > 0.01


def test_replacement_rule():
    from src.external.run_champion import ml_replaces_champion
    idx = pd.MultiIndex.from_product([["champion", "lgbm_q"], ["A", "B"], [3]], names=["model", "series", "horizon"])
    tab = pd.DataFrame({"rel_QL": [0.95, 0.96, 0.90, 0.91], "coverage": [0.80, 0.80, 0.79, 0.79]}, index=idx)
    assert ml_replaces_champion(tab, "champion", "lgbm_q")["adopt"] is True       # meilleur, couverture -0.01
    tab.loc["lgbm_q", "coverage"] = 0.70
    assert ml_replaces_champion(tab, "champion", "lgbm_q")["adopt"] is False      # couverture trop basse
    tab.loc["lgbm_q", "coverage"] = 0.80
    tab.loc["lgbm_q", "rel_QL"] = 1.00
    assert ml_replaces_champion(tab, "champion", "lgbm_q")["adopt"] is False      # pas meilleur
