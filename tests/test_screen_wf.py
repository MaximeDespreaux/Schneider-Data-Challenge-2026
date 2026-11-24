"""Walk-forward signal screen and ChampionWF (spec 5.5), on synthetic data."""

import numpy as np
import pandas as pd

from src.external.models import ChampionWF
from src.external.screen import screen_at
from src.testbench.canaries import truncation_identity
from src.testbench.models import Naive


def signal_panel(n=240, seed=3):
    """Target S01 (i.i.d. monthly log changes). X_CU_TERM leads its 3-month change with corr 0.5 (enters the
    features as ``cu_term``); X_AL_TERM is pure noise (``al_term``)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    ly = np.cumsum(rng.normal(0, 0.03, n))
    fut3 = np.r_[ly[3:] - ly[:-3], np.full(3, np.nan)]
    lead = fut3 + rng.normal(0, np.sqrt(3) * np.nanstd(fut3), n)  # corr = 1 / sqrt(1 + 3) = 0.5
    return pd.DataFrame({"S01": 100 * np.exp(ly), "X_CU_TERM": lead, "X_AL_TERM": rng.normal(0, 1, n)}, index=idx)


def test_screen_at_ignores_future():
    info = signal_panel()
    t = pd.Timestamp("2016-06-01")
    cut = screen_at(info.loc[:t], "S01")
    masked = info.copy()
    masked.loc[masked.index > t] = np.nan
    pert = info.copy()
    pert.loc[pert.index > t] *= np.random.default_rng(0).uniform(0.2, 5, size=pert.loc[pert.index > t].shape)
    assert cut == screen_at(masked, "S01")
    assert cut == screen_at(pert.loc[:t], "S01")
    assert set(cut) == {("S01", h) for h in (3, 6, 9)}


def test_screen_at_only_uses_realised_targets():
    # the target ends earlier than the features: pairs whose target month is past its last valid month are unused
    info = signal_panel()
    t = pd.Timestamp("2016-06-01")
    ragged = info.loc[:t].copy()
    ragged.loc[ragged.index > pd.Timestamp("2016-03-01"), "S01"] = np.nan
    assert screen_at(ragged, "S01") == screen_at(ragged.loc[:"2016-03-01"], "S01")


def test_screen_finds_planted_signal():
    sel = screen_at(signal_panel(), "S01", max_keep=6)
    assert "cu_term" in sel[("S01", 3)]
    assert sel[("S01", 3)][0] == "cu_term"  # ranked by |corr|
    assert all("al_term" not in v for v in sel.values())
    assert all(len(v) <= 6 for v in sel.values())


def test_champion_wf_truncation_identity():
    origins = [pd.Timestamp("2010-06-01"), pd.Timestamp("2015-01-01")]
    res = truncation_identity(signal_panel(), lambda df: df, ["S01"], [Naive(), ChampionWF()], origins,
                              "submission", None)
    assert (res["n"] > 0).all() and res["PASS"].all(), res


def test_champion_wf_uses_the_screen():
    info = signal_panel().loc[:"2018-06-01"]  # the planted lead is NaN on its last 3 rows
    y = info["S01"].dropna().rename("S01")
    m = ChampionWF(min_signals=1)
    fc = m.forecast(y, [3, 6, 9], exog=info)
    assert m.name == "champion_wf"
    assert (fc["p10"] <= fc["p50"]).all() and (fc["p50"] <= fc["p90"]).all()
    assert fc.loc[3, "p50"] != y.iloc[-1]  # the planted signal moves the P50 away from the naive
