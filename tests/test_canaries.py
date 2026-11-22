"""Leakage canaries (a) to (d) on synthetic panels (spec 5.4). Must stay fast: CI runs them."""

import json

import numpy as np
import pandas as pd

from src.testbench import canaries as cn
from src.testbench.backtest import run_backtest
from src.testbench.models import AR1Point, Conformal, Naive, NaiveVolScaled, baseline_models
from tests.conftest import toy_derive

TARGETS = ["AR", "RW1", "RW2"]
ORIGINS = pd.date_range("2008-01-01", "2018-12-01", freq="3MS")
FEW = [pd.Timestamp("2012-06-01"), pd.Timestamp("2016-03-01")]


def identity(df):
    return df


def ar_panel(n=240, phi=0.9, seed=3):
    """Three synthetic series of 240 months; AR is AR(1) in log changes (so AR1 has real skill), the others
    are random walks. Zero drift, so no model can win on a trend alone once the changes are permuted."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 0.02, n)
    d = np.zeros(n)
    for t in range(1, n):
        d[t] = phi * d[t - 1] + e[t]
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    return pd.DataFrame({"AR": 100 * np.exp(np.cumsum(d)),
                         "RW1": 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n))),
                         "RW2": 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))}, index=idx)


def test_planted_leak_has_absurd_skill_and_is_flagged():
    raw = ar_panel()
    prep = cn.planted_leak_prepare(TARGETS)
    fc = run_backtest(prep(raw), TARGETS, [Naive(), cn.PlantedLeak()], origins=ORIGINS, verbose=False)
    assert cn.pooled_rel_ql(fc, horizon=3)["planted_leak"] < 0.2
    aud = cn.perturbation_audit(raw, prep, TARGETS, [Naive(), cn.PlantedLeak()], FEW, "submission", None)
    assert aud.set_index("model").loc["naive", "PASS"].all()
    assert (aud.set_index("model").loc["planted_leak", "max_abs_diff"] > 0).all()


def test_permutation_kills_skill():
    raw = ar_panel()
    ar1 = Conformal(AR1Point())
    before = run_backtest(raw, ["AR"], [Naive(), ar1], origins=ORIGINS, verbose=False)
    perm = cn.permute_targets(raw, TARGETS, seed=0)
    after = run_backtest(perm, ["AR"], [Naive(), ar1], origins=ORIGINS, verbose=False)
    assert cn.pooled_rel_ql(before)[ar1.name] < 0.97
    assert cn.pooled_rel_ql(after)[ar1.name] >= 0.97
    # same first value, same multiset of demeaned monthly log changes; deterministic for a seed
    np.testing.assert_allclose(perm.iloc[0], raw.iloc[0])
    d = np.diff(np.log(raw["AR"]))
    np.testing.assert_allclose(np.sort(np.diff(np.log(perm["AR"]))), np.sort(d - d.mean()), atol=1e-12)
    pd.testing.assert_frame_equal(perm, cn.permute_targets(raw, TARGETS, seed=0))


def test_truncation_identity_passes_for_baselines_and_deriver(panel):
    for mode, lags in [("submission", None), ("realtime", {"A": 2, "B": 1})]:
        res = cn.truncation_identity(panel, identity, ["A", "B", "IDX"], baseline_models(),
                                     [pd.Timestamp("2019-06-01"), pd.Timestamp("2021-12-01")], mode, lags,
                                     derive=toy_derive)
        assert (res["n"] > 0).all() and res["PASS"].all(), res


def test_off_by_one_is_caught(panel):
    prep = cn.off_by_one_prepare("A")
    origins = [pd.Timestamp("2019-06-01"), pd.Timestamp("2021-12-01")]
    res = cn.truncation_identity(panel, prep, ["A"], [Naive(), cn.OffByOneModel("A")], origins, "submission", None)
    res = res.set_index("model")
    assert res.loc["naive", "PASS"].all()
    assert not res.loc["off_by_one", "PASS"].any()


def test_report_written_and_keyed_by_head(tmp_path):
    rep = cn.run_canaries(ar_panel(), identity, TARGETS, [Naive(), NaiveVolScaled()], None,
                          {"submission": None}, ORIGINS[::2], tmp_path, extra_checks={"meta_identity": lambda: True})
    path = tmp_path / f"{cn.git_head()}.json"
    assert path.exists()
    saved = json.loads(path.read_text())
    assert set(saved) == {"head", "dirty", "results", "PASS"}
    assert saved["head"] == cn.git_head() and isinstance(saved["dirty"], bool)
    assert saved["results"]["extra"] == {"meta_identity": True}
    assert set(saved["results"]["submission"]) == {"a_planted_leak", "b_permuted", "c_truncation", "d_off_by_one"}
    assert saved["PASS"] is True and rep["PASS"] is True


def test_failing_extra_check_fails_report(tmp_path, monkeypatch):
    # (a) to (d) stubbed as passing: only the extra check decides
    monkeypatch.setattr(cn, "_mode_checks", lambda *a, **k: {"x": {"PASS": True}})
    rep = cn.run_canaries(ar_panel(), identity, TARGETS, [Naive()], None, {"submission": None}, FEW, tmp_path,
                          extra_checks={"meta_identity": lambda: False})
    assert rep["PASS"] is False
