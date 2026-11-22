"""Margin-protection replay (synthetic quotes)."""

import numpy as np
import pandas as pd
import pytest

from src.business.margin_replay import Economics, choose_policy, policy_outcomes, summarize_policies

ECO = Economics(material_share=0.20, target_margin=0.15, floor=0.08)


def _q(ratio, w_cu=0.5, cu_ratio=None, fwd=1.0, p50=1.0, p90=1.1, hist90=1.15, n=1):
    cu_ratio = ratio if cu_ratio is None else cu_ratio
    return pd.DataFrame({"origin": pd.date_range("2020-01-01", periods=n, freq="MS"), "horizon": 6,
                         "ratio": ratio, "w_cu": w_cu, "cu_ratio": cu_ratio, "cu_fwd_ratio": fwd,
                         "p50": p50, "p90": p90, "hist_p90": hist90})


def test_spot_margin_unchanged_when_costs_flat():
    out = policy_outcomes(_q(1.0), ECO, "spot")
    assert out["margin"].iloc[0] == pytest.approx(0.15)
    assert out["markup"].iloc[0] == pytest.approx(0.0)


def test_spot_margin_loses_share_times_cost_rise():
    # material = 20% of price at quote; +10% material cost -> -2 margin points
    assert policy_outcomes(_q(1.10), ECO, "spot")["margin"].iloc[0] == pytest.approx(0.13)


def test_buffer_raises_price_and_margin():
    spot = policy_outcomes(_q(1.10), ECO, "spot").iloc[0]
    buf = policy_outcomes(_q(1.10, p90=1.10), ECO, "p90_model").iloc[0]
    assert buf["markup"] > 0 and buf["margin"] == pytest.approx(0.15)
    assert buf["margin"] > spot["margin"]


def test_two_way_clause_passes_only_the_excess():
    up = policy_outcomes(_q(1.20), ECO, "clause", k=0.05).iloc[0]
    # price rises by 15% of the material basis (beyond the 5% threshold)
    assert up["price_change"] == pytest.approx(0.15 * 0.20, rel=1e-9)
    down = policy_outcomes(_q(0.80), ECO, "clause", k=0.05).iloc[0]
    assert down["price_change"] == pytest.approx(-0.15 * 0.20, rel=1e-9)
    inside = policy_outcomes(_q(1.04), ECO, "clause", k=0.05).iloc[0]
    assert inside["price_change"] == 0.0


def test_copper_hedge_locks_copper_share():
    # all-copper basket, copper +50%, forward = spot: hedged margin equals target, spot margin collapses
    q = _q(1.5, w_cu=1.0, cu_ratio=1.5, fwd=1.0)
    assert policy_outcomes(q, ECO, "hedge_cu")["margin"].iloc[0] == pytest.approx(0.15)
    assert policy_outcomes(q, ECO, "spot")["margin"].iloc[0] == pytest.approx(0.05)


def test_summary_and_rule():
    rng = np.random.default_rng(0)
    q = _q(np.exp(rng.normal(0.02, 0.15, 200)), n=200, p90=1.25, hist90=1.30)
    tab = summarize_policies(q, ECO, policies={"spot": {}, "p90_model": {}, "p90_hist": {}, "clause_5": {"k": 0.05}})
    assert {"breach_rate", "p5_margin", "mean_margin", "mean_markup", "revision_rate"} <= set(tab.columns)
    assert tab.loc["spot", "breach_rate"] > tab.loc["p90_model", "breach_rate"]
    pick = choose_policy(tab, max_breach=0.05, floor=ECO.floor)
    eligible = tab[(tab.breach_rate <= 0.05) & (tab.p5_margin >= ECO.floor)]
    assert pick == eligible["mean_markup"].idxmin()


def test_build_quotes_from_backtest_rows():
    from src.business.margin_replay import build_quotes
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    panel = pd.DataFrame({"D01": np.linspace(100, 200, 24), "X_CU_TERM": 0.03}, index=idx)
    o = idx[6]
    rows = []
    for m, p50, p90 in [("wf_blend", 101.0, 108.0), ("naive", 100.0, 112.0)]:
        rows.append((m, "RM6_NE_IQI_433520", o, 3, idx[9], o, 3, 95.0, p50, p90, 104.0))
    fc = pd.DataFrame(rows, columns=["model", "series", "origin", "horizon", "target_month", "last_obs", "step",
                                     "p10", "p50", "p90", "y_true"])
    w_cu = lambda product, origin: 0.5  # noqa: E731
    q = build_quotes(fc, panel, "RM6_NE_IQI_433520", "wf_blend", w_cu)
    r = q.iloc[0]
    assert r["ratio"] == pytest.approx(1.04) and r["p50"] == pytest.approx(1.01) and r["p90"] == pytest.approx(1.08)
    assert r["hist_p90"] == pytest.approx(1.12)                       # naive empirical band
    assert r["cu_ratio"] == pytest.approx(panel["D01"].iloc[9] / panel["D01"].iloc[6])
    assert r["cu_fwd_ratio"] == pytest.approx(np.exp(0.03))          # 3M spread scaled to h/3
    assert r["w_cu"] == 0.5


def test_report_lists_decision_map_and_tables(tmp_path):
    from src.business.run_margin_replay import write_report
    res = pd.DataFrame({"policy": ["spot", "clause_10"] * 2, "breach_rate": [.1, 0, .05, 0], "p5_margin": [.02, .08, .05, .09],
                        "mean_margin": [.1, .14, .12, .15], "mean_markup": [0, 0, 0, 0], "revision_rate": [0, .3, 0, .2],
                        "mean_abs_revision": [0, .01, 0, .01], "n": 100, "product": "RM6", "model": "wf_blend",
                        "material_share": 0.2, "horizon": 6, "window": ["dev", "dev", "graded", "graded"]})
    pk = pd.DataFrame({"product": ["RM6"], "model": ["wf_blend"], "material_share": [0.2], "horizon": [6],
                       "rule_pick_dev": ["clause_10"]})
    path = write_report(res, pk, tmp_path)
    txt = path.read_text(encoding="utf-8")
    assert "clause_10" in txt and "Decision map" in txt and "graded" in txt


def test_copper_hedge_keeps_fx_risk():
    # copper flat in USD, EUR strengthens 10% (USD per EUR up): EUR copper cost falls; a USD forward does not
    # remove that FX move -> hedged copper cost ratio = fwd_usd * fx_t / fx_t+h
    from src.business.margin_replay import build_quotes
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    fx = pd.Series(1.0, index=idx); fx.iloc[9:] = 1.1
    panel = pd.DataFrame({"D01": 100 / fx, "S06": fx, "X_CU_TERM": 0.0}, index=idx)
    fc = pd.DataFrame([("wf_blend", "RM6_NE_IQI_433520", idx[6], 3, idx[9], idx[6], 3, 95.0, 100.0, 105.0, 100.0),
                       ("naive", "RM6_NE_IQI_433520", idx[6], 3, idx[9], idx[6], 3, 95.0, 100.0, 105.0, 100.0)],
                      columns=["model", "series", "origin", "horizon", "target_month", "last_obs", "step", "p10", "p50",
                               "p90", "y_true"])
    q = build_quotes(fc, panel, "RM6_NE_IQI_433520", "wf_blend", lambda s, o: 1.0)
    assert q["cu_fwd_ratio"].iloc[0] == pytest.approx(1 / 1.1)


def test_effective_markup_counts_revisions():
    tab = summarize_policies(_q(np.array([1.2, 1.0]), n=2), ECO, policies={"clause_5": {"k": 0.05}})
    assert tab.loc["clause_5", "mean_final_markup"] == pytest.approx(0.5 * 0.15 * 0.20 / 1.0)
