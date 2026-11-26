"""Scenario engine of the local app: bands, shocks, co-movement, BOM and margin (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

from src.app import scenarios as sc
from src.business.margin_replay import Economics

ORIGIN = pd.Timestamp("2026-09-01")
IDX = pd.date_range("1995-01-01", "2026-09-01", freq="MS")


def _walk(seed, idx=IDX, vol=0.04):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, vol, len(idx)))), index=idx)


def test_band_at_80_is_p10_p90_and_scales_with_level():
    lo, hi = sc.scale_band(90.0, 100.0, 120.0, 0.80)
    assert lo == pytest.approx(90.0) and hi == pytest.approx(120.0)
    lo50, hi50 = sc.scale_band(90.0, 100.0, 120.0, 0.50)
    lo98, hi98 = sc.scale_band(90.0, 100.0, 120.0, 0.98)
    assert lo98 < lo < lo50 < 100 < hi50 < hi < hi98


def test_history_stats_ignore_data_after_origin():
    s = _walk(1)
    longer = pd.concat([s, pd.Series(1e6, index=pd.date_range("2026-10-01", periods=12, freq="MS"))])
    assert sc.sigma(longer, 6, ORIGIN) == sc.sigma(s, 6, ORIGIN)
    assert sc.worst_move(longer, 6, ORIGIN, up=True) == sc.worst_move(s, 6, ORIGIN, up=True)
    assert sc.worst_move(s, 6, ORIGIN, up=True) > 0 > sc.worst_move(s, 6, ORIGIN, up=False)


def test_pass_through_recovers_beta_and_lag():
    x = _walk(2)
    lx = np.log(x)
    y = np.exp(0.5 * lx.shift(2) + np.random.default_rng(3).normal(0, 0.002, len(x))).dropna()
    r = sc.pass_through(y, x, 3, ORIGIN, max_lag=6)
    assert r["lag"] == 2 and r["beta"] == pytest.approx(0.5, abs=0.05)


def test_stress_beta_uses_stress_windows_only():
    x = _walk(4)
    lx = np.log(x)
    stress = sc.in_stress(x.index)
    noise = np.random.default_rng(5).normal(0, 0.001, len(x))
    ly = np.where(stress, 1.0, 0.2) * lx.diff().fillna(0) + noise
    y = np.exp(pd.Series(ly, index=x.index).cumsum())
    full = sc.pass_through(y, x, 1, ORIGIN)
    crisis = sc.pass_through(y, x, 1, ORIGIN, stress=True)
    assert crisis["sample"] == "stress" and crisis["beta"] == pytest.approx(1.0, abs=0.1)
    assert full["beta"] < 0.6


def test_moves_only_driver_vs_others_follow_with_energy_lag():
    x = _walk(6)
    y = np.exp(0.5 * np.log(x).shift(2)).dropna()
    panel = pd.DataFrame({"GAS": x, "STEEL": y})
    shock = sc.Shock(driver="GAS", kind="sigma", k=2.0, up=True)
    alone = sc.scenario_moves(panel, ["GAS", "STEEL"], 6, ORIGIN, shock, follow=False, energy={"GAS"})
    assert alone.at["STEEL", "move"] == 0.0
    assert alone.at["GAS", "move"] == pytest.approx(2.0 * sc.sigma(x, 6, ORIGIN))
    moved = sc.scenario_moves(panel, ["GAS", "STEEL"], 6, ORIGIN, shock, follow=True, energy={"GAS"})
    r = moved.loc["STEEL"]
    assert r["lag"] == 2
    assert r["move"] == pytest.approx(r["beta"] * alone.at["GAS", "move"] * (6 - 2) / 6)


def test_market_wide_moves_every_series_in_its_own_units():
    panel = pd.DataFrame({"A": _walk(7, vol=0.02), "B": _walk(8, vol=0.08)})
    m = sc.scenario_moves(panel, ["A", "B"], 3, ORIGIN, sc.Shock(driver=sc.MARKET, kind="sigma", k=1.0, up=False),
                          follow=True)
    assert m.at["A", "move"] == pytest.approx(-sc.sigma(panel.A, 3, ORIGIN))
    assert m.at["B", "move"] == pytest.approx(-sc.sigma(panel.B, 3, ORIGIN))


def test_bom_band_without_shock_is_the_forecast_and_moves_with_weights():
    comps = pd.DataFrame({"weight": [0.6, 0.4], "ratio50": [1.0, 1.1]}, index=["CU", "ST"])
    lo, mid, hi = sc.bom_band((90.0, 100.0, 115.0), comps, pd.Series({"CU": 0.0, "ST": 0.0}), 0.80)
    assert (lo, mid, hi) == pytest.approx((0.90, 1.00, 1.15))
    _, mid_up, _ = sc.bom_band((90.0, 100.0, 115.0), comps, pd.Series({"CU": np.log(1.2), "ST": 0.0}), 0.80)
    assert mid_up == pytest.approx((0.6 * 1.2 + 0.4 * 1.1) / (0.6 + 0.44))


def test_margin_table_spot_central_hits_target_and_energy_line_lowers_it():
    eco = Economics(material_share=0.4, target_margin=0.25, floor=0.10)
    pts = {"central": (1.0, 1.0)}
    basis = {"p50": 1.0, "p90": 1.1, "hist_p90": 1.15}
    t = sc.margin_table(eco, 6, pts, basis, w_cu=0.5, cu_fwd_ratio=1.0)
    spot = t[(t.policy == "spot") & (t.point == "central")].iloc[0]
    assert spot.margin == pytest.approx(0.25)
    t2 = sc.margin_table(eco, 6, pts, basis, w_cu=0.5, cu_fwd_ratio=1.0, energy_share=0.05,
                         energy_ratio={"central": 2.0})
    spot2 = t2[(t2.policy == "spot") & (t2.point == "central")].iloc[0]
    assert spot2.margin == pytest.approx(0.25 - 0.05)  # energy cost doubles: +5% of price
    assert set(t.policy) == set(sc.DEFAULT_POLICIES)


def test_competitor_gaps_weights_group_changes_by_teardown_shares():
    shares = pd.DataFrame({"brand": ["Schneider", "ABB", "Eaton"],
                           "value_share_copper": [0.5, 0.9, 0.2], "value_share_carbon_steel": [0.5, 0.1, 0.8]})
    gaps, ours = sc.competitor_gaps(shares, {"copper": 0.20, "carbon_steel": 0.0}, ["ABB", "Eaton", "Siemens"])
    assert ours == pytest.approx(0.10)
    assert gaps.set_index("brand").loc["ABB", "gap_pp"] == pytest.approx(8.0)       # 18% vs our 10%
    assert gaps.set_index("brand").loc["Eaton", "gap_pp"] == pytest.approx(-6.0)    # 4% vs 10%
    assert gaps.set_index("brand").loc["Competitor average", "change"] == pytest.approx(0.11)
    assert "Siemens" not in set(gaps.brand)                                          # no teardown -> left out


@pytest.mark.parametrize("delta,pt,gpts", [(100.0, 1.0, 0.0), (100.0, 0.5, 3.0), (900.0, 1.0, 5.0), (-50.0, 1.0, 2.0)])
def test_apply_levers_matches_the_business_module(delta, pt, gpts):
    from src.business import pricing_levers as pl
    price, margin, floor = 4000.0, 0.15, 0.08
    ctx = {"price": {"X": price}, "margin": {"X": margin}, "floor": {"X": floor},
           "cost_base": {("X", "b"): 1000.0}, "index": {("X", 6): {"p50": 100 + delta / 10}},
           "competitor": {6: {"pass_through": {"s": pt}}}, "quote": {("X", 6): {"indexation_recommended": False,
                                                                                 "validity_days": 30, "buffer_eur": 0}},
           "customer": {"customer_tiers": {"t": {"max_margin_giveup_pts": gpts}},
                        "deal_size_eur": {"small": {"range": "< 250k", "extra_giveup_pts": 0}},
                        "large_or_long_deals_require_indexation": False},
           "default_strategy": "s", "default_basis": "b"}
    ref = pl.recommend("X", 6, "p50", tier="t", deal_value_eur=100_000, ctx=ctx)
    ref_pt = ref["pass_through"]  # recommend applies the competitor lever to RM6 only
    out = sc.apply_levers(ref["delta_eur"], price, margin, floor, ref_pt, gpts)
    assert out["increase_eur"] == pytest.approx(ref["increase_eur"])
    assert out["margin_after"] == pytest.approx(ref["margin_after"])
    assert out["floor_binding"] == ref["floor_binding"]
