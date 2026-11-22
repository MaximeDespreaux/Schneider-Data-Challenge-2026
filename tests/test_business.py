"""Tests of the business modules, on SYNTHETIC data only (they run with an empty data/ folder, as in CI)."""

import numpy as np
import openpyxl
import pandas as pd
import pytest

from src import data as d
from src.business import eaton, lags, pricing_levers, shocks, suppliers, tco


# ---------------------------------------------------------------------------------------------- 1. TCO
def _tco_cfg(fixed=True, carbon=90.0, n=50):
    def v(x, lo=None, hi=None):
        return {"value": x} if fixed or lo is None else {"low": lo, "high": hi}
    return {"n_draws": n,
            "lifetime_yrs": {"value": 35} if fixed else {"choices": [30, 35, 40]},
            "sf6_charge_kg": v(2.0, 1.2, 3.24), "leak_rate_pct_yr": v(0.08, 0.05, 0.10),
            "eol_loss_pct": v(3.0, 1.0, 5.0), "gwp_sf6": {"value": 24300},
            "carbon_price_eur_t": {"choices": [carbon], "labels": ["test"]} if fixed else
            {"choices": [0, 50, 90, 126], "labels": ["a", "b", "c", "d"]},
            "discount_rate": v(0.06, 0.05, 0.08), "airset_premium_pct": v(0.2, 0.1, 0.3),
            "maint_rm6_eur_yr": v(200.0, 150, 300), "maint_airset_reduction": {"value": 0.4},
            "sf6_admin_eur_yr": v(50.0, 20, 80), "eol_sf6_recovery_eur": v(500.0, 200, 800)}


def test_tco_breakeven_and_zero_carbon():
    df = tco.simulate(_tco_cfg(), capex=10_000.0, seed=1)
    assert len(df) == 50
    x = {k: df[k].to_numpy() for k in tco.INPUTS}
    x["airset_premium_pct"] = df["breakeven_premium"].to_numpy()
    res = tco.compute(x, 10_000.0)
    np.testing.assert_allclose(res["tco_air"], res["tco_rm6"], atol=1e-6)
    zero = tco.simulate(_tco_cfg(carbon=0.0), capex=10_000.0, seed=1)
    assert (zero["carbon_cost"] == 0).all()


def test_tco_deterministic_for_seed():
    a = tco.simulate(_tco_cfg(fixed=False), capex=10_000.0, seed=42)
    b = tco.simulate(_tco_cfg(fixed=False), capex=10_000.0, seed=42)
    pd.testing.assert_frame_equal(a, b)
    assert a["lifetime_yrs"].nunique() > 1


# ---------------------------------------------------------------------------------------------- 2. shocks
def test_shock_table_toy():
    w = pd.DataFrame({"product": ["P", "P"], "material": ["m1", "m2"], "value_eur": [100.0, 300.0]})
    tab = shocks.shock_table(w, {"P": 10_000.0}, trigger_pct=5, material_share=0.2, materials=["m1", "m2"],
                             combos={}, shocks_pct=[10])
    r = tab[tab["item"] == "m2"].iloc[0]
    assert r["delta_eur"] == pytest.approx(30.0)
    assert r["pct_of_material_cost"] == pytest.approx(0.075)
    assert r["margin_pts"] == pytest.approx(-0.3)
    assert bool(r["crosses_indexation_trigger"])


# ---------------------------------------------------------------------------------------------- 3. lags
def test_lag_stats_recovers_distributed_lag():
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.date_range("1990-01-01", periods=n, freq="MS")
    dx = rng.normal(0, 0.05, n)
    dy = np.zeros(n)
    dy[3:] = 0.5 * dx[1:-2] + 0.5 * dx[:-3] + rng.normal(0, 0.002, n - 3)
    x = pd.Series(100 * np.exp(np.cumsum(dx)), index=idx)
    y = pd.Series(100 * np.exp(np.cumsum(dy)), index=idx)
    st = lags.lag_stats(x, y, max_lag=6)
    assert st["peak_lag"] in (2, 3)
    assert abs(st["C_max"] - 1.0) < 0.1
    # True C_2 = 0.5 × C_max exactly: "first k with C_k >= 0.5 × C_max" is a tie that the noise breaks either way,
    # so lag_50 is 2 or 3; what must hold is that C_2 sits at half of the total pass-through.
    assert abs(st["C_2"] - 0.5 * st["C_max"]) < 0.05
    assert st["lag_50"] in (2, 3)


# ---------------------------------------------------------------------------------------------- 4. recommend
def _ctx(idx_p90=130.0, margin=0.15, floor=0.08, cost_base=5_000.0):
    customer = {"customer_tiers": {"strategic": {"max_margin_giveup_pts": 5}, "key": {"max_margin_giveup_pts": 3},
                                   "standard": {"max_margin_giveup_pts": 0}},
                "deal_size_eur": {"small": {"range": "< 250k", "extra_giveup_pts": 0},
                                  "medium": {"range": "250k - 1M", "extra_giveup_pts": 1},
                                  "large": {"range": "> 1M", "extra_giveup_pts": 2}},
                "large_or_long_deals_require_indexation": True}
    return {"price": {"RM6": 10_000.0}, "margin": {"RM6": margin}, "floor": {"RM6": floor},
            "cost_base": {("RM6", "bom_value"): cost_base},
            "index": {("RM6", 6): {"p10": 80.0, "p50": 100.0, "p90": idx_p90}},
            "competitor": {6: {"band": "we_hit_harder", "gap_pp": -3.0,
                               "pass_through": {"protect_margin": 0.5, "gain_share": 0.5}}},
            "quote": {("RM6", 6): {"indexation_recommended": False, "validity_days": 90, "buffer_eur": 0.0}},
            "customer": customer, "default_basis": "bom_value", "default_strategy": "protect_margin"}


def test_recommend_floor_binds():
    # delta = 5000 × 30 % = 1500 EUR; pass-through 0.5 -> +750; margin would fall to 15 % − 7.5 % = 7.5 % < 8 %
    r = pricing_levers.recommend("RM6", 6, "p90", "standard", 200_000, ctx=_ctx())
    assert r["floor_binding"] is True
    assert r["margin_after"] == pytest.approx(0.08)
    assert r["increase_eur"] == pytest.approx(1500 - (0.15 - 0.08) * 10_000)


def test_recommend_negative_delta_and_giveup_monotone():
    r = pricing_levers.recommend("RM6", 6, "p10", "standard", 200_000, ctx=_ctx())
    assert r["delta_eur"] < 0 and r["increase_eur"] == 0
    ctx = _ctx(idx_p90=110.0, floor=0.0)
    incs = [pricing_levers.recommend("RM6", 6, "p90", t, deal, ctx=ctx)["increase_eur"]
            for t, deal in [("standard", 100_000), ("standard", 500_000), ("key", 500_000), ("strategic", 2_000_000)]]
    assert all(b <= a + 1e-9 for a, b in zip(incs, incs[1:]))  # more give-up points never raises the price


def test_deal_buckets_from_config_strings():
    ds = _ctx()["customer"]["deal_size_eur"]
    assert pricing_levers.deal_bucket(150_000, ds) == "small"
    assert pricing_levers.deal_bucket(600_000, ds) == "medium"
    assert pricing_levers.deal_bucket(2_000_000, ds) == "large"


# ---------------------------------------------------------------------------------------------- 5. suppliers
def _pitchbook(path, overview=None, income=None, balance=None):
    """Tiny PitchBook-like workbook. Each arg: (periods, {label: values}) or None (sheet absent)."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, spec, hdr_row in [("Overview", overview, 5), ("Income Statement", income, 6),
                                ("Balance Sheet", balance, 6)]:
        if spec is None:
            continue
        ws = wb.create_sheet(name)
        ws.cell(row=7, column=3, value="Amount in thousands  (EUR) (except Ratios, Multiples & per share items)")
        periods, rows = spec
        for j, p in enumerate(periods):
            ws.cell(row=hdr_row, column=13 + j, value=p)
        for i, (label, vals) in enumerate(rows.items()):
            ws.cell(row=10 + i, column=3, value=f"    {label}")
            for j, v in enumerate(vals):
                if v is not None:
                    ws.cell(row=10 + i, column=13 + j, value=v)
    wb.save(path)


def test_pitchbook_parser_three_rules(tmp_path):
    a = tmp_path / "PitchBook_-_Alpha_(Paris)_Company_Financials.xlsx"
    _pitchbook(a, overview=(["Dec 2022", "Dec 2023", "Dec 2024"],
                            {"Total Revenue": [100.0, 110.0, 120.0], "Total Assets": [50.0, 55.0, 60.0]}))
    b = tmp_path / "PitchBook_-_Beta_Company_Financials.xlsx"
    _pitchbook(b, overview=(["Dec 2023", "Dec 2024"], {"Total Assets": [70.0, 80.0]}),
               income=(["Dec 2024", "Dec 2023"], {"Revenue": [None, None], "Business Revenue": [200.0, 190.0],
                                                  "Other Revenue": [5.0, 1.0]}))
    c = tmp_path / "PitchBook_-_Gamma_Company_Financials.xlsx"
    _pitchbook(c, overview=(["Dec 2023", "Dec 2024"], {"Total Assets": [9.0, 10.0]}),
               balance=(["Dec 2024", "Dec 2023"], {"Total Assets": [10.0, 9.0]}))

    ra = suppliers.parse_pitchbook(a, 1.2)
    assert ra["revenue_rule"] == "a" and ra["revenue_keur"] == 120.0 and ra["period"] == "Dec 2024"
    assert ra["revenue_source"] == "PitchBook Overview: Total Revenue Dec 2024"
    rb = suppliers.parse_pitchbook(b, 1.2)
    assert rb["revenue_rule"] == "b" and rb["revenue_keur"] == 205.0 and rb["period"] == "Dec 2024"
    assert rb["revenue_source"].startswith("PitchBook Income Statement: Revenue")
    rc = suppliers.parse_pitchbook(c, 1.2)
    assert rc["revenue_rule"] == "c" and rc["revenue_keur"] == pytest.approx(12.0)
    assert rc["revenue_source"].startswith("PROXY")


def test_pitchbook_units_and_file_matching(tmp_path, monkeypatch):
    p = tmp_path / "PitchBook_-_Alpha_(Paris)_Company_Financials.xlsx"
    _pitchbook(p, overview=(["Dec 2024"], {"Total Revenue": [1.0]}))
    monkeypatch.setattr(d, "RAW", tmp_path.parent)
    (tmp_path.parent / "suppliers").mkdir(exist_ok=True)
    p.rename(tmp_path.parent / "suppliers" / p.name)
    assert suppliers.find_file("Alpha_Paris").name == p.name  # brackets ignored when matching
    with pytest.raises(FileNotFoundError):
        suppliers.find_file("Nobody")
    bad = tmp_path / "bad.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Overview"
    ws.cell(row=7, column=3, value="Amount in millions (USD)")
    wb.save(bad)
    with pytest.raises(ValueError):
        suppliers.parse_pitchbook(bad, 1.2)


def _supplier_cfg():
    return {"suppliers": {"n_peers": 200, "peer_seed": 7, "share_peers_reporting_scope3": 0.4,
                          "grid_kg_per_kwh": {"FR": 0.04, "PL": 0.6}, "gas_kg_per_kwh": 0.184,
                          "sectors": {"s_a": {"elec": [0.1, 0.2], "gas": [0.1, 0.2], "purchased_share": [0.4, 0.5],
                                              "upstream_kg_per_eur": [0.5, 0.6], "spend_factor": 0.4},
                                      "s_b": {"elec": [0.01, 0.02], "gas": [0.01, 0.02], "purchased_share": [0.2, 0.3],
                                              "upstream_kg_per_eur": [0.2, 0.3], "spend_factor": 0.2}}}}


def test_peers_deterministic_and_estimator_rules():
    cfg = _supplier_cfg()
    p1, p2 = suppliers.generate_peers(cfg), suppliers.generate_peers(cfg)
    pd.testing.assert_frame_equal(p1, p2)
    assert (p1["source"] == "SYNTHETIC (demo)").all()
    assert p1.loc[~p1["reports_s3"], "s3_t"].isna().all()
    # country with < 5 peers -> sector-only fallback
    pool = p1.copy()
    keep_pl = pool.index[(pool["sector"] == "s_a") & (pool["country"] == "PL")][:3]
    pool = pool[(pool["country"] != "PL") | pool.index.isin(keep_pl)]
    r = suppliers.predict("PL", "s_a", 50_000, pool, "s12", k=10)
    assert r["sector_only"] is True and r["n"] == 10
    # s3 pool only uses reporting peers
    s3_pool = p1[p1["s3_int"].notna()]
    r3 = suppliers.predict("FR", "s_a", 50_000, p1, "s3", k=1000)
    assert r3["n"] == len(s3_pool[(s3_pool["sector"] == "s_a") & (s3_pool["country"] == "FR")])
    assert r3["p10"] <= r3["p50"] <= r3["p90"]


# ---------------------------------------------------------------------------------------------- 6. Eaton
def test_sf6_flags():
    td = pd.DataFrame({"brand": ["Eaton", "Eaton", "ABB", "ABB", "SGC"],
                       "group": ["copper", "carbon_steel", "copper", "sf6", "copper"],
                       "mass_kg": [26.4, 73.8, 150.0, 1.2, 0.03], "broken": [False, False, False, False, True]})
    f = eaton.sf6_flags(td)
    assert f["Eaton"] is True
    assert f["ABB"] is False
    assert pd.isna(f["SGC"])


# ---------------------------------------------------------------------------------------------- competitor gap p90
def test_competitor_gap_nonzero_at_p90():
    cfg = {"competitor_lever": {"priced_groups": {"copper": "D01", "carbon_steel": "L01"},
                                "brass_mix": {"copper": 0.63, "zinc": 0.37}, "band_scenario": "p90"}}
    comp_cfg = {"competitors": ["ABB"],
                "bands_pct_points": {"competitors_hit_harder": ">= +2", "similar": "-2 to +2", "we_hit_harder": "<= -2"},
                "pass_through": {"competitors_hit_harder": 1.0, "similar": 1.0, "we_hit_harder": 0.5},
                "gain_share_pass_through": 0.5}
    shares = pd.DataFrame({"brand": ["Schneider", "ABB"], "value_share_copper": [0.2, 0.8],
                           "value_share_carbon_steel": [0.8, 0.2]})
    fc = pd.DataFrame({"series": ["D01", "L01"], "horizon": [6, 6], "p10": [90.0, 100.0], "p50": [100.0, 100.0],
                       "p90": [120.0, 100.0], "last_value": [100.0, 100.0]})
    out = pricing_levers.competitor_lever(shares, fc, cfg, comp_cfg, [6], realised={"copper": 0.10, "carbon_steel": 0.0})
    abb = out[out["brand"] == "ABB"].iloc[0]
    assert abb["gap_pp_p50"] == pytest.approx(0.0)              # naive P50 = no change
    assert abb["gap_pp_p90"] == pytest.approx((0.8 - 0.2) * 0.20 * 100)  # +12 pp: ABB holds more copper
    assert abb["band"] == "competitors_hit_harder"               # band taken from the p90 gap
    assert abb["pass_through_gain_share"] == 0.5
    assert abb["gap_realised_12m_pp"] == pytest.approx((0.8 - 0.2) * 0.10 * 100)


# ---------------------------------------------------------------------------------------------- forecast cache
def test_cache_invalidated_when_champion_newer(tmp_path):
    import os

    from src.business import forecasts

    origin = pd.Timestamp("2026-09-01")
    cache, champ = tmp_path / "cache.parquet", tmp_path / "champion_holdout.parquet"
    pd.DataFrame({"series": ["A"], "horizon": [3], "source": ["naive_vol"]}).to_parquet(cache)
    assert forecasts.cache_decision(cache, champ, origin, "champion") == (True, "cache up to date")  # no champion
    pd.DataFrame({"model": ["champion"], "series": ["B"], "origin": [origin], "horizon": [3]}).to_parquet(champ)
    os.utime(cache, (1_000_000, 1_000_000))
    os.utime(champ, (2_000_000, 2_000_000))
    use, why = forecasts.cache_decision(cache, champ, origin, "champion")
    assert use is False and "newer" in why
    os.utime(champ, (500_000, 500_000))  # older champion that lacks series A -> cache kept
    assert forecasts.cache_decision(cache, champ, origin, "champion")[0] is True
    pd.DataFrame({"model": ["champion"], "series": ["A"], "origin": [origin], "horizon": [3]}).to_parquet(champ)
    os.utime(champ, (500_000, 500_000))  # older but now covers A -> naive_vol cache replaced
    use, why = forecasts.cache_decision(cache, champ, origin, "champion")
    assert use is False and "available" in why
