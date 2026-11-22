"""Causal BOM cost index (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

from src import data as d
from src.testbench import targets as tg


@pytest.fixture
def toy(monkeypatch):
    rng = np.random.default_rng(1)
    idx = pd.date_range("2010-01-01", periods=120, freq="MS")
    walk = lambda lvl, s: lvl * np.exp(np.cumsum(rng.normal(0, s, len(idx))))  # noqa: E731
    p = pd.DataFrame({"D01": walk(6000, .05), "L01": walk(600, .04), "L01_FR": walk(620, .04),
                      "L02": walk(2500, .04), "S11": walk(100, .02), "EL01": walk(100, .01)}, index=idx)
    p.loc[: idx[29], "L01_FR"] = np.nan          # regional series starts later
    p.loc[idx[-1], "L02"] = np.nan               # stainless not yet published at the last month
    bom = pd.DataFrame({"product": ["RM6", "RM6", "RM6", "RM6", "AirSeT", "AirSeT"],
                        "material_family": ["Copper", "Carbon steel", "Stainless steel", "Plastic",
                                            "Carbon steel", "Copper"],
                        "mass_kg": [40.0, 140.0, 55.0, 6.0, 340.0, 28.0], "implied_total_kg": 0})
    monkeypatch.setattr(d, "load_bom", lambda *a, **k: bom)
    a = {"bom": {"version": "xlsx_v2", "rm6_big_unlabelled_share": 0.45},
         "cost_index": {"base_month": "2026-09"},
         "material_series": {"copper": {"series": "D01"}, "carbon_steel": {"series": "L01"},
                             "stainless_steel": {"series": "L02"}, "plastics": {"series": "S11"},
                             "electronics": {"series": "EL01"}},
         "material_series_by_product": {"RM6": {"carbon_steel": "L01_FR"}},
         "eur_per_kg_at_base": {k: {"value": v} for k, v in
                                {"copper": 12.0, "carbon_steel": .8, "stainless_steel": 2.7, "plastics": 2.5}.items()},
         "extras": {"electronics": {"include": True,
                                    "eur_per_box_at_base": {"rm6": {"value": 50.0}, "airset": {"value": 300.0}}}}}
    return p, a


def test_index_is_100_at_origin_even_if_a_component_is_late(toy):
    p, a = toy
    der = tg.make_bom_deriver(p, a)
    o = p.index[-1]
    out = der(p, p, o)
    assert np.allclose(out.loc[o], 100.0)


def test_truncation_identity(toy):
    p, a = toy
    der = tg.make_bom_deriver(p, a)
    for o in p.index[[20, 45, 80, 119]]:
        cut = p.loc[:o]
        pd.testing.assert_frame_equal(der(cut, cut, o), der(p, cut, o).loc[:o])


def test_no_future_level_constant(toy):
    p, a = toy
    der = tg.make_bom_deriver(p, a)
    o = p.index[60]
    q = p.copy()
    q.loc[q.index > o] *= 3.0                       # change every value after the origin
    der_q = tg.make_bom_deriver(q, a)
    pd.testing.assert_frame_equal(der(p.loc[:o], p.loc[:o], o), der_q(q.loc[:o], q.loc[:o], o))


def test_regional_override_and_late_component_dropped(toy):
    p, a = toy
    der = tg.make_bom_deriver(p, a)
    w_late = der.weights_at(p, p.index[60]).set_index(["product", "material"])
    assert w_late.loc[("RM6", "carbon_steel"), "series"] == "L01_FR"
    assert w_late.loc[("AirSeT", "carbon_steel"), "series"] == "L01"
    w_early = der.weights_at(p, p.index[10]).set_index(["product", "material"])
    assert ("RM6", "carbon_steel") not in w_early.index            # L01_FR unknown at that origin
    assert np.isclose(w_early.xs("RM6")["weight"].sum(), 1.0)
    out = der(p.loc[: p.index[10]], p.loc[: p.index[10]], p.index[10])
    assert np.isclose(out.loc[p.index[10], "RM6_NE_IQI_433520"], 100.0)


def test_index_priced_materials_use_config_level(toy):
    p, a = toy
    der = tg.make_bom_deriver(p, a)
    w = der.weights_at(p, p.index[60]).set_index(["product", "material"])
    assert w.loc[("RM6", "plastics"), "eur_per_kg"] == 2.5
    assert w.loc[("RM6", "electronics"), "value"] == 50.0


def test_history_uses_available_components_but_truth_needs_all(toy):
    p, a = toy
    p = p.copy()
    p.loc[: p.index[49], "L02"] = np.nan                 # stainless starts late
    der = tg.make_bom_deriver(p, a)
    o = p.index[80]
    hist = der(p.loc[:o], p.loc[:o], o)["RM6_NE_IQI_433520"]
    assert hist.loc[p.index[30]:].notna().all()         # history before stainless starts is still defined
    assert hist.loc[o] == pytest.approx(100.0)
    q = p.copy()
    q.loc[p.index[90], "L02"] = np.nan                   # a component missing at a target month
    truth = der(q, q.loc[:o], o)["RM6_NE_IQI_433520"]
    assert np.isnan(truth.loc[p.index[90]])              # never score a truth built from a partial basket
