"""Tests des correctifs de données (acier MEPS/CRU, epoxy, électronique, BOM v2, teardowns), sur données
SYNTHÉTIQUES uniquement : ils tournent avec un dossier data/ vide, comme en CI."""

from datetime import datetime

import numpy as np
import openpyxl
import pandas as pd
import pytest

from src import data as d
from src.testbench import targets as t
from src.testbench.run import default_targets


# ---------------------------------------------------------------------------- helpers
def _sheet(wb, name, rows, first_row=1):
    ws = wb.create_sheet(name)
    for i, r in enumerate(rows, start=first_row):
        for j, v in enumerate(r, start=1):
            if v is not None:
                ws.cell(row=i, column=j, value=v)
    return ws


def _toy_bom(rows):
    """Faux résultat de d.load_bom : rows = [(product, family, kg)], masse totale implicite = 100 kg."""
    return pd.DataFrame([{"product": p, "material_family": f, "bom_version": "v2", "mass_kg": kg,
                          "share_pct": kg / 100.0, "implied_total_kg": 100.0} for p, f, kg in rows])


def _toy_assumptions(base_month="2020-06", electronics=True, epoxy_extra=False):
    return {
        "cost_index": {"base_month": base_month},
        "bom": {"version": "xlsx_v2", "rm6_big_unlabelled_share": 0.45},
        "steel_price_source": "licensed",
        "material_series": {"copper": {"series": "D01"}, "epoxy": {"series": "EP01"},
                            "electronics": {"series": "EL01"}},
        "eur_per_kg_at_base": {"copper": {"value": 10.0}, "epoxy": {"value": 5.0}},
        "extras": {"epoxy": {"include": epoxy_extra, "share_of_mass": 0.01},
                   "electronics": {"include": electronics,
                                   "eur_per_box_at_base": {"rm6": {"value": 300.0}, "airset": {"value": 300.0}}}},
    }


# ---------------------------------------------------------------------------- 1. acier sous licence
def test_load_licensed_steel(tmp_path, monkeypatch):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    junk = [("MEPS LOW", "Germany"), ("x",), ("x",), ("x",)]
    _sheet(wb, "MEPS", junk + [
        ("Month", "HR coil", "CR coil", "HDG coil", "EG coil"),
        (datetime(2026, 7, 1), 600, 780, 770, 800),
        (datetime(2026, 8, 1), 610, "-", 800, 810),
        ("note", 1, 2, 3, 4),  # pas une date -> ignorée
        (datetime(2026, 9, 1), 620, 0, 0, 820),
    ])
    _sheet(wb, "CRU Stainless", junk + [
        ("Month", "BASE", "Alloy"),
        (datetime(2026, 7, 15), 500, 2200),
        (datetime(2026, 8, 15), 500, 2228),
        (datetime(2026, 9, 15), None, None),
    ])
    wb.save(tmp_path / "Copia_de_Europe_Steel_INDEXES.xlsx")
    monkeypatch.setattr(d, "RAW", tmp_path)

    s = d.load_licensed_steel()
    assert list(s.columns) == ["L01", "L02", "L03"]
    assert s.index.name == "month"
    assert list(s.index) == list(pd.date_range("2026-07-01", "2026-09-01", freq="MS"))
    assert s.at[pd.Timestamp("2026-08-01"), "L02"] == 500 + 2228
    assert np.isnan(s.at[pd.Timestamp("2026-08-01"), "L01"])  # cellule '-'
    assert np.isnan(s.at[pd.Timestamp("2026-09-01"), "L02"])  # pas encore publié
    assert s.at[pd.Timestamp("2026-09-01"), "L03"] == 0
    assert all(s.dtypes == float)


def test_load_licensed_steel_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "RAW", tmp_path)
    with pytest.raises(FileNotFoundError):
        d.load_licensed_steel()


# ---------------------------------------------------------------------------- 2. teardowns
def test_teardown_parser(tmp_path, monkeypatch):
    wb = openpyxl.Workbook()
    wb.active.title = "Benchmark"
    _sheet(wb, "BOM ormazabal Test-1", [
        ("Teardown — ormazabal Test-1 HN",), ("Part",),
        ("METALS · PRECIOUS — 11 000 g · 2%",),
        ("  Copper — 7 000 g · 11 pcs",), ("Busbar",),
        ("  Steel — 1 234 567 g · 3 pcs",),
        ("PLASTICS — 500 g · 1%",),
        ("  Epoxy — 500 g · 1 pcs",),
        ("  Wood — 2000 g · 1 pcs",),
        ("  SF6 — 300 g · 1 pcs",),
        ("  FR-4, — 120 g · 1 pcs",),
        ("  Mystery — 10 g · 1 pcs",),
    ])
    _sheet(wb, "BOM SGC – Tiny nv", [
        ("Teardown — SGC – Tiny nv DR-6",), ("Part",),
        ("  Copper — 34 g · 3 pcs",), ("  Steel — 89 g · 36 pcs",),
    ])
    wb.save(tmp_path / "competitor_teardown_rm6_class.xlsx")
    monkeypatch.setattr(d, "RAW", tmp_path)

    df = d.load_competitor_teardowns()
    assert list(df.columns) == ["brand", "product", "material", "group", "mass_kg", "total_kg", "broken"]
    orm = df[df.brand == "Ormazabal"].set_index("material")
    assert (orm["product"] == "ormazabal Test-1 HN").all()
    assert orm.at["Copper", "mass_kg"] == pytest.approx(7.0)
    assert orm.at["Steel", "mass_kg"] == pytest.approx(1234.567)
    groups = orm["group"].to_dict()
    assert groups["Copper"] == "copper" and groups["Steel"] == "carbon_steel" and groups["Epoxy"] == "epoxy"
    assert groups["Wood"] == "packaging" and groups["SF6"] == "sf6"
    assert groups["FR-4"] == "electronics" and groups["Mystery"] == "other"
    assert not any("METALS" in m for m in df["material"])  # en-têtes de groupe ignorés
    assert not orm["broken"].any()
    sgc = df[df.brand == "SGC"]
    assert sgc["total_kg"].iloc[0] == pytest.approx(0.123) and sgc["broken"].all()


# ---------------------------------------------------------------------------- 3. dérivation (électronique)
def test_derive_with_electronics_pseudo_row(monkeypatch):
    monkeypatch.setattr(d, "load_bom", lambda *a, **k: _toy_bom([("RM6", "Copper", 10.0)]))
    a = _toy_assumptions()
    idx = pd.date_range("2018-01-01", periods=48, freq="MS")
    origin = pd.Timestamp("2020-06-01")
    panel = pd.DataFrame({"D01": 10_000.0, "EL01": np.where(idx <= origin, 100.0, 110.0)}, index=idx)

    w = t.bom_weights(a)
    el = w.set_index("material").loc["electronics"]
    assert el["kg"] == 1.0 and el["eur_per_kg"] == 300.0 and el["series"] == "EL01"
    w_el = el["weight"]  # 300 / (10 kg × 10 EUR/kg + 300) = 0.75
    assert w_el == pytest.approx(0.75)

    derive = t.make_bom_deriver(panel, a)
    out = derive(panel, panel.loc[:origin], origin)["RM6_NE_IQI_433520"]
    assert out[origin] == pytest.approx(100.0)
    assert out[pd.Timestamp("2020-07-01")] == pytest.approx(100.0 * (1 + 0.10 * w_el))


# ---------------------------------------------------------------------------- 4. cibles par défaut
def test_default_targets_drop_empty_extras():
    idx = pd.date_range("2020-01-01", periods=5, freq="MS")
    panel = pd.DataFrame({"S01": 1.0, "L01": np.nan, "EP01": 2.0}, index=idx)
    tg = default_targets(panel)
    assert "S01" in tg and "EP01" in tg and "L01" not in tg
    assert tg[-2:] == ["RM6_NE_IQI_433520", "RMAirSeT_IIQ_24kV_IEC"]


# ---------------------------------------------------------------------------- 5. dernière valeur ≤ ancrage
def test_last_valid_at_or_before_anchor(monkeypatch):
    s = pd.Series([90.0, 100.0, np.nan], index=pd.date_range("2020-04-01", periods=3, freq="MS"), name="EL01")
    assert t.last_valid_at(s, "2020-06-01") == 100.0

    monkeypatch.setattr(d, "load_bom", lambda *a, **k: _toy_bom([("RM6", "Copper", 10.0)]))
    a = _toy_assumptions()
    idx = pd.date_range("2018-01-01", "2020-06-01", freq="MS")
    panel = pd.DataFrame({"D01": 10_000.0, "EL01": 100.0}, index=idx)
    panel.loc["2020-06-01", "EL01"] = np.nan  # indice pas encore publié au mois d'ancrage
    derive = t.make_bom_deriver(panel, a)
    wa = derive.weights_at(panel, "2020-06-01").set_index("material")
    assert wa.at["electronics", "value"] == pytest.approx(300.0)  # k = 300 / EL01(mai) = 3

    out, _ = t.add_bom_indices(panel, a)
    assert out.at[pd.Timestamp("2020-06-01"), "RM6_NE_IQI_433520"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------- 6. BOM v2
def test_load_bom_v2(tmp_path, monkeypatch):
    rm6 = "RM6_NE_IQI_433520"
    rows = [
        (rm6, "Carbon Steel EU (CR coil)", "Carbon steel", 0.50, 50.0),
        (rm6, "Epoxy Resin EU", "Plastic", 0.10, 10.0),
        (rm6, "Brass", "Brass", 0.05, 5.0),
        (rm6, "PP Polypropylene EU", "Plastic", 0.02, 2.0),
        (rm6, "Unidentified remainder", "To confirm", 0.05, 5.0),
        ("Total visible labelled mass – RM6 (kg)", None, None, None, 72.0),
        ("Source total mass – REDACTED / RM6 (kg)", None, None, None, 100.0),
    ]
    bom = pd.DataFrame(rows, columns=["product_ref", "material_label_source", "material_family", "share_pct",
                                      "mass_kg"])
    with pd.ExcelWriter(tmp_path / "RM_Prices_XHEC_Challenge_v1.xlsx") as xw:
        bom.to_excel(xw, sheet_name="BOM_Materials", index=False)
    monkeypatch.setattr(d, "RAW", tmp_path)

    out = d.load_bom()  # v2 par défaut
    fam = out.set_index("material_family")["mass_kg"]
    assert set(out["product"]) == {"RM6"} and set(out["bom_version"]) == {"v2"}
    assert fam["Epoxy"] == 10.0 and fam["Brass"] == 5.0 and fam["Plastic"] == 2.0
    assert "To confirm" not in fam.index
    assert out["implied_total_kg"].iloc[0] == pytest.approx(100.0)


# ---------------------------------------------------------------------------- 7. double comptage epoxy
def test_epoxy_double_counting_raises(monkeypatch):
    monkeypatch.setattr(d, "load_bom", lambda *a, **k: _toy_bom([("RM6", "Copper", 10.0), ("RM6", "Epoxy", 5.0)]))
    with pytest.raises(ValueError, match="deux fois"):
        t.bom_weights(_toy_assumptions(epoxy_extra=True))
    w = t.bom_weights(_toy_assumptions(epoxy_extra=False, electronics=False))
    assert w.set_index("material").at["epoxy", "series"] == "EP01"
