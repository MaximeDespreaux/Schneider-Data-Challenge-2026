"""Erwan's stack reads the official v1 BOM workbook (synthetic frames only, no real mass)."""

import pandas as pd
import pytest

from src import data_loading as dl
from src import features as fe


def _bom(rows):
    cols = ["product_ref", "plant_code", "plant_name", "material_label_source", "material_family", "share_pct",
            "mass_kg", "eu_series_ids", "ww_series_ids", "comment"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows], columns=cols)


def test_load_bom_reads_the_v1_workbook(monkeypatch):
    seen = {}
    monkeypatch.setattr(pd, "read_excel", lambda path, sheet_name: seen.update(path=path, sheet=sheet_name) or _bom([]))
    dl.load_bom()
    assert seen["path"].name == "RM_Prices_XHEC_Challenge_v1.xlsx" and seen["sheet"] == "BOM_Materials"


def test_bom_masses_v1_layout_keeps_material_rows_only(monkeypatch):
    v1 = _bom([("A", "P1", "x", "l1", "Carbon steel", 0.5, 10.0, "", "", ""),
               ("A", "P1", "x", "l2", "Copper", 0.2, 4.0, "", "", ""),
               ("A", "P1", "x", "l3", "Brass", 0.1, 2.0, "", "", ""),
               ("B", "P2", "y", "l1", "Stainless steel", 0.3, 6.0, "", "", ""),
               ("Total labelled mass A (kg)", None, None, None, None, None, 16.0, None, None, None)])
    monkeypatch.setattr(dl, "load_bom", lambda *a, **k: v1)
    m = fe.bom_masses()
    assert m.loc["CRU_CR_DE", "A"] == 10.0 and m.loc["D01", "A"] == 4.0 and m.loc["CRU_SS304_DE", "B"] == 6.0
    assert m.sum().sum() == pytest.approx(20.0)  # brass and the total row are left out


def test_bom_masses_old_layout_still_fills_the_unlabelled_rm6_row(monkeypatch):
    old = _bom([("RM6_NE_IQI_433520", "P", "x", "l1", "Stainless steel", 0.25, 5.0, "", "", ""),
                ("RM6_NE_IQI_433520", "P", "x", "l2", "Carbon steel (to confirm)", None, None, "", "", "")])
    monkeypatch.setattr(dl, "load_bom", lambda *a, **k: old)
    assert fe.bom_masses().loc["CRU_CR_DE", "RM6_NE_IQI_433520"] == pytest.approx(0.45 * 20.0)
