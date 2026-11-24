"""Licensed CRU/MEPS loader, tested on a SYNTHETIC workbook that mimics the layout (no licensed values)."""

from datetime import datetime

import openpyxl
import pandas as pd
import pytest

from src.licensed import LIC_COLUMNS, load_licensed

N = 30


def make_wb(tmp_path, drop: str | None = None):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "MEPS"
    ws["A1"], ws["B1"], ws["G1"] = "MEPS LOW", "Germany", "France"
    products = ["HR coil", "CR coil", "HDG coil", "EG coil"]
    ws["A5"] = "Month"
    for block, start in ((0, 2), (1, 7)):
        for j, prod in enumerate(products):
            if prod != drop:
                ws.cell(row=5, column=start + j, value=prod)
    for i in range(N):
        r = 6 + i
        ws.cell(row=r, column=1, value=datetime(2012 + (10 + i) // 12, (10 + i) % 12 + 1, 1))
        ws.cell(row=r, column=2, value=500 + i)   # HR DE
        ws.cell(row=r, column=3, value=600 + i)   # CR DE
        ws.cell(row=r, column=4, value=700 + i)   # HDG DE
        ws.cell(row=r, column=8, value=610 + i)   # CR FR
        ws.cell(row=r, column=9, value=710 + i)   # HDG FR
    ss = wb.create_sheet("CRU Stainless")
    ss["A1"], ss["B1"], ss["H1"] = "CRU Stainless", "Germany", "France"
    ss["B2"], ss["D2"], ss["H2"] = "Grade 304", "Grade 430", "Grade 304"
    ss["A5"] = "Month"
    for col, lab in ((2, "BASE"), (3, "Alloy"), (4, "BASE"), (5, "Alloy"), (8, "BASE"), (9, "Alloy")):
        ss.cell(row=5, column=col, value=lab)
    for i in range(N):
        r = 6 + i
        ss.cell(row=r, column=1, value=datetime(2012 + (10 + i) // 12, (10 + i) % 12 + 1, 15))
        ss.cell(row=r, column=2, value=2000 + i)
        ss.cell(row=r, column=3, value=1000 + i)
        ss.cell(row=r, column=4, value=1500 + i)
        ss.cell(row=r, column=5, value=100 + i)
    ss.cell(row=6 + 5, column=2).value = None  # blank base (cell(value=None) would not clear it)
    ss.cell(row=6 + 7, column=3, value="-")    # dash alloy
    path = tmp_path / "steel.xlsx"
    wb.save(path)
    return path


@pytest.fixture
def wb_path(tmp_path):
    return make_wb(tmp_path)


def test_columns_units_and_month_start(wb_path):
    df = load_licensed(wb_path)
    assert list(df.columns) == list(LIC_COLUMNS)
    assert df.index.min() == pd.Timestamp("2012-11-01") and (df.index.day == 1).all()
    assert df.at[pd.Timestamp("2012-12-01"), "L01"] == 601
    assert df.at[pd.Timestamp("2012-12-01"), "L01_FR"] == 611
    assert df.at[pd.Timestamp("2012-12-01"), "L03"] == 701
    assert df.at[pd.Timestamp("2012-12-01"), "L02"] == 2001 + 1001


def test_dash_and_blank_become_nan_not_zero(wb_path):
    df = load_licensed(wb_path)
    assert df["L02"].isna().sum() == 2 and (df.fillna(1) > 0).all().all()


def test_missing_label_raises(tmp_path):
    with pytest.raises(ValueError, match="HDG coil"):
        load_licensed(make_wb(tmp_path, drop="HDG coil"))
