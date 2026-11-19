"""Licensed CRU / MEPS steel prices (inputs only, NEVER targets, NEVER written to outputs or git).

The workbook (``Copia de Europe Steel INDEXES.xlsx``) sits in data/raw/ (gitignored). Columns are located by their
header labels, never by fixed letters, so a reshuffled export fails loudly instead of reading the wrong series.

Output: EUR/t, month-start index, columns ``LIC_COLUMNS``. ``LIC_CRU_SS304_DE`` = base price + alloy surcharge.
The ``LIC_`` prefix avoids any clash with the empty official L01-L08 columns.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from src.data import RAW

LICENSED_FILE = RAW / "Copia de Europe Steel INDEXES.xlsx"
LIC_COLUMNS = ("LIC_MEPS_CR_DE", "LIC_MEPS_HDG_DE", "LIC_MEPS_CR_FR", "LIC_CRU_SS304_DE")

# column -> (sheet, header path ...). Header rows: MEPS = (country row 1, product row 5);
# CRU Stainless = (country row 1, grade row 2, component row 5).
MEPS_SPEC = {"LIC_MEPS_CR_DE": ("Germany", "CR coil"), "LIC_MEPS_HDG_DE": ("Germany", "HDG coil"),
             "LIC_MEPS_CR_FR": ("France", "CR coil")}
SS_SPEC = {"LIC_CRU_SS304_DE": ("Germany", "Grade 304")}


def _num(v) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else np.nan


def _read(ws, header_rows: list[int], first_data_row: int):
    rows = list(ws.iter_rows(min_row=1, values_only=True))
    width = max(len(r) for r in rows)
    headers = []
    for r in header_rows:
        vals = list(rows[r - 1]) + [None] * (width - len(rows[r - 1]))
        headers.append(vals)
    # forward-fill block labels (country, grade) across the block; leave the last header row as is
    for h in headers[:-1]:
        last = None
        for j in range(1, width):
            if h[j] is not None and str(h[j]).strip():
                last = str(h[j]).strip()
            h[j] = last
    data = [r for r in rows[first_data_row - 1:] if isinstance(r[0], datetime)]
    idx = pd.DatetimeIndex([r[0] for r in data]).to_period("M").to_timestamp()
    if not idx.is_unique:
        raise ValueError(f"sheet {ws.title}: duplicate months")
    return headers, data, idx


def _find(headers, labels: tuple[str, ...], sheet: str) -> int:
    hits = [j for j in range(1, len(headers[0]))
            if all(str(h[j] or "").strip() == lab for h, lab in zip(headers, labels))]
    if len(hits) != 1:
        raise ValueError(f"sheet {sheet}: expected one column {labels}, found {len(hits)}")
    return hits[0]


def load_licensed(path: Path = LICENSED_FILE) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = {}
    headers, data, idx = _read(wb["MEPS"], [1, 5], 6)
    for col, labels in MEPS_SPEC.items():
        j = _find(headers, labels, "MEPS")
        out[col] = pd.Series([_num(r[j]) if j < len(r) else np.nan for r in data], index=idx)
    headers, data, idx = _read(wb["CRU Stainless"], [1, 2, 5], 6)
    for col, (country, grade) in SS_SPEC.items():
        jb = _find(headers, (country, grade, "BASE"), "CRU Stainless")
        ja = _find(headers, (country, grade, "Alloy"), "CRU Stainless")
        out[col] = pd.Series([_num(r[jb]) + _num(r[ja]) for r in data], index=idx)
    wb.close()
    df = pd.DataFrame(out).sort_index()[list(LIC_COLUMNS)]
    df = df.where(df > 0)
    empty = [c for c in df if df[c].isna().all()]
    if empty:
        raise ValueError(f"licensed columns with no data: {empty}")
    return df
