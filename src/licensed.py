"""Licensed CRU / MEPS steel prices (EUR/t). Forecast internally; NEVER in git, CI logs or the submission CSV.

The workbook (``Copia de Europe Steel INDEXES.xlsx``) sits in data/raw/ (gitignored). Columns are located by their
header labels, never by fixed letters, so a reshuffled export fails loudly instead of reading the wrong series.

Output: EUR/t, month-start index named "month". Column ids follow Schneider's BOM mapping (eu_series_ids):
L01 = MEPS Germany CR coil, L02 = CRU Germany 304 CR-2mm (base + alloy surcharge), L03 = MEPS Germany HDG coil,
plus L01_FR = MEPS France CR coil (regional index; optional, skipped if absent).
Schneider allowed MEPS/CRU use (2026-11-19). Licensed: never in git, CI logs or the submission CSV.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

import src.data as _d

FILE_NAMES = ("Copia de Europe Steel INDEXES.xlsx", "Copia_de_Europe_Steel_INDEXES.xlsx")
LIC_COLUMNS = ("L01", "L02", "L03", "L01_FR")
OPTIONAL = {"L01_FR"}

# column -> (sheet, header path ...). Header rows: MEPS = (country row 1, product row 5);
# CRU Stainless = (country row 1, grade row 2, component row 5).
MEPS_SPEC = {"L01": ("Germany", "CR coil"), "L03": ("Germany", "HDG coil"), "L01_FR": ("France", "CR coil")}
SS_SPEC = {"L02": ("Germany", "Grade 304")}


def licensed_file() -> Path:
    for name in FILE_NAMES:
        if (_d.RAW / name).exists():
            return _d.RAW / name
    raise FileNotFoundError(f"Fichier acier MEPS/CRU introuvable dans {_d.RAW} ({' ou '.join(FILE_NAMES)}) : "
                            "fichier sous licence, à placer en local")


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


def load_licensed(path: Path | None = None) -> pd.DataFrame:
    path = licensed_file() if path is None else path
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = {}
    headers, data, idx = _read(wb["MEPS"], [1, 5], 6)
    for col, labels in MEPS_SPEC.items():
        try:
            j = _find(headers, labels, "MEPS")
        except ValueError:
            if col in OPTIONAL:
                continue
            raise
        out[col] = pd.Series([_num(r[j]) if j < len(r) else np.nan for r in data], index=idx)
    headers, data, idx = _read(wb["CRU Stainless"], [1, 2, 5], 6)
    for col, (country, grade) in SS_SPEC.items():
        jb = _find(headers, (country, grade, "BASE"), "CRU Stainless")
        ja = _find(headers, (country, grade, "Alloy"), "CRU Stainless")
        out[col] = pd.Series([_num(r[jb]) + _num(r[ja]) for r in data], index=idx)
    wb.close()
    df = pd.DataFrame(out).sort_index()[[c for c in LIC_COLUMNS if c in out]]
    df = df.where(df > 0).astype(float)
    df.index.name = "month"
    empty = [c for c in df if df[c].isna().all()]
    if empty:
        raise ValueError(f"licensed columns with no data: {empty}")
    return df
