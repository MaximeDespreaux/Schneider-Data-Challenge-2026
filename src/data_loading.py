"""Loaders for the Schneider × X-HEC Data Challenge 2026 files.

Every loader returns a tidy pandas DataFrame. Raw zips in ``data/`` are extracted
once into ``data/raw/`` (gitignored) by :func:`extract_archives`.

Conventions
-----------
- Monthly dates are normalised to month-start ``Timestamp`` in a column/index named ``month``.
- Long frames use columns ``month, series, value`` (+ descriptive columns).
"""

from __future__ import annotations

import re
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CORE = RAW / "data"  # contents of data.zip
SUPPLIERS = RAW / "Supplier_balancesheet"

STEEL_XLSX = DATA / "Copia de Europe Steel INDEXES.xlsx"
LME_XLSX = DATA / "LME_Copper_Aluminium_EU_Monthly_2016_2026.xlsx"
EPOXY_XLSX = DATA / "Epoxy for student.xlsx"
BENCH_XLSX = DATA / (
    "benchmark-modules-fonctionnels-rm6-ne-iqi-3-fonctions-400a-"
    "appareillage-isolement-et-coupure-dans-le-gaz-sf6.xlsx"
)
PRICE_XLSX = DATA / "RMU_IQI_EU_Price_Benchmark_2017-2026_Synthetic.xlsx"  # was DRM-locked; unlocked copy 18-Nov-2026
DRM_XLSX = PRICE_XLSX  # backward-compatible alias


# --------------------------------------------------------------------------- #
# Archives
# --------------------------------------------------------------------------- #
def extract_archives(force: bool = False) -> Path:
    """Unzip ``data.zip`` and ``Supplier_balancesheet.zip`` into ``data/raw``."""
    RAW.mkdir(parents=True, exist_ok=True)
    for name, marker in [("data.zip", CORE), ("Supplier_balancesheet.zip", SUPPLIERS)]:
        if force or not marker.exists():
            with zipfile.ZipFile(DATA / name) as zf:
                zf.extractall(RAW)
    return RAW


def _month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()


# --------------------------------------------------------------------------- #
# Core challenge set (data.zip)
# --------------------------------------------------------------------------- #
def load_rm_wide() -> pd.DataFrame:
    """Raw-material prices, one row per month (S01–S13, D01–D02, L01–L08)."""
    df = pd.read_csv(CORE / "rm_prices_monthly_wide.csv")
    df["month"] = _month(df["month"])
    return df.set_index("month").drop(columns=["year", "month_num"], errors="ignore")


def load_energy_wide() -> pd.DataFrame:
    """Energy prices, one row per month (E01–E24, ED01–ED06)."""
    df = pd.read_csv(CORE / "energy_prices_monthly_wide.csv")
    df["month"] = _month(df["month"])
    return df.set_index("month")


def load_bom() -> pd.DataFrame:
    return pd.read_csv(CORE / "bom_materials.csv")


def load_catalogs() -> dict[str, pd.DataFrame]:
    return {
        "rm": pd.read_csv(CORE / "source_catalog.csv"),
        "energy": pd.read_csv(CORE / "energy_source_catalog.csv"),
        "dictionary": pd.read_csv(CORE / "data_dictionary.csv"),
    }


# --------------------------------------------------------------------------- #
# Steel indexes workbook (MEPS / CRU)
# --------------------------------------------------------------------------- #
def _num(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")


def load_meps() -> pd.DataFrame:
    """MEPS flat-coil 'low' prices by country, long format (EUR/GBP/SEK per t)."""
    raw = pd.read_excel(STEEL_XLSX, sheet_name="MEPS", header=None)
    country = raw.iloc[0].ffill()
    unit = raw.iloc[1].ffill().astype(str).str.extract(r"(EUR|GBP|SEK)/t")[0]
    product = raw.iloc[4]
    rows = []
    body = raw.iloc[5:].copy()
    months = _month(body[0])
    for c in range(1, 45):  # flat-coil country blocks only
        p = product[c]
        if not isinstance(p, str) or "coil" not in p.lower() or country[c] == "x":
            continue
        v = pd.to_numeric(body[c], errors="coerce")
        rows.append(pd.DataFrame({"month": months, "source": "MEPS", "country": country[c],
                                  "product": p.replace(" coil", "").upper(), "market": "Domestic",
                                  "unit": unit[c], "value": v}))
    out = pd.concat(rows).dropna(subset=["month", "value"])
    # The workbook has two "Sweden" blocks: one EUR, one SEK -> unit disambiguates.
    return out.reset_index(drop=True)


def load_cru_flat() -> pd.DataFrame:
    """CRU flat-steel monthly averages / spot prices by country & product, long format."""
    raw = pd.read_excel(STEEL_XLSX, sheet_name="CRU Flat", header=None)
    kind, product, country, market, unit = (raw.iloc[i] for i in (1, 2, 3, 4, 5))
    body = raw.iloc[6:]
    months = _month(body[0])
    rows = []
    for c in range(1, raw.shape[1]):
        if kind[c] not in ("AVERAGE", "Spot price"):
            continue
        v = pd.to_numeric(body[c], errors="coerce")
        rows.append(pd.DataFrame({"month": months, "source": "CRU", "country": country[c],
                                  "product": str(product[c]).replace(" coil", "").upper(),
                                  "market": market[c], "unit": unit[c], "value": v}))
    return pd.concat(rows).dropna(subset=["month", "value"]).reset_index(drop=True)


def load_cru_stainless() -> pd.DataFrame:
    """CRU stainless CR-2mm base price + alloy surcharge; ``value`` = base + alloy."""
    raw = pd.read_excel(STEEL_XLSX, sheet_name="CRU Stainless", header=None)
    country = raw.iloc[0].ffill()
    grade = raw.iloc[1].ffill()
    unit = raw.iloc[3].ffill()
    body = raw.iloc[5:]
    months = _month(body[0])
    rows = []
    for c in range(1, raw.shape[1], 2):
        base = pd.to_numeric(body[c], errors="coerce")
        alloy = pd.to_numeric(body[c + 1], errors="coerce")
        rows.append(pd.DataFrame({"month": months, "source": "CRU", "country": country[c],
                                  "product": grade[c].replace("Grade ", "SS"), "unit": unit[c],
                                  "base": base, "alloy": alloy, "value": base + alloy}))
    return pd.concat(rows).dropna(subset=["month", "value"]).reset_index(drop=True)


def load_steel_mapping() -> pd.DataFrame:
    """Schneider part reference → steel index mapping (plant, index, lead time)."""
    raw = pd.read_excel(STEEL_XLSX, sheet_name="Reference to Index Mapping", header=1)
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")]
    raw.columns = [str(c).strip() if not isinstance(c, pd.Timestamp) else c for c in raw.columns]
    return raw.dropna(how="all")


# --------------------------------------------------------------------------- #
# Other workbooks
# --------------------------------------------------------------------------- #
def load_lme() -> pd.DataFrame:
    """LME copper & aluminium cash / 3M in USD and EUR (monthly, Jan-2016 → Nov-2026 MTD)."""
    df = pd.read_excel(LME_XLSX, sheet_name="Monthly Prices", header=4)
    df = df.dropna(subset=["Month"])
    df["month"] = _month(df["Month"])
    df = df.dropna(subset=["month"]).drop(columns="Month").set_index("month")
    return df


def load_epoxy() -> pd.DataFrame:
    """Epoxy resin and feedstock prices (EUR/t)."""
    df = pd.read_excel(EPOXY_XLSX, header=0)
    df = df.rename(columns={df.columns[0]: "month"}).dropna(axis=1, how="all")
    df["month"] = _month(df["month"])
    df = df.dropna(subset=["month"]).set_index("month")
    df.columns = [re.split(r"[_ (]", c.replace("OTHER COSTS", "OTHER_COSTS"))[0].title() if "RESIN" not in c
                  else "Epoxy_resin" for c in df.columns]
    return _num(df)


def load_price_benchmark() -> pd.DataFrame:
    """Synthetic yearly RMU IQI EU selling prices per brand (EUR ex-VAT), long format.

    Columns: brand_id, brand, year, p10, p50, p90. Values are labelled "AI/ML" in the file.
    """
    df = pd.read_excel(PRICE_XLSX, sheet_name="ML_Trend", header=3).dropna(subset=["Year"])
    df = df.rename(columns={"ID": "brand_id", "Brand": "brand", "Year": "year",
                            "P10 (ML)": "p10", "P50 (ML)": "p50", "P90 (ML)": "p90"})
    df["year"] = df["year"].astype(int)
    return df[["brand_id", "brand", "year", "p10", "p50", "p90"]].reset_index(drop=True)


def load_benchmark() -> pd.DataFrame:
    """Should-cost benchmark (Schneider RM6 vs 5 competitors), one row per product."""
    raw = pd.read_excel(BENCH_XLSX, sheet_name="Benchmark", header=None)
    names = raw.iloc[4, 1:].astype(str).str.replace("\n", " · ", regex=False)
    role = raw.iloc[3, 1:]
    body = raw.iloc[5:].set_index(0)
    body.columns = names
    keep = ["total", "material", "process", "labour", "overhead", "List price",
            "Mass (g)", "Made in", "Insulating gas", "Rated busbar current Ir (A)"]
    t = body.loc[body.index.isin(keep)].T
    t.insert(0, "role", role.values)
    for c in ["total", "material", "process", "labour", "overhead", "List price", "Mass (g)"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t.index.name = "product"
    return t


def _supplier_name(path: Path) -> str:
    m = re.search(r"PitchBook_-_(.+?)_Company_Financials", path.name)
    return m.group(1).replace("_", " ") if m else path.stem


def load_supplier_financials() -> pd.DataFrame:
    """All PitchBook statements (Income Statement, Balance Sheet, Ratios) in long format.

    Columns: supplier, statement, item, period, value (kEUR unless ratio).
    """
    rows = []
    for f in sorted(SUPPLIERS.glob("*.xlsx")):
        xl = pd.ExcelFile(f)
        for sheet in [s for s in ["Income Statement", "Balance Sheet", "Cash Flow", "Ratios"] if s in xl.sheet_names]:
            raw = xl.parse(sheet, header=None)
            # the period header row is the one containing 'Dec 20xx'-like labels
            hdr_row = next(i for i in range(12) if raw.iloc[i].astype(str).str.match(r"^[A-Z][a-z]{2} \d{4}$").any())
            periods = raw.iloc[hdr_row]
            pcols = [c for c in raw.columns if re.match(r"^[A-Z][a-z]{2} \d{4}$", str(periods[c]))]
            body = raw.iloc[hdr_row + 1:]
            for _, r in body.iterrows():
                item = r[2]
                if not isinstance(item, str):
                    continue
                for c in pcols:
                    v = pd.to_numeric(r[c], errors="coerce")
                    if pd.notna(v):
                        rows.append((_supplier_name(f), sheet, item.strip(), periods[c], v))
    df = pd.DataFrame(rows, columns=["supplier", "statement", "item", "period", "value"])
    df["year"] = df["period"].str[-4:].astype(int)
    return df


# --------------------------------------------------------------------------- #
# Inventory helper
# --------------------------------------------------------------------------- #
def is_drm_protected(path: Path) -> bool:
    """True for OLE-container 'xlsx' files (IRM / sensitivity-label encryption)."""
    with open(path, "rb") as fh:
        return fh.read(8) == bytes.fromhex("D0CF11E0A1B11AE1")


def file_inventory() -> pd.DataFrame:
    """One row per (file, sheet) with shape and readability for every challenge file."""
    rows = []
    files = sorted([p for p in DATA.iterdir() if p.is_file() and not p.name.startswith(".")])
    files += sorted(CORE.glob("*")) + sorted(SUPPLIERS.glob("*.xlsx"))
    for p in files:
        rel = p.relative_to(DATA).as_posix()
        size_kb = round(p.stat().st_size / 1024, 1)
        if p.suffix == ".xlsx":
            if is_drm_protected(p):
                rows.append((rel, "—", np.nan, np.nan, size_kb, "DRM-protected (unreadable)"))
                continue
            xl = pd.ExcelFile(p)
            for s in xl.sheet_names:
                d = xl.parse(s, header=None)
                rows.append((rel, s, d.shape[0], d.shape[1], size_kb, "ok"))
        elif p.suffix == ".csv":
            d = pd.read_csv(p)
            rows.append((rel, "—", d.shape[0], d.shape[1], size_kb, "ok"))
        else:
            rows.append((rel, "—", np.nan, np.nan, size_kb, p.suffix.lstrip(".") or "file"))
    return pd.DataFrame(rows, columns=["file", "sheet", "rows", "cols", "size_kb", "status"])
