"""Data profiling helpers + the preprocessing pipeline that writes ``data/processed/*.parquet``.

Used by ``notebooks/00_data_profiling_preprocessing.ipynb``.

Outputs (all in ``data/processed/``)
------------------------------------
- ``panel_monthly.parquet``         wide, month-start index 2012-01 → 2026-10, *raw* cleaned levels (NaN kept)
- ``panel_monthly_filled.parquet``  same grid, ragged right edge (and 1-month internal holes) imputed
- ``panel_impute_method.parquet``   same grid, string per cell: "" = observed, else the imputation method
- ``panel_monthly_long.parquet``    tidy: month, series, value, value_raw, is_imputed, impute_method + metadata
- ``series_catalog.parquet``        one row per panel series (source, unit, family, role, publication lag …)
- ``beyond_panel_end.parquet``      observations dated after the panel end (Nov-2026 LME MTD, CRU spot) for nowcasting
- ``bom.parquet``, ``steel_index_mapping.parquet``, ``competitor_benchmark.parquet``, ``teardown_bom.parquet``,
  ``price_benchmark.parquet``, ``supplier_financials.parquet``, ``supplier_kpis.parquet``

Imputation rules (filled panel only — the raw panel is never modified except for documented data errors)
-------------------------------------------------------------------------------------------------------
1. ``interp_internal``  single-month holes *inside* a series → linear interpolation.
   ``interp_data_error`` same, for cells set to NaN because they were a proven data-entry error (epoxy May-2026).
2. ``lme_proxy``        S01/S02 (IMF copper/aluminium USD) after their last month ← LME workbook cash USD × median
   ratio over the previous 12 months (the two sources agree to ~0.1 %).
3. ``identity``         derived series recomputed from their components (D01 = S01/S06, ED02 = E05·3.412/S06 …).
4. ``ffill``            remaining trailing gap of <= 3 months: forward-fill (random-walk nowcast). Series whose
                        trailing gap is longer (stopped / lagging sources) are left NaN and flagged in the catalog.
Nothing is ever filled *before* a series' first observation.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import data_loading as dl
from . import features as fe

PROCESSED = dl.DATA / "processed"
PANEL_START = "2012-01-01"
PANEL_END = "2026-10-01"
FFILL_LIMIT = 3

CC = {"Germany": "DE", "GERMANY": "DE", "France": "FR", "Italy": "IT", "Spain": "ES", "UK": "UK",
      "UK (GBP)": "UK", "Belgium": "BE", "Nordic": "NORDIC", "Sweden": "SE"}

# panel names of the target series used in src/features.py
FEATURES_RENAME = {"SS304_ALLOY_DE": "CRU_SS304_DE_ALLOY"}


# =========================================================================== #
# 1. Generic profiling helpers
# =========================================================================== #
def profile(df: pd.DataFrame, max_examples: int = 3) -> pd.DataFrame:
    """One row per column: dtype, counts, missingness, cardinality, range, examples."""
    rows = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        num = pd.to_numeric(s, errors="coerce") if s.dtype == object else s
        is_num = pd.api.types.is_numeric_dtype(num) and num.notna().sum() >= max(1, 0.8 * s.notna().sum())
        ex = s.dropna().astype(str).unique()[:max_examples]
        rows.append({
            "column": c, "dtype": str(s.dtype), "n_non_null": int(s.notna().sum()),
            "n_missing": int(s.isna().sum()), "pct_missing": round(100 * s.isna().mean(), 1) if n else np.nan,
            "n_unique": int(s.nunique(dropna=True)),
            "min": num.min() if is_num else None, "max": num.max() if is_num else None,
            "examples": " | ".join(e[:40] for e in ex),
        })
    return pd.DataFrame(rows).set_index("column")


def series_coverage(wide: pd.DataFrame) -> pd.DataFrame:
    """For a month-indexed wide frame: first/last month, leading / internal / trailing NaN counts."""
    rows = []
    for c in wide.columns:
        s = wide[c]
        fv, lv = s.first_valid_index(), s.last_valid_index()
        if fv is None:
            rows.append({"series": c, "first": pd.NaT, "last": pd.NaT, "n_obs": 0, "leading_nan": len(s),
                         "internal_nan": 0, "trailing_nan": 0, "internal_gap_months": ""})
            continue
        inner = s.loc[fv:lv]
        gaps = inner.index[inner.isna()]
        rows.append({"series": c, "first": fv, "last": lv, "n_obs": int(s.notna().sum()),
                     "leading_nan": int(s.loc[:fv].isna().sum()), "internal_nan": int(inner.isna().sum()),
                     "trailing_nan": int(s.loc[lv:].isna().sum()),
                     "internal_gap_months": ", ".join(g.strftime("%Y-%m") for g in gaps)})
    return pd.DataFrame(rows).set_index("series")


def identical_columns(df: pd.DataFrame, rtol: float = 1e-6) -> list[tuple[str, str]]:
    """Pairs of numeric columns that are equal (within rtol) wherever both are observed."""
    num = df.select_dtypes("number")
    cols, out = list(num.columns), []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            both = num[[a, b]].dropna()
            if len(both) >= 3 and np.allclose(both[a], both[b], rtol=rtol, atol=0):
                out.append((a, b))
    return out


def near_duplicate_columns(df: pd.DataFrame, thresh: float = 0.98, min_overlap: int = 24) -> pd.DataFrame:
    """Column pairs whose monthly % changes correlate above ``thresh`` (redundant information)."""
    r = df.select_dtypes("number").pct_change(fill_method=None)
    r = r.loc[:, r.std() > 0]
    c = r.corr(min_periods=min_overlap)
    iu = np.triu_indices_from(c, k=1)
    pairs = pd.DataFrame({"a": c.index[iu[0]], "b": c.columns[iu[1]], "corr_pct_change": c.values[iu]})
    return pairs[pairs.corr_pct_change >= thresh].sort_values("corr_pct_change", ascending=False).reset_index(drop=True)


def duplicate_report(df: pd.DataFrame, keys: list[str] | None = None) -> pd.Series:
    """Exact duplicate rows, duplicate keys and identical columns."""
    out = {"rows": len(df), "exact_duplicate_rows": int(df.duplicated().sum())}
    if keys:
        out["duplicate_keys" + str(tuple(keys))] = int(df.duplicated(subset=keys).sum())
    if isinstance(df.index, pd.DatetimeIndex):
        out["duplicate_index"] = int(df.index.duplicated().sum())
    out["identical_column_pairs"] = identical_columns(df) or "none"
    return pd.Series(out, dtype=object)


def file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def duplicate_files() -> pd.DataFrame:
    """Byte-identical files anywhere under data/ (excluding processed/)."""
    files = [p for p in dl.DATA.rglob("*") if p.is_file() and "processed" not in p.parts and not p.name.startswith(".")]
    h = pd.DataFrame({"file": [p.relative_to(dl.DATA).as_posix() for p in files], "md5": [file_md5(p) for p in files]})
    return h[h.md5.duplicated(keep=False)].sort_values("md5")


# =========================================================================== #
# 2. Extra loaders (sheets not covered by src/data_loading.py)
# =========================================================================== #
def load_cru_long() -> pd.DataFrame:
    """CRU long products (beams, merchant bar, rebar, wire rod) by country, long format."""
    raw = pd.read_excel(dl.STEEL_XLSX, sheet_name="CRU Long", header=None)
    country, product, unit = raw.iloc[0].ffill(), raw.iloc[1], raw.iloc[4]
    body = raw.iloc[5:]
    months = dl._month(body[0])
    rows = []
    for c in range(1, raw.shape[1]):
        if not isinstance(product[c], str):
            continue
        rows.append(pd.DataFrame({"month": months, "country": str(country[c]).title(), "product": product[c],
                                  "unit": unit[c], "value": pd.to_numeric(body[c], errors="coerce")}))
    return pd.concat(rows).dropna(subset=["month", "value"]).reset_index(drop=True)


def load_delta_sheet(sheet: str) -> pd.DataFrame:
    """The ▲ sheets (M-(M-1) or Q-(Q-1) price changes in EUR/t used by price-adjustment clauses), long."""
    raw = pd.read_excel(dl.STEEL_XLSX, sheet_name=sheet, header=None)
    product, country, unit, source = raw.iloc[0].ffill(), raw.iloc[1].ffill(), raw.iloc[2].ffill(), raw.iloc[3]
    body = raw.iloc[4:].dropna(subset=[0])
    rows = []
    for c in range(1, raw.shape[1]):
        if source[c] not in ("MEPS", "CRU"):
            continue
        rows.append(pd.DataFrame({"period": body[0].values, "product": str(product[c]).split(":")[0].strip(),
                                  "country": country[c], "unit": unit[c], "source": source[c],
                                  "delta": pd.to_numeric(body[c], errors="coerce").values}))
    return pd.concat(rows).reset_index(drop=True)


def load_teardown_boms() -> pd.DataFrame:
    """Part-level teardown BOMs of the 6 competitor products (one sheet each), long format."""
    xl = pd.ExcelFile(dl.BENCH_XLSX)
    rows = []
    for sh in [s for s in xl.sheet_names if s.startswith("BOM ")]:
        raw = pd.read_excel(xl, sheet_name=sh, header=None)
        product = str(raw.iat[0, 0]).replace("Teardown — ", "").strip()
        category = group = None
        for _, r in raw.iloc[2:].iterrows():
            label = r[0]
            if not isinstance(label, str):
                continue
            if pd.isna(r[1]) and "—" in label:
                name = label.split("—")[0].strip()
                if re.search(r"pcs\s*$", label):
                    group = name
                else:
                    category, group = name, None
                continue
            rows.append({"product": product, "category": category, "material_group": group, "part": label.strip(),
                         "material": r[1], "process": r[2], "mass_g": pd.to_numeric(r[3], errors="coerce"),
                         "qty": pd.to_numeric(r[4], errors="coerce"), "source_note": r[5],
                         "pricing": r[6] if len(r) > 6 else None})
    df = pd.DataFrame(rows)
    df["total_mass_g"] = df.mass_g * df.qty.fillna(1)
    df["mass_inferred"] = df.source_note.astype(str).str.contains("inferred")
    return df


# =========================================================================== #
# 3. Panel blocks — each returns a month-indexed wide frame + a metadata frame
# =========================================================================== #
def _meta(cols, **kw) -> pd.DataFrame:
    return pd.DataFrame({"series": list(cols), **kw}).set_index("series")


def rm_block():
    rm = dl.load_rm_wide()
    rm = rm.drop(columns=[c for c in rm if c.startswith("L")])  # licensed, 100 % empty on purpose
    cat = dl.load_catalogs()["rm"].set_index("series_id")
    meta = pd.DataFrame({"source_file": "data.zip/rm_prices_monthly_wide.csv",
                         "provider": cat.provider, "material": cat.material, "region": cat.region,
                         "description": cat.series_name, "unit": cat.unit, "family": "rm_free_proxy"}).reindex(rm.columns)
    meta.index.name = "series"
    meta.loc[["D01", "D02"], "family"] = "rm_free_proxy"
    return rm, meta


def energy_block():
    en = dl.load_energy_wide()
    assert np.allclose(en.E23.dropna(), dl.load_rm_wide().S06.reindex(en.E23.dropna().index)), "E23 != S06"
    en = en.drop(columns=["E23"])  # exact duplicate of S06 (USD per EUR)
    cat = dl.load_catalogs()["energy"].set_index("series_id")
    meta = pd.DataFrame({"source_file": "data.zip/energy_prices_monthly_wide.csv", "provider": cat.provider,
                         "material": cat.energy, "region": cat.region, "description": cat.series_name,
                         "unit": cat.unit, "family": "energy"}).reindex(en.columns)
    meta.index.name = "series"
    return en, meta


def cru_flat_block():
    cru = dl.load_cru_flat()
    cru = cru[(cru.market == "Domestic") & (cru["product"] != "EG")]  # EG discontinued 2017/2020; Italy imports: 10 months
    cru = cru.assign(series="CRU_" + cru["product"] + "_" + cru.country.map(CC))
    w = cru.pivot_table(index="month", columns="series", values="value", aggfunc="mean")
    m = cru.drop_duplicates("series").set_index("series")
    meta = pd.DataFrame({"source_file": "Copia de Europe Steel INDEXES.xlsx/CRU Flat", "provider": "CRU",
                         "material": m["product"] + " coil", "region": m.country, "unit": m.unit,
                         "description": "CRU " + m["product"] + " coil, domestic, monthly average/spot",
                         "family": "steel_market"}).reindex(w.columns)
    return w, meta


def meps_block():
    meps = dl.load_meps()
    meps = meps[~meps.country.isin(["Nordic", "Sweden"])]  # 18–30-month fragments only
    meps = meps.assign(series="MEPS_" + meps["product"] + "_" + meps.country.map(CC))
    w = meps.pivot_table(index="month", columns="series", values="value", aggfunc="mean")
    m = meps.drop_duplicates("series").set_index("series")
    meta = pd.DataFrame({"source_file": "Copia de Europe Steel INDEXES.xlsx/MEPS", "provider": "MEPS",
                         "material": m["product"] + " coil", "region": m.country.str.replace(" (GBP)", ""),
                         "unit": m.unit + "/t", "description": "MEPS " + m["product"] + " coil, low value",
                         "family": "steel_market"}).reindex(w.columns)
    return w, meta


def cru_stainless_block():
    ss = dl.load_cru_stainless()
    ss = ss.assign(stem="CRU_" + ss["product"] + "_" + ss.country.map(CC))
    parts = {"": "value", "_BASE": "base", "_ALLOY": "alloy"}
    frames, metas = [], []
    for suf, col in parts.items():
        w = ss.pivot_table(index="month", columns="stem", values=col, aggfunc="mean").add_suffix(suf)
        frames.append(w)
        m = ss.drop_duplicates("stem").set_index("stem")
        m.index = m.index + suf
        what = {"": "base + alloy surcharge (transaction price)", "_BASE": "base price", "_ALLOY": "alloy surcharge"}[suf]
        metas.append(pd.DataFrame({"source_file": "Copia de Europe Steel INDEXES.xlsx/CRU Stainless", "provider": "CRU",
                                   "material": "Stainless " + m["product"], "region": m.country, "unit": m.unit,
                                   "description": "CRU stainless CR-2mm " + m["product"] + ", " + what,
                                   "family": "stainless"}))
    w = pd.concat(frames, axis=1)
    return w, pd.concat(metas).reindex(w.columns)


def cru_long_block():
    lg = load_cru_long()
    slug = lg["product"].str.upper().str.replace(r"[^A-Z]+", "", regex=True)
    lg = lg.assign(series="CRU_LONG_" + slug + "_" + lg.country.map(CC).fillna(lg.country.str.upper()))
    w = lg.pivot_table(index="month", columns="series", values="value", aggfunc="mean")
    m = lg.drop_duplicates("series").set_index("series")
    meta = pd.DataFrame({"source_file": "Copia de Europe Steel INDEXES.xlsx/CRU Long", "provider": "CRU",
                         "material": m["product"], "region": m.country, "unit": m.unit,
                         "description": "CRU long product " + m["product"] + ", domestic delivered",
                         "family": "steel_long"}).reindex(w.columns)
    return w, meta


LME_COLS = {"Cu cash USD/t": "LME_CU_CASH_USD", "Cu 3m USD/t": "LME_CU_3M_USD", "Al cash USD/t": "LME_AL_CASH_USD",
            "Al 3m USD/t": "LME_AL_3M_USD", "Cu cash EUR/t": "LME_CU_CASH_EUR", "Cu 3m EUR/t": "LME_CU_3M_EUR",
            "Al cash EUR/t": "LME_AL_CASH_EUR", "Al 3m EUR/t": "LME_AL_3M_EUR", "Cu spread EUR/t": "LME_CU_SPREAD_EUR",
            "Al spread EUR/t": "LME_AL_SPREAD_EUR", "Cu avg stocks t": "LME_CU_STOCKS", "Al avg stocks t": "LME_AL_STOCKS"}


def lme_block(include_mtd: bool = False):
    lme = dl.load_lme()
    if not include_mtd:
        lme = lme[lme["Period status"].eq("Completed month")]
    w = lme[list(LME_COLS)].rename(columns=LME_COLS).apply(pd.to_numeric, errors="coerce")
    # relative cash–3M spread (>0 contango, <0 backwardation), as used in src/features.py
    w["LME_CU_SPREAD_REL"] = (w.LME_CU_3M_USD - w.LME_CU_CASH_USD) / w.LME_CU_CASH_USD
    w["LME_AL_SPREAD_REL"] = (w.LME_AL_3M_USD - w.LME_AL_CASH_USD) / w.LME_AL_CASH_USD
    units = {c: ("t" if "STOCKS" in c else "ratio" if "REL" in c else "EUR/t" if "EUR" in c else "USD/t") for c in w}
    meta = pd.DataFrame({"source_file": "LME_Copper_Aluminium_EU_Monthly_2016_2026.xlsx", "provider": "LME",
                         "material": ["Copper" if "_CU_" in c else "Aluminium" for c in w],
                         "region": "World", "unit": [units[c] for c in w],
                         "description": [c.replace("LME_", "LME ").replace("_", " ").lower() for c in w],
                         "family": "lme"}, index=pd.Index(w.columns, name="series"))
    return w, meta


EPOXY_COLS = {"Benzene": "EPOXY_BENZENE", "Acetone": "EPOXY_ACETONE", "Epychlorhydrine": "EPOXY_ECH",
              "Other": "EPOXY_OTHER", "Epoxy_resin": "EPOXY_RESIN"}
EPOXY_ERROR_MONTH = "2026-05-01"


def epoxy_block(fix_errors: bool = True):
    ep = dl.load_epoxy().rename(columns=EPOXY_COLS)
    if fix_errors:
        # Data-entry error: ECH = 257 EUR/t (neighbours ≈ 2,500); the resin price that month is computed from it.
        assert ep.loc[EPOXY_ERROR_MONTH, "EPOXY_ECH"] < 500, "expected epoxy data-entry error not found"
        ep.loc[EPOXY_ERROR_MONTH, ["EPOXY_ECH", "EPOXY_RESIN"]] = np.nan
    meta = pd.DataFrame({"source_file": "Epoxy for student.xlsx", "provider": "Schneider (epoxy cost build-up)",
                         "material": ["Benzene", "Acetone", "Epichlorohydrin", "Other costs", "Epoxy resin"],
                         "region": "EU", "unit": "EUR/t",
                         "description": ["feedstock", "feedstock", "feedstock", "conversion & other costs",
                                         "epoxy resin price (cost build-up)"],
                         "family": "epoxy"}, index=pd.Index(ep.columns, name="series"))
    return ep, meta


BLOCKS = {"rm": rm_block, "energy": energy_block, "cru_flat": cru_flat_block, "meps": meps_block,
          "cru_stainless": cru_stainless_block, "cru_long": cru_long_block, "lme": lme_block, "epoxy": epoxy_block}


def build_raw_panel(start: str = PANEL_START, end: str = PANEL_END) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge every block on a gap-free month-start grid. Returns (wide panel, series metadata)."""
    dl.extract_archives()
    frames, metas = [], []
    for name, fn in BLOCKS.items():
        w, m = fn()
        frames.append(w)
        metas.append(m.assign(block=name))
    panel = pd.concat(frames, axis=1).sort_index()
    assert not panel.columns.duplicated().any(), panel.columns[panel.columns.duplicated()]
    idx = pd.date_range(start, end, freq="MS", name="month")
    beyond = panel.loc[panel.index > pd.Timestamp(end)]
    panel = panel.reindex(idx)
    panel = panel.loc[:, panel.notna().any()]  # drop series with no data inside the window
    meta = pd.concat(metas)
    meta = meta.loc[panel.columns]
    return panel, meta


def beyond_panel_end(end: str = PANEL_END) -> pd.DataFrame:
    """Observations dated after ``end`` (LME Nov-2026 month-to-date, CRU Nov spot) — nowcasting inputs only."""
    frames = []
    for name, fn in BLOCKS.items():
        w = fn(include_mtd=True)[0] if name == "lme" else fn()[0]
        w = w.loc[w.index > pd.Timestamp(end)].dropna(axis=1, how="all")
        if len(w):
            frames.append(w.stack().rename("value").reset_index().rename(columns={"level_1": "series"})
                          .assign(block=name))
    out = pd.concat(frames, ignore_index=True)
    out.columns = ["month", "series", "value", "block"]
    out["is_provisional"] = out.block.eq("lme")  # LME row is month-to-date
    return out


# =========================================================================== #
# 4. Imputation (filled panel)
# =========================================================================== #
IDENTITIES = {  # target: (function of panel, components)
    "D01": (lambda p: p.S01 / p.S06, ["S01", "S06"]),
    "D02": (lambda p: p.S02 / p.S06, ["S02", "S06"]),
    "ED01": (lambda p: p.E01 / p.S06, ["E01", "S06"]),
    "ED02": (lambda p: p.E05 * 3.412 / p.S06, ["E05", "S06"]),
    "ED03": (lambda p: p.E07 * 3.412 / p.S06, ["E07", "S06"]),
    "ED04": (lambda p: p.E06 * 3.412 / p.S06, ["E06", "S06"]),
    "ED05": (lambda p: p.E12 * 1000 / p.E24, ["E12", "E24"]),
    "ED06": (lambda p: p.E20 / p.E21, ["E20", "E21"]),
}
LME_PROXY = {"S01": "LME_CU_CASH_USD", "S02": "LME_AL_CASH_USD"}


def _trailing_mask(s: pd.Series) -> pd.Series:
    lv = s.last_valid_index()
    return pd.Series(False, index=s.index) if lv is None else (s.index > lv) & s.isna()


def impute_panel(raw: pd.DataFrame, error_cells: list[tuple[str, str]] | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (filled panel, method frame). See module docstring for the rules."""
    filled = raw.copy()
    method = pd.DataFrame("", index=raw.index, columns=raw.columns)
    error_cells = error_cells or []

    # 1. single-month internal holes
    for c in filled:
        s = filled[c]
        fv, lv = s.first_valid_index(), s.last_valid_index()
        if fv is None:
            continue
        inner = s.loc[fv:lv]
        hole = inner.isna() & inner.shift(1).notna() & inner.shift(-1).notna()
        if hole.any():
            interp = inner.interpolate(limit=1, limit_area="inside")
            filled.loc[hole[hole].index, c] = interp[hole]
            method.loc[hole[hole].index, c] = "interp_internal"
    for m, c in error_cells:
        if method.at[pd.Timestamp(m), c] == "interp_internal":
            method.at[pd.Timestamp(m), c] = "interp_data_error"

    # 2. LME proxy for IMF copper / aluminium (USD)
    for tgt, src in LME_PROXY.items():
        if tgt not in filled or src not in filled:
            continue
        tr = _trailing_mask(filled[tgt])
        ratio = (filled[tgt] / filled[src]).dropna().iloc[-12:].median()
        val = filled[src] * ratio
        ok = tr & val.notna()
        filled.loc[ok, tgt] = val[ok]
        method.loc[ok, tgt] = "lme_proxy"

    # 3. exact identities
    for tgt, (f, comps) in IDENTITIES.items():
        if tgt not in filled or not all(c in filled for c in comps):
            continue
        tr = _trailing_mask(filled[tgt])
        val = f(filled)
        ok = tr & val.notna()
        filled.loc[ok, tgt] = val[ok]
        method.loc[ok, tgt] = "identity"

    # 4. forward-fill the remaining ragged edge — only when the whole trailing gap is <= FFILL_LIMIT months
    #    (series that stopped earlier are 'lagging'/'discontinued' and are left NaN on purpose)
    for c in filled:
        tr = _trailing_mask(filled[c])
        if not tr.any() or tr.sum() > FFILL_LIMIT:
            continue
        ff = filled[c].ffill(limit=FFILL_LIMIT)
        ok = tr & ff.notna()
        filled.loc[ok, c] = ff[ok]
        method.loc[ok, c] = "ffill"
    return filled, method


# =========================================================================== #
# 5. Catalog + long format
# =========================================================================== #
BOM_LINK = [  # (regex on series id, BOM material family)
    (r"^(CRU|MEPS)_CR_", "Carbon steel"), (r"^(CRU|MEPS)_HR_", "Carbon steel (upstream)"),
    (r"^(CRU|MEPS)_HDG_|^S10$", "Galvanised steel"), (r"^CRU_SS304", "Stainless steel"), (r"^S09$", "Carbon & stainless steel"),
    (r"^(D01|S01|S07|LME_CU_)", "Copper"), (r"^(D02|S02|S08|LME_AL_)", "Aluminium"), (r"^(S11|S13)$", "Plastics"),
    (r"^EPOXY_RESIN$", "Epoxy (insulators/bushings)"), (r"^S03$", "Stainless steel (nickel driver)"),
    (r"^S04$", "Carbon steel (iron-ore driver)"), (r"^S05$", "Galvanised steel (zinc driver)"),
]


def _pub_lag(series: str, family: str) -> int:
    inv = {v: k for k, v in FEATURES_RENAME.items()}
    if series in fe.PUB_LAGS:
        return fe.PUB_LAGS[series]
    if inv.get(series) in fe.PUB_LAGS:
        return fe.PUB_LAGS[inv[series]]
    return {"steel_market": 0, "stainless": 0, "steel_long": 0, "lme": 0, "epoxy": 0}.get(family, 1)


def build_catalog(raw: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    cov = series_coverage(raw)
    cat = meta.join(cov)
    targets = {FEATURES_RENAME.get(t, t) for t in fe.TARGETS}
    cat["role"] = np.where(cat.index.isin(targets), "target", "driver")
    cat["pub_lag_months"] = [_pub_lag(s, f) for s, f in zip(cat.index, cat.family)]
    link = pd.Series(pd.NA, index=cat.index, dtype=object)
    for pat, fam in BOM_LINK:
        hit = pd.Series(cat.index.str.contains(pat), index=cat.index) & link.isna()
        link[hit] = fam
    cat["bom_material"] = link
    cat["is_index"] = cat.unit.astype(str).str.contains("Index|ratio|share", case=False)
    cat["status"] = np.select([cat.trailing_nan <= FFILL_LIMIT, cat.trailing_nan <= 12], ["live", "lagging"],
                              default="discontinued")
    return cat


def to_long(raw: pd.DataFrame, filled: pd.DataFrame, method: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    def st(df, name):
        return df.stack(future_stack=True).rename(name)
    long = pd.concat([st(filled, "value"), st(raw, "value_raw"), st(method, "impute_method")], axis=1)
    long.index.names = ["month", "series"]
    long = long.reset_index().dropna(subset=["value"])
    long["is_imputed"] = long.impute_method.ne("")
    long = long.merge(catalog[["family", "provider", "unit", "role", "pub_lag_months", "bom_material"]],
                      left_on="series", right_index=True, how="left")
    return long.sort_values(["series", "month"]).reset_index(drop=True)


# =========================================================================== #
# 6. Static tables
# =========================================================================== #
def bom_clean() -> pd.DataFrame:
    """Official BOM (v1 workbook), materials only, with the price series used in modelling, assumption flags and the
    mass of the previous official version for comparison."""
    bom = dl.load_bom()
    bom = bom[bom.plant_code.notna()].copy()  # drops the 'Total … / Implied …' rows
    implied = (bom.mass_kg / bom.share_pct).groupby(bom.product_ref).median()
    bom["implied_total_mass_kg"] = bom.product_ref.map(implied).round(1)
    masses = fe.bom_masses()
    series = {"Carbon steel": "CRU_CR_DE", "Carbon steel (to confirm)": "CRU_CR_DE", "Galvanised steel": "CRU_HDG_DE",
              "Stainless steel": "CRU_SS304_DE", "Copper": "D01", "Aluminium": "D02",
              "Engineering plastic": "S11", "Plastic": "S11"}
    bom["model_series"] = bom.material_family.map(series)
    bom["mass_kg_model"] = bom.mass_kg
    m = bom.material_family.eq("Carbon steel (to confirm)")
    if m.any():  # older BOM versions only; v1 labels RM6 fully
        bom.loc[m, "mass_kg_model"] = masses.loc["CRU_CR_DE", "RM6_NE_IQI_433520"]
    bom["is_assumption"] = m
    root = dl.load_bom(dl.BOM_PREVIOUS_XLSX)
    root = root[root.plant_code.notna()]
    bom = bom.merge(root[["product_ref", "material_label_source", "mass_kg"]]
                    .rename(columns={"mass_kg": "mass_kg_root_xlsx"}), how="left",
                    on=["product_ref", "material_label_source"])
    return bom.reset_index(drop=True)


MAPPING_STATIC = ["Key", "Plant Name", "Unique ID", "Reference", "UoM", "ERP Code", "Factory", "Index",
                  "Lead Time (Months)", "Weight", "Plant country", "Index country acc to CRU", "Description",
                  "Grade in RFQ", "Grade\nCorrected", "R.M GRADE", "FORM", "THICKNESS", "Width", "Length",
                  "weight / PT", "Star Family", "Star Description", "Cluster", "Cdty", "H1 2017 allocation",
                  "Base Prix Arcelor offers Q4-2016", "Devise"]

MAPPING_INDEX_TO_SERIES = {"CR": "CRU_CR", "HDG": "CRU_HDG", "HR": "CRU_HR", "EZ": "MEPS_EG"}


def mapping_clean() -> pd.DataFrame:
    mp = dl.load_steel_mapping()
    mp = mp[[c for c in MAPPING_STATIC if c in mp.columns]].copy()
    mp.columns = [re.sub(r"[^0-9a-z]+", "_", c.lower()).strip("_") for c in mp.columns]
    grade = mp["index"].str.split().str[0]
    country = mp["index"].str.split().str[1:].str.join(" ")
    cc = country.map({**CC, "Benelux": "BE"})
    # Benelux HDG has no CRU quote -> MEPS Belgium; EZ (electro-zinc) -> MEPS EG
    mp["panel_series"] = np.where(country.eq("Benelux") | grade.eq("EZ"),
                                  "MEPS_" + grade.replace({"EZ": "EG"}) + "_" + cc,
                                  grade.map(MAPPING_INDEX_TO_SERIES) + "_" + cc)
    for c in ["weight", "thickness", "width", "length", "weight_pt", "base_prix_arcelor_offers_q4_2016",
              "lead_time_months"]:
        if c in mp:
            mp[c] = pd.to_numeric(mp[c], errors="coerce")
    return mp.reset_index(drop=True)


def competitor_benchmark() -> pd.DataFrame:
    b = dl.load_benchmark().reset_index()
    b["brand"] = b["product"].str.split(" · ").str[0]
    b["is_valid"] = b.total > 100  # SGC = €19 → broken teardown
    b["material_share"] = b.material / b.total
    b["eur_per_kg"] = b.total / (b["Mass (g)"] / 1000)
    b.columns = [re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_").lower() for c in b.columns]
    for c in b.columns:
        if b[c].dtype == object:
            b[c] = b[c].astype("string")
    return b


def supplier_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    fin = dl.load_supplier_financials()
    fin.loc[(fin.statement == "Income Statement") & (fin.item == "Total Revenue"), "item"] = "Revenue"
    fin = fin.drop_duplicates(["supplier", "statement", "item", "period"])

    def latest(item, st=None):
        f = fin[(fin.item == item) & ((fin.statement == st) if st else True)]
        return f.sort_values("year").groupby("supplier").agg(v=("value", "last"), y=("year", "last"))

    k = pd.concat({
        "first_year": fin.groupby("supplier").year.min(), "last_year": fin.groupby("supplier").year.max(),
        "n_datapoints": fin.groupby("supplier").size(),
        "revenue_keur": latest("Revenue", "Income Statement").v, "revenue_year": latest("Revenue", "Income Statement").y,
        "ebitda_keur": latest("EBITDA", "Income Statement").v, "ebitda_margin": latest("EBITDA Margin").v,
        "total_debt_keur": latest("Total Debt", "Balance Sheet").v, "total_equity_keur": latest("Total Equity", "Balance Sheet").v,
        "current_ratio": latest("Current Ratio").v, "interest_coverage": latest("Interest Coverage").v,
        "pitchbook_debt_to_equity": latest("Debt to Equity").v,
    }, axis=1)
    k["debt_to_equity"] = k.total_debt_keur / k.total_equity_keur
    k["debt_to_ebitda"] = k.total_debt_keur / k.ebitda_keur.where(k.ebitda_keur > 0)
    k["flag"] = np.select([k.ebitda_keur < 0, k.debt_to_ebitda > 10, k.current_ratio < 1, k.revenue_keur.isna()],
                          ["negative EBITDA", "debt/EBITDA > 10", "current ratio < 1", "no P&L"], default="")
    return fin.reset_index(drop=True), k.reset_index().rename(columns={"index": "supplier"})


# =========================================================================== #
# 7. Orchestration
# =========================================================================== #
def run_pipeline() -> dict[str, pd.DataFrame]:
    raw, meta = build_raw_panel()
    filled, method = impute_panel(raw, error_cells=[(EPOXY_ERROR_MONTH, "EPOXY_ECH"), (EPOXY_ERROR_MONTH, "EPOXY_RESIN")])
    catalog = build_catalog(raw, meta)
    catalog["n_imputed"] = method.ne("").sum()
    fin, kpis = supplier_tables()
    return {
        "panel_monthly": raw, "panel_monthly_filled": filled, "panel_impute_method": method,
        "panel_monthly_long": to_long(raw, filled, method, catalog), "series_catalog": catalog,
        "beyond_panel_end": beyond_panel_end(), "bom": bom_clean(), "steel_index_mapping": mapping_clean(),
        "competitor_benchmark": competitor_benchmark(), "teardown_bom": load_teardown_boms(),
        "price_benchmark": dl.load_price_benchmark(), "supplier_financials": fin, "supplier_kpis": kpis,
    }


def write_processed(tables: dict[str, pd.DataFrame], out: Path = PROCESSED) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, df in tables.items():
        df = df.copy()
        for c in df.columns:  # parquet needs homogeneous object columns
            if df[c].dtype == object:
                df[c] = df[c].astype("string")
        path = out / f"{name}.parquet"
        df.to_parquet(path)
        rows.append({"file": path.relative_to(dl.ROOT).as_posix(), "rows": df.shape[0], "cols": df.shape[1],
                     "size_kb": round(path.stat().st_size / 1024, 1)})
    return pd.DataFrame(rows)
