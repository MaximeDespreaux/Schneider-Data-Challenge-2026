"""Target panel, exogenous drivers and leakage-safe feature engineering.

Forecasting convention
----------------------
* An *origin* ``t`` is the month of the latest observation of the target series.
* For horizon ``h`` the target is the cumulative log-change ``log(y[t+h] / y[t])``
  (direct multi-horizon: one model per h).
* **Publication lags.** A target with publication lag ``L_y`` is only known at real
  time ``t + L_y``. A driver with lag ``L_x`` is then known up to month
  ``t + L_y - L_x``. We use the conservative shift ``max(0, L_x - L_y)``, so a driver
  never carries information that was unpublished when the forecast was made.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_loading as dl

# months between the end of the reference month and publication (conservative)
PUB_LAGS: dict[str, int] = {
    # IMF via FRED (metals, energy) & derived EUR prices
    **{s: 1 for s in ["S01", "S02", "S03", "S04", "S05", "D01", "D02",
                      "E01", "E02", "E03", "E05", "E06", "E07", "E08", "E22", "ED01", "ED02", "ED03", "ED04"]},
    # Eurostat PPIs (1–2 months; take 2) and US BLS PPIs
    **{s: 2 for s in ["S07", "S08", "S09", "S10", "S11", "E13", "E14", "E15", "E16"]},
    **{s: 1 for s in ["S12", "S13", "E17", "E18"]},
    # Eurostat crude imports lag 3–4 months
    **{s: 4 for s in ["E20", "E21", "ED06"]},
    # Bank of Japan CGPI: published around the middle of the following month
    **{s: 1 for s in ["E04", "E19"]},
    # ECB FX, Ember power, JEPX, CRU/MEPS market prices, LME workbook: known at month end.
    # Convention: here a decision is taken at month END, so a monthly average of month t is known at t. The team test
    # bench (config/availability.yaml) decides MID-month, hence lag 1 for the same series.
    **{s: 0 for s in ["S06", "E23", "E24", "E09", "E10", "E11", "E12", "ED05",
                      "CRU_CR_DE", "CRU_CR_FR", "CRU_HDG_DE", "CRU_SS304_DE", "SS304_ALLOY_DE",
                      "cu_spread", "al_spread", "cu_stocks", "al_stocks"]},
}

TARGETS = ["CRU_CR_DE", "CRU_CR_FR", "CRU_HDG_DE", "CRU_SS304_DE", "SS304_ALLOY_DE",
           "D01", "D02", "S03", "S04", "S11", "S09", "S10"]
FAMILY = {"CRU_CR_DE": "steel_market", "CRU_CR_FR": "steel_market", "CRU_HDG_DE": "steel_market",
          "CRU_SS304_DE": "stainless", "SS304_ALLOY_DE": "stainless", "D01": "lme", "D02": "lme",
          "S03": "lme", "S04": "ore", "S11": "ppi", "S09": "ppi", "S10": "ppi"}

# drivers used by the per-series dynamic regression (tier 2)
DRIVERS: dict[str, list[str]] = {
    "CRU_CR_DE": ["S04", "ED02", "E09", "S06", "gap_steel"],
    "CRU_CR_FR": ["S04", "ED02", "E09", "S06", "gap_steel"],
    "CRU_HDG_DE": ["S04", "S05", "ED02", "E09", "S06", "gap_steel"],
    "CRU_SS304_DE": ["S03", "S06"],
    "SS304_ALLOY_DE": ["S03", "S06"],
    "D01": ["S06", "ED01", "cu_spread", "cu_stocks"],
    "D02": ["S06", "E13", "ED02", "al_spread", "al_stocks"],
    "S03": ["S06", "ED01"],
    "S04": ["S06", "ED01"],
    "S11": ["ED01", "E22"],
    "S09": ["CRU_CR_DE", "ED02", "gap_steel"],
    "S10": ["CRU_HDG_DE", "ED02", "gap_steel"],
}
COMMON_DRIVERS = ["ED01", "ED02", "E09", "E13", "S06", "S03", "S04", "D01", "D02", "S09",
                  "CRU_CR_DE", "cu_spread", "al_spread", "cu_stocks", "al_stocks", "gap_steel"]
LEVEL_FEATURES = {"cu_spread", "al_spread", "gap_steel"}  # used in levels/deviation, not log-changes


def _cru(df, **kw):
    m = np.logical_and.reduce([df[k] == v for k, v in kw.items()])
    return df.loc[m].groupby("month").value.mean()


def build_panel(start: str = "2012-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (targets, exog): month-start indexed wide frames of raw price levels."""
    dl.extract_archives()
    rm, en, lme = dl.load_rm_wide(), dl.load_energy_wide(), dl.load_lme()
    cru, ss = dl.load_cru_flat(), dl.load_cru_stainless()
    ss_de = ss.query("country == 'Germany' and product == 'SS304'").groupby("month")[["value", "alloy"]].mean()
    market = pd.DataFrame({
        "CRU_CR_DE": _cru(cru, country="Germany", product="CR", market="Domestic"),
        "CRU_CR_FR": _cru(cru, country="France", product="CR", market="Domestic"),
        "CRU_HDG_DE": _cru(cru, country="Germany", product="HDG", market="Domestic"),
        "CRU_SS304_DE": ss_de["value"],
        "SS304_ALLOY_DE": ss_de["alloy"],
    })
    lme_x = pd.DataFrame({
        "cu_spread": (lme["Cu 3m USD/t"] - lme["Cu cash USD/t"]) / lme["Cu cash USD/t"],
        "al_spread": (lme["Al 3m USD/t"] - lme["Al cash USD/t"]) / lme["Al cash USD/t"],
        "cu_stocks": lme["Cu avg stocks t"], "al_stocks": lme["Al avg stocks t"],
    })
    # drop the provisional month-to-date row of the LME workbook
    lme_x = lme_x.loc[lme["Period status"].eq("Completed month")]
    allx = pd.concat([market, rm.drop(columns=[c for c in rm if c.startswith("L")]), en, lme_x], axis=1).sort_index()
    allx = allx.loc[start:"2026-10-01"].asfreq("MS")
    targets = allx[TARGETS].copy()
    exog = allx.drop(columns=[c for c in TARGETS if c not in COMMON_DRIVERS + sum(DRIVERS.values(), [])]).copy()
    return targets, exog


_CATALOG_LAGS: dict[str, int] | None = None


def pub_lag(series: str, default: int = 1) -> int:
    """Publication lag in months: ``PUB_LAGS`` first, then ``series_catalog.parquet`` (if built), else ``default``."""
    global _CATALOG_LAGS
    if series in PUB_LAGS:
        return PUB_LAGS[series]
    if _CATALOG_LAGS is None:
        path = dl.DATA / "processed" / "series_catalog.parquet"
        _CATALOG_LAGS = (pd.read_parquet(path, columns=["pub_lag_months"]).pub_lag_months.astype(int).to_dict()
                         if path.exists() else {})
    return _CATALOG_LAGS.get(series, default)


def has_declared_lag(series: str) -> bool:
    """True if the publication lag of ``series`` is declared (``PUB_LAGS`` or the catalog), not the silent default."""
    return pub_lag(series, default=-1) >= 0


def published_by(value_month: pd.Series, value_series: pd.Series, origin: pd.Timestamp, target: str) -> pd.Series:
    """Was the value of ``value_series`` for ``value_month`` published when ``target`` was forecast from ``origin``?

    The forecast of ``target`` is made when its own month ``origin`` is published, i.e. at ``origin + lag(target)``.
    For the same series this is simply ``value_month <= origin``; it differs when residuals or training rows are
    pooled over series with different publication lags."""
    vm = pd.DatetimeIndex(value_month)
    lag = pd.Series(value_series).map({s: pub_lag(s) for s in pd.unique(pd.Series(value_series))}).to_numpy()
    known = vm.year * 12 + vm.month + lag
    made = origin.year * 12 + origin.month + pub_lag(target)
    return pd.Series(np.asarray(known <= made), index=getattr(value_month, "index", None))


def _shift_for(driver: str, target: str) -> int:
    base = "S09" if driver == "gap_steel" else driver  # the gap uses the Eurostat PPI -> its lag
    return max(0, pub_lag(base) - pub_lag(target))


def _driver_frame(exog: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    x = exog.copy()
    x["gap_steel"] = np.log(targets["CRU_CR_DE"]) - np.log(exog["S09"]) if "S09" in exog else np.nan
    if "CRU_CR_DE" not in x:
        x["CRU_CR_DE"] = targets["CRU_CR_DE"]
    if "CRU_HDG_DE" not in x:
        x["CRU_HDG_DE"] = targets["CRU_HDG_DE"]
    return x


def make_features(targets: pd.DataFrame, exog: pd.DataFrame, h: int,
                  drivers: list[str] | None = None) -> pd.DataFrame:
    """Long frame indexed by (month, series) with features at origin t and target y = log(y[t+h]/y[t]).

    All features use data dated <= t (minus the publication shift); only the column ``y`` looks ahead.
    """
    drivers = drivers or COMMON_DRIVERS
    X = _driver_frame(exog, targets)
    frames = []
    for s in targets.columns:
        y = targets[s]
        ly = np.log(y)
        f = pd.DataFrame(index=targets.index)
        f["series"], f["family"] = s, FAMILY[s]
        f["y_level"] = y
        f["y"] = ly.shift(-h) - ly
        d1 = ly.diff()
        for k in (1, 2, 3):
            f[f"own_dlog_l{k-1}"] = d1.shift(k - 1)
        for k in (3, 6, 12):
            f[f"own_mom{k}"] = ly - ly.shift(k)
        f["own_vol6"] = d1.rolling(6).std()
        f["own_vol12"] = d1.rolling(12).std()
        f["own_dev_ma12"] = ly - np.log(y.rolling(12).mean())
        f["own_dev_ma24"] = ly - np.log(y.rolling(24).mean())
        for d in drivers:
            if d not in X:
                continue
            xs = X[d].shift(_shift_for(d, s))
            if d in LEVEL_FEATURES:
                f[f"x_{d}"] = xs
                f[f"x_{d}_dev24"] = xs - xs.rolling(24, min_periods=12).mean()
            else:
                lx = np.log(xs.where(xs > 0))
                f[f"x_{d}_d1"] = lx.diff()
                f[f"x_{d}_d3"] = lx - lx.shift(3)
        frames.append(f)
    # an origin is usable only where y_level (the target at t) is observed
    return pd.concat(frames).rename_axis("month").set_index("series", append=True).sort_index()


def check_no_leakage(targets: pd.DataFrame, exog: pd.DataFrame, h: int, origin: str) -> None:
    """Assert that features at months <= origin are unchanged when every value after origin is scrambled."""
    rng = np.random.default_rng(0)
    T = pd.Timestamp(origin)
    t2, x2 = targets.copy(), exog.copy()
    for df in (t2, x2):
        fut = df.index > T
        df.loc[fut] = df.loc[fut] * rng.uniform(0.5, 1.5, size=df.loc[fut].shape)
    a = make_features(targets, exog, h).loc[:T].drop(columns="y")
    b = make_features(t2, x2, h).loc[:T].drop(columns="y")
    pd.testing.assert_frame_equal(a, b)


def bom_masses() -> pd.DataFrame:
    """BOM masses (data.zip version) per product × price series, kg per unit.

    Assumption (flagged in DATA_OVERVIEW.md): RM6 carbon steel = 45% × implied total mass.
    """
    bom = dl.load_bom().dropna(subset=["product_ref"]).iloc[:14].copy()
    rm6 = bom.query("product_ref == 'RM6_NE_IQI_433520' and material_family == 'Stainless steel'")
    total = (rm6.mass_kg / rm6.share_pct).iat[0]
    bom.loc[bom.material_family == "Carbon steel (to confirm)", ["material_family", "mass_kg"]] = ["Carbon steel", round(0.45 * total, 1)]
    series = {"Carbon steel": "CRU_CR_DE", "Galvanised steel": "CRU_HDG_DE", "Stainless steel": "CRU_SS304_DE",
              "Copper": "D01", "Aluminium": "D02", "Engineering plastic": "S11", "Plastic": "S11"}
    bom["series"] = bom.material_family.map(series)
    return bom.dropna(subset=["series", "mass_kg"]).groupby(["product_ref", "series"]).mass_kg.sum().unstack(0)
