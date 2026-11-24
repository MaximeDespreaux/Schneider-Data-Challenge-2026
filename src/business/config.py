"""Shared configuration of the business modules: config/business.yaml (tracked, no licensed values) plus helpers
on config/assumptions.yaml (local, gitignored)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src import data as d

CONFIG = d.ROOT / "config" / "business.yaml"
SUPPLIERS_LOCAL = d.ROOT / "config" / "suppliers.local.yaml"  # git-ignored (NDA): names, reference values
BOM_IDS = {"RM6": "RM6_NE_IQI_433520", "AirSeT": "RMAirSeT_IIQ_24kV_IEC"}
PRODUCT_KEY = {"RM6": "rm6", "AirSeT": "airset"}  # product name -> key used in assumptions.yaml


def load_business_config(path: Path = CONFIG, suppliers_path: Path = SUPPLIERS_LOCAL) -> dict:
    """``business.yaml``; the supplier list and reference values come from the git-ignored local file if it exists."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    local = {}
    if suppliers_path.exists():
        with open(suppliers_path, encoding="utf-8") as f:
            local = yaml.safe_load(f) or {}
    for key, empty in [("companies", []), ("expected_keur", {}), ("expected_rule", {})]:
        cfg["suppliers"][key] = local.get(key) or cfg["suppliers"].get(key) or empty
    return cfg


def selling_prices(assumptions: dict, cfg: dict) -> dict:
    """{"RM6": price, "AirSeT": price, "airset_price_assumed": bool} in EUR per box.

    AirSeT = pricing.airset_selling_price_eur if filled, else RM6 × (1 + airset_price_fallback_premium)."""
    p = assumptions["pricing"]
    rm6 = float(p["rm6_selling_price_eur"]["value"])
    air = p["airset_selling_price_eur"]["value"]
    assumed = air is None
    if assumed:
        air = rm6 * (1 + float(cfg["pricing"]["airset_price_fallback_premium"]))
    return {"RM6": rm6, "AirSeT": float(air), "airset_price_assumed": assumed}


def origin_ts(cfg: dict) -> pd.Timestamp:
    return pd.Timestamp(cfg["origin_month"] + "-01")


def last_valid(s: pd.Series, month) -> float:
    """Last non-NaN value at or before ``month``."""
    known = s.loc[:pd.Timestamp(month)].dropna()
    return float(known.iloc[-1]) if not known.empty else float("nan")
