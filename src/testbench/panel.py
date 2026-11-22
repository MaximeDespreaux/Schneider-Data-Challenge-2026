"""The ONE panel builder used by the backtest, the canaries and the submission, plus the backtest config.

Panel = challenge raw-material + energy panel (all-NaN columns such as the empty official L01-L08 dropped),
+ added series (BR01 brass proxy, EP01 epoxy resin, EL01-03 Eurostat electronics, and the licensed MEPS/CRU steel
L01-L03 / L01_FR only when ``use_licensed``), extended back to 1992 from identical sources, + public external signals
(X_*), + epoxy cost inputs (EPX_*). Every column must have a row in config/availability.yaml.
"""

from __future__ import annotations

import warnings

import pandas as pd

from src import data as d

REQUIRED = ("use_licensed", "live_origin", "team_id", "availability_file")


def load_backtest_config(assumptions: dict | None = None) -> dict:
    if assumptions is None:
        from .targets import load_assumptions
        assumptions = load_assumptions()
    if "backtest" not in assumptions:
        raise KeyError("config/assumptions.yaml has no 'backtest' section (added 2026-11-19): copy it from "
                       "config/assumptions.example.yaml")
    cfg = dict(assumptions["backtest"])
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise KeyError(f"'backtest' section misses {missing}: see config/assumptions.example.yaml")
    cfg["use_licensed"] = bool(cfg["use_licensed"])
    return cfg


def base_panel(use_licensed: bool = True) -> pd.DataFrame:
    """Challenge panel (rm + energy, all-NaN columns dropped) + BR01, L01-L03/L01_FR, EP01, EL01-03."""
    p = d.load_rm_wide().join(d.load_energy_wide(), how="left")
    p = p.dropna(axis=1, how="all")
    # brass proxy (EUR/t): ~63% copper + 37% zinc (S05 USD/t, S06 USD per EUR). Not a CSV target.
    p["BR01"] = 0.63 * p["D01"] + 0.37 * p["S05"] / p["S06"]
    if use_licensed:
        try:
            steel = d.load_licensed_steel()
            for c in steel.columns:
                p[c] = steel[c].reindex(p.index)
        except FileNotFoundError as e:
            warnings.warn(f"{e} -> steel falls back on the Eurostat indices (S09/S10).", stacklevel=2)
    try:
        p["EP01"] = d.load_epoxy_series().reindex(p.index)
    except FileNotFoundError as e:
        warnings.warn(f"{e} -> no epoxy series EP01.", stacklevel=2)
    ppi = d.RAW / "external" / "eurostat_ppi_long.csv"
    eu = pd.read_csv(ppi, index_col=0, parse_dates=True) if ppi.exists() else pd.DataFrame()
    for c in ["EL01", "EL02", "EL03"]:
        if c in eu:
            p[c] = eu[c].reindex(p.index)
    if "EL01" not in eu:
        warnings.warn("EL01 missing: run `uv run python -m src.external.fetch_public`", stacklevel=2)
    return p


def build_full_panel(use_licensed: bool, long: bool = True, external: bool = True) -> pd.DataFrame:
    from src.external.signals import extend_history, load_external

    p = base_panel(use_licensed=use_licensed)
    if long:
        p = extend_history(p)
    if external:
        p = p.join(load_external(), how="left")
    p = p.join(d.load_epoxy(), how="left")
    p = p.dropna(axis=1, how="all")
    return p.sort_index()
