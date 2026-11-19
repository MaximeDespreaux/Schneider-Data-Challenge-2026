"""The ONE panel builder used by the backtest, the canaries and the submission, plus the backtest config.

Panel = challenge raw-material + energy panel (all-NaN columns such as the empty official L01-L08 dropped),
extended back to 1992 from identical public sources, + public external signals (X_*), + epoxy cost inputs (EPX_*),
+ licensed CRU/MEPS steel prices (LIC_*) only when ``use_licensed``. Every column must have a row in
config/availability.yaml.
"""

from __future__ import annotations

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


def build_full_panel(use_licensed: bool, long: bool = True, external: bool = True) -> pd.DataFrame:
    from src.external.signals import extend_history, load_external

    p = d.load_rm_wide().join(d.load_energy_wide(), how="left")
    p = p.dropna(axis=1, how="all")
    if long:
        p = extend_history(p)
    if external:
        p = p.join(load_external(), how="left")
    p = p.join(d.load_epoxy(), how="left")
    if use_licensed:
        from src.licensed import load_licensed
        p = p.join(load_licensed(), how="left")
    p = p.dropna(axis=1, how="all")
    return p.sort_index()
