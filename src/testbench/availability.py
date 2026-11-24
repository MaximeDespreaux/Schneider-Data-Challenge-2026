"""Availability table (config/availability.yaml): the ONE place publication lags are defined.

- ``realtime_lags``   : realtime mode, a buyer deciding mid-month t sees reference month t - lag.
- ``ragged_edge_lags``: submission mode, each input series is masked by the gap it has TODAY between its last
  delivered month and the live origin, at every backtest origin (spec decision 6.4). Targets are exempt
  (Schneider's convention: known up to the origin).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from src.data import ROOT

from .vintage import months_between

AVAILABILITY = ROOT / "config" / "availability.yaml"
FIELDS = ["provider", "formation", "realtime_lag", "revision_months", "vintages", "licence"]


@lru_cache(maxsize=4)
def _load(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    df = pd.DataFrame.from_dict(raw, orient="index").rename_axis("series_id")
    missing = [c for c in FIELDS if c not in df]
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")
    if df[FIELDS].isna().any().any():
        raise ValueError(f"{path}: incomplete rows {df.index[df[FIELDS].isna().any(axis=1)].tolist()}")
    df["realtime_lag"] = df["realtime_lag"].astype(int)
    if (df["realtime_lag"] < 0).any():
        raise ValueError(f"{path}: negative realtime_lag")
    return df[FIELDS]


def load_availability(path: Path = AVAILABILITY) -> pd.DataFrame:
    return _load(str(path)).copy()


def check_coverage(columns, av: pd.DataFrame | None = None) -> None:
    av = load_availability() if av is None else av
    missing = [c for c in columns if c not in av.index]
    if missing:
        raise KeyError(f"series without a row in config/availability.yaml: {missing}")


def realtime_lags(av: pd.DataFrame | None = None) -> dict[str, int]:
    av = load_availability() if av is None else av
    return av["realtime_lag"].astype(int).to_dict()


def ragged_edge_lags(panel: pd.DataFrame, live_origin, exempt) -> dict[str, int]:
    """Gap (months) between each non-exempt column's last valid month and ``live_origin``; >= 0.
    Computed ONCE on the full delivered panel, then applied at every backtest origin."""
    live = pd.Timestamp(live_origin)
    out = {}
    for c in panel.columns:
        if c in exempt:
            continue
        last = panel[c].loc[:live].last_valid_index()
        if last is None:
            continue
        out[c] = max(0, months_between(last, live))
    return out
