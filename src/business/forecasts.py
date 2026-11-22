"""Task 1 — forecasts at the origin (2026-09), computed once and cached.

    uv run python -m src.business.forecasts

Source: the champion forecasts if data/outputs/external/champion_holdout.parquet has them for every series, else ONE
single-origin run of NaiveVolScaled (cheap). Output: data/outputs/business/forecasts_origin_202609.parquet.
"""

from __future__ import annotations

import time
import warnings

import pandas as pd

from src import data as d

from .config import BOM_IDS, load_business_config, origin_ts
from . import paths

SERIES = [BOM_IDS["RM6"], BOM_IDS["AirSeT"], "D01", "D02", "L01", "L02", "L03", "EP01", "EL01", "S11", "S05"]
COLUMNS = ["series", "horizon", "target_month", "p10", "p50", "p90", "last_value", "source"]


def cache_path(cfg: dict):
    return paths.OUT / f"forecasts_origin_{cfg['origin_month'].replace('-', '')}.parquet"


GRADED = d.OUT / "backtest" / "graded"


def champion_path():
    return d.OUT / "external" / "champion_holdout.parquet"


def source_path(prefer: str):
    """File holding the preferred model's forecasts: the newest graded run (submission mode) for any model of the
    hardened backtest (``champion_wf``, ``wf_blend``...); the old champion holdout for ``champion``."""
    if prefer != "champion":
        runs = sorted(GRADED.glob("*/submission/forecasts.parquet"), key=lambda p: p.stat().st_mtime)
        return runs[-1] if runs else GRADED / "missing" / "forecasts.parquet"
    return champion_path()


def champion_rows(champ, origin: pd.Timestamp, series, model: str = "champion") -> pd.DataFrame | None:
    """``model`` forecasts at the origin if the file has them for EVERY series, else None."""
    if not champ.exists():
        return None
    c = pd.read_parquet(champ)
    c = c[(c["model"] == model) & (pd.to_datetime(c["origin"]) == origin)]
    return c[c["series"].isin(series)] if set(series) <= set(c["series"]) else None


def cache_decision(cache, champ, origin: pd.Timestamp, prefer: str) -> tuple[bool, str]:
    """(use_cache, reason). The cache is invalidated if the champion file is newer than it, or if it holds
    naive_vol forecasts while usable champion forecasts now exist."""
    if not cache.exists():
        return False, "no cache"
    if champ.exists():
        if champ.stat().st_mtime > cache.stat().st_mtime:
            return False, f"{champ.name} is newer than the cache"
        cached = pd.read_parquet(cache, columns=["series", "source"])
        if cached["source"].iloc[0] != prefer and champion_rows(champ, origin, set(cached["series"]),
                                                                model=prefer) is not None:
            return False, f"cache holds {cached['source'].iloc[0]} but {prefer} forecasts are now available"
    return True, "cache up to date"


def get_origin_forecasts(cfg: dict | None = None) -> pd.DataFrame:
    cfg = load_business_config() if cfg is None else cfg
    path = cache_path(cfg)
    origin = origin_ts(cfg)
    prefer = cfg["forecast_source"]["prefer"]
    use, reason = cache_decision(path, source_path(prefer), origin, prefer)
    if use:
        df = pd.read_parquet(path)
        df.attrs = {"runtime_s": 0.0, "cached": True}
        print(f"    forecast source: {df['source'].iloc[0]} (from cache: {reason})")
        return df
    print(f"    forecast cache not used: {reason}")

    from src.testbench.backtest import run_backtest
    from src.testbench.models import NaiveVolScaled
    from src.testbench.run import bom_deriver, build_panel

    t0 = time.perf_counter()
    panel = build_panel()
    series = []
    for s in SERIES:
        if s in BOM_IDS.values() or (s in panel and panel[s].notna().any()):
            series.append(s)
        else:
            warnings.warn(f"{s} absent from the panel: skipped", stacklevel=2)

    fc, source = None, None
    champ = source_path(prefer)
    fc = champion_rows(champ, origin, series, model=prefer)
    if fc is not None:
        source, why = prefer, f"{champ} has {prefer} forecasts for every series at {origin:%Y-%m}"
    elif champ.exists():
        why = f"{champ.name} lacks {prefer} forecasts for some series at {origin:%Y-%m}"
    else:
        why = f"{champ} not found (run the graded backtest, or src.external.run_champion for 'champion')"
    if fc is None:
        if cfg["forecast_source"]["fallback_model"] != "naive_vol":
            raise ValueError(f"unknown fallback_model: {cfg['forecast_source']['fallback_model']}")
        fc = run_backtest(panel, series, [NaiveVolScaled()], origins=[origin], derive=bom_deriver(panel),
                          verbose=False)
        source = "naive_vol"
    print(f"    forecast source: {source} ({why})")

    out = fc[["series", "horizon", "target_month", "p10", "p50", "p90"]].copy()
    last = {s: (100.0 if s in BOM_IDS.values() else float(panel[s].loc[:origin].dropna().iloc[-1])) for s in series}
    out["last_value"] = out["series"].map(last)
    out["source"] = source
    out = out[COLUMNS].sort_values(["series", "horizon"]).reset_index(drop=True)
    out.attrs = {}  # drop the backtest's per-model runtimes before caching
    paths.out_dir()
    out.to_parquet(path)
    out.attrs = {"runtime_s": time.perf_counter() - t0, "cached": False}
    return out


def pct_change(df: pd.DataFrame, series: str, horizon: int, q: str) -> float:
    """p_q / last_value − 1 for one series and horizon (q = "p10" | "p50" | "p90")."""
    row = df[(df["series"] == series) & (df["horizon"] == horizon)]
    if row.empty:
        raise KeyError(f"no forecast for {series} at h={horizon}")
    return float(row[q].iloc[0] / row["last_value"].iloc[0] - 1)


def run(cfg: dict | None = None) -> dict:
    cfg = load_business_config() if cfg is None else cfg
    df = get_origin_forecasts(cfg)
    head = {"source": df["source"].iloc[0], "n_series": int(df["series"].nunique()),
            "runtime_s": round(float(df.attrs.get("runtime_s", 0.0)), 2), "from_cache": bool(df.attrs.get("cached"))}
    for prod, sid in BOM_IDS.items():
        for q in ["p50", "p90"]:
            head[f"{prod}_index_{q}_change_h9_pct"] = round(100 * pct_change(df, sid, 9, q), 2)
    return head


if __name__ == "__main__":
    print(run())
