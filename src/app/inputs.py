"""Local inputs of the scenario app, all as known at the forecast origin. Reads confidential local files only."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import data as d
from src.business.run_margin_replay import economics
from src.testbench.candidates import SUBMITTED
from src.testbench.models import NaiveVolScaled
from src.testbench.targets import TARGET_IDS, load_assumptions

ORIGIN = pd.Timestamp("2026-09-01")
HORIZONS = (3, 6, 9)
GRADED = d.OUT / "backtest" / "graded"
# energy drivers shown in the app (public series). Which power price applies to which plant is under NDA: it is read
# from config/assumptions.yaml (plant_power_series), never written here.
ENERGY = {"ED02": "Gas TTF (EUR/MWh)", "ED01": "Brent (EUR)", "E09": "Power Germany", "E10": "Power France",
          "E11": "Power Hungary"}


def plant_power(a: dict, product: str) -> str | None:
    return (a.get("plant_power_series") or {}).get(product)


FRESH = "fresh"  # no graded run: forecast at the origin computed here (cached)
FRESH_CACHE = d.OUT / "app" / "fresh_forecasts.parquet"


def graded_tags() -> list[str]:
    """Graded runs with submission forecasts, newest first."""
    runs = sorted(GRADED.glob("*/submission/forecasts.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.parents[1].name for p in runs]


def sources() -> list[str]:
    return graded_tags() + [FRESH]


def _fresh_forecasts(panel: pd.DataFrame, der) -> pd.DataFrame:
    """The submitted model and the naive model at the origin only, exactly as one origin of the graded run in
    submission mode (same models, targets and live ragged edge). Recomputed when a file in data/raw is newer."""
    newest_input = max((p.stat().st_mtime for p in d.RAW.rglob("*") if p.is_file()), default=0.0)
    if FRESH_CACHE.exists() and FRESH_CACHE.stat().st_mtime > newest_input:
        return pd.read_parquet(FRESH_CACHE)
    from src.testbench.backtest import run_backtest
    from src.testbench.candidates import candidate_models
    from src.testbench.pipeline import lags_for, targets

    tg = targets(panel)
    models = [m for m in candidate_models() if m.name in (SUBMITTED, "naive")]
    fc = run_backtest(panel, tg, models, origins=[ORIGIN], mode="submission", derive=der,
                      lags=lags_for("submission", panel, tg, str(ORIGIN.date())), verbose=False)
    FRESH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(FRESH_CACHE)
    return fc


@dataclass
class Inputs:
    tag: str
    panel: pd.DataFrame
    weights: pd.DataFrame    # product, material, series, value (EUR per box at origin), weight
    fc: pd.DataFrame         # series, horizon, p10, p50, p90, last (value at the last observed month), source
    hist_p90: dict           # (product, h) -> historical P90 BOM ratio (naive model)
    picks: pd.DataFrame      # dev-rule policy per product, material share, horizon
    cu_term: float           # LME copper 3M/cash log spread at the origin (0 if unknown)
    assumptions: dict


def _naive_vol(y: pd.Series) -> pd.DataFrame:
    f = NaiveVolScaled(halflife=3).forecast(y, list(HORIZONS))
    return f.rename_axis("horizon").reset_index().assign(last=float(y.iloc[-1]), source="naive_vol (no model forecast)")


def load(tag: str) -> Inputs:
    """``tag``: a graded run, or FRESH to compute the forecast at the origin here."""
    from src.external.signals import build_panel
    from src.testbench.panel import load_backtest_config
    from src.testbench.run import bom_deriver

    panel = build_panel(use_licensed=load_backtest_config()["use_licensed"])
    known = panel.loc[:ORIGIN]
    der = bom_deriver(panel)
    weights = der.weights_at(known, ORIGIN)

    raw = _fresh_forecasts(panel, der) if tag == FRESH else pd.read_parquet(GRADED / tag / "submission" / "forecasts.parquet")
    raw = raw[raw.origin == ORIGIN]
    sub = raw[raw.model == SUBMITTED].copy()
    sub["last"] = [float(panel.at[r.last_obs, r.series]) if r.series in panel else 100.0 for r in sub.itertuples()]
    sub.loc[sub.series.isin(TARGET_IDS.values()), "last"] = 100.0  # BOM indices: origin = 100
    sub["source"] = f"{SUBMITTED} ({'computed here' if tag == FRESH else f'graded run {tag}'})"
    rows = [sub[["series", "horizon", "p10", "p50", "p90", "last", "source"]]]
    need = (set(weights.series) | set(ENERGY)) - set(sub.series)
    for s in sorted(need):
        y = known[s].dropna()
        if len(y) >= 36:
            rows.append(_naive_vol(y).assign(series=s))
    fc = pd.concat(rows, ignore_index=True)

    naive = raw[(raw.model == "naive") & raw.series.isin(TARGET_IDS.values())]
    prod = {v: k for k, v in TARGET_IDS.items()}
    hist_p90 = {(prod[r.series], int(r.horizon)): r.p90 / 100 for r in naive.itertuples()}
    picks_path = d.OUT / "margin" / "policy_picks.csv"
    picks = pd.read_csv(picks_path) if picks_path.exists() else pd.DataFrame()
    term = known["X_CU_TERM"].dropna() if "X_CU_TERM" in known else pd.Series(dtype=float)
    return Inputs(tag, panel, weights, fc, hist_p90, picks, float(term.iloc[-1]) if len(term) else 0.0,
                  load_assumptions())


def default_economics(inp: Inputs, product: str):
    return economics(inp.assumptions, product)


def recommended_policy(inp: Inputs, product: str, share: float, h: int) -> tuple[str, float] | None:
    """Decision-map pick (dev-window rule) for the nearest material share in the replay grid."""
    p = inp.picks
    if p.empty:
        return None
    p = p[(p["product"] == product) & (p["horizon"] == h) & (p["model"] == SUBMITTED)]
    if p.empty:
        return None
    r = p.iloc[(p["material_share"] - share).abs().argsort().iloc[0]]
    return str(r["rule_pick_dev"]), float(r["material_share"])
