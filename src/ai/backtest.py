"""Rolling-origin backtest of the AI models, in the exact row format of ``src/backtest.py``.

Rows: series, origin, h, model, pred, q10, q50, q90 (all as log-change vs the origin level), then
``attach_truth`` adds target_month, y_origin, y_true, actual. So the AI rows can be concatenated with
``data/processed/backtest.parquet`` and scored with ``src.backtest.score`` / ``dm_test``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import backtest as bt
from src import features as fe

from . import config as C
from .foundation import build_input, predict_batch


def _pairs(targets: pd.DataFrame, start: str, live: bool, series=None) -> list[tuple[str, pd.Timestamp]]:
    """(series, origin) pairs: backtest origins (any horizon scorable) or, if live, the latest observation."""
    pairs = []
    for s in (series or targets.columns):
        if live:
            pairs.append((s, targets[s].last_valid_index()))
            continue
        origins = set().union(*[bt._origins(targets[s], h, start) for h in C.HORIZONS])
        pairs += [(s, t) for t in sorted(origins)]
    return pairs


def run_foundation(targets: pd.DataFrame, exog: pd.DataFrame, keys=tuple(C.MODELS),
                   start: str = C.BACKTEST_START, live: bool = False,
                   series: list[str] | None = None, horizons=C.HORIZONS) -> tuple[pd.DataFrame, dict[str, float]]:
    """Forecast every (series, origin) with each pretrained model. Returns (rows, runtime seconds per model).
    ``targets`` must be the full target panel (drivers such as the steel gap use it); ``series`` restricts
    which targets are forecast."""
    pairs = _pairs(targets, start, live, series)
    X = fe._driver_frame(exog, targets)
    rows, runtime = [], {}
    for key in keys:
        cov = C.MODELS[key]["covariates"]
        inputs = [build_input(targets, exog, s, t, covariates=cov, X=X) for s, t in pairs]
        arr, runtime[key] = predict_batch(key, inputs, horizon=max(horizons))
        name = f"{key}+cov" if cov else key
        for (s, t), d, q in zip(pairs, inputs, arr):
            y0 = d["target"][-1]
            for h in horizons:
                lq = np.log(np.maximum(q[h - 1], 1e-9) / y0)
                rows.append(dict(series=s, origin=t, h=h, model=name, pred=lq[1], q10=lq[0], q50=lq[1], q90=lq[2]))
    res = pd.DataFrame(rows)
    if live:
        res["target_month"] = res.origin + res.h.apply(lambda h: pd.DateOffset(months=int(h)))
        res["y_origin"] = [targets.at[t, s] for s, t in zip(res.series, res.origin)]
        return res, runtime
    return bt.attach_truth(res, targets), runtime


# --------------------------------------------------------------------------------------------- calibration
def add_cqr(res: pd.DataFrame, model: str, name: str | None = None, alpha: float = 0.2,
            min_resid: int = 12) -> pd.DataFrame:
    """Online conformalised quantile regression (Romano et al. 2019) on a model's own P10/P90.

    At each origin, only forecasts already realised and published (``features.published_by``) are used to measure how far the
    P10–P90 band missed; the band is widened (or narrowed) by that conformity quantile.
    """
    r = res[res.model == model].copy().sort_values("origin")
    e = np.maximum(r.q10 - r.actual, r.actual - r.q90)
    adj = []
    for _, row in r.iterrows():
        past = (r.h == row.h) & fe.published_by(r.target_month, r.series, row.origin, row.series)
        own = e[past & (r.series == row.series)]
        pool = own if len(own) >= min_resid else e[past]
        n = len(pool)
        adj.append(np.quantile(pool, min(1.0, (1 - alpha) * (n + 1) / n)) if n >= min_resid else np.nan)
    r["q10"], r["q90"] = r.q10 - adj, r.q90 + adj
    r["model"] = name or f"{model}+cqr"
    return pd.concat([res, r], ignore_index=True)


def add_mean(res: pd.DataFrame, members: list[str], name: str) -> pd.DataFrame:
    """Equal-weight average of point forecasts (log-change), intervals added later by conformal."""
    out = bt.add_ensemble(res, members=tuple(members), name=name)
    out.loc[out.model == name, ["q10", "q90"]] = np.nan
    return bt.add_conformal(out, name)


# --------------------------------------------------------------------------------------------- comparison
def compare(res: pd.DataFrame, models: list[str], window: pd.Series | None = None, bench: str = "naive") -> pd.DataFrame:
    """Per model, on the rows it shares with ``bench`` (and where both have P10/P90):
    relMAE and relQL (pinball of P10 & P90) vs the benchmark, coverage of P10–P90, mean width, n."""
    r = res if window is None else res[window]
    key = ["series", "origin", "h"]
    b = r[r.model == bench].set_index(key)
    rows = []
    for m in models:
        a = r[r.model == m].set_index(key)
        j = a.join(b[["pred", "q10", "q90"]], rsuffix="_b", how="inner").dropna(subset=["q10", "q90", "q10_b", "q90_b"])
        if j.empty:
            continue
        pin = lambda lo, hi: (bt._pinball(j.actual, lo, 0.1) + bt._pinball(j.actual, hi, 0.9)) / 2  # noqa: E731
        rows.append(dict(model=m, n=len(j),
                         relMAE=(j.actual - j.pred).abs().sum() / (j.actual - j.pred_b).abs().sum(),
                         relQL=pin(j.q10, j.q90) / pin(j.q10_b, j.q90_b),
                         coverage=((j.actual >= j.q10) & (j.actual <= j.q90)).mean(),
                         width_pct=100 * (np.exp(j.q90) - np.exp(j.q10)).mean()))
    return pd.DataFrame(rows).set_index("model")


def select_preregistered(res: pd.DataFrame, candidates: list[str], split: str = C.SELECTION_SPLIT,
                         name: str = "ai_selected") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pre-registered rule: per (series, h) pick the candidate with the lowest P10/P90 pinball loss on targets
    BEFORE ``split``; the chosen model's rows with ORIGIN from ``split`` on become ``name`` (the only rows reported).

    Embargo: a forecast made in 2023 for a 2024 month is in neither set. The choice uses outcomes up to December
    2023, which were not known at that origin."""
    cut = pd.Timestamp(split)
    pre = res[(res.model.isin(candidates)) & (res.target_month < cut)].dropna(subset=["q10", "q90"])
    pin = pre.assign(pl=lambda d: (np.maximum(0.1 * (d.actual - d.q10), -0.9 * (d.actual - d.q10))
                                   + np.maximum(0.9 * (d.actual - d.q90), -0.1 * (d.actual - d.q90))) / 2)
    choice = pin.groupby(["series", "h", "model"]).pl.mean().groupby(["series", "h"]).idxmin().map(lambda t: t[2])
    choice = choice.rename("chosen").reset_index()
    post = res[res.origin >= cut].merge(choice, on=["series", "h"])
    sel = post[post.model == post.chosen].drop(columns="chosen").assign(model=name)
    return pd.concat([res, sel], ignore_index=True), choice


# --------------------------------------------------------------------------------------------- combinations
def _aligned(res: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    key = ["series", "origin", "h"]
    return res[res.model == a].merge(res[res.model == b][key + ["pred", "q10", "q50", "q90"]], on=key,
                                     suffixes=("", "_b"))


def add_hybrid(res: pd.DataFrame, point: str, band: str, name: str) -> pd.DataFrame:
    """P50 from ``point``; band shape (P10/P90 offsets around its own P50) from ``band``.
    Same idea as the team's champion hybrid (centre from one model, ``naive_vol`` fan)."""
    j = _aligned(res, point, band)
    j["q10"], j["q90"] = j.pred + (j.q10_b - j.pred_b), j.pred + (j.q90_b - j.pred_b)
    j["q50"] = j.pred
    return pd.concat([res, j[res.columns.intersection(j.columns)].assign(model=name)], ignore_index=True)


def add_qavg(res: pd.DataFrame, a: str, b: str, name: str) -> pd.DataFrame:
    """Equal-weight quantile averaging (Vincentisation) of two probabilistic models."""
    j = _aligned(res, a, b)
    for c in ("pred", "q10", "q50", "q90"):
        j[c] = (j[c] + j[f"{c}_b"]) / 2
    return pd.concat([res, j[res.columns.intersection(j.columns)].assign(model=name)], ignore_index=True)
