"""Strategy A: walk-forward model selection and combination (meta pass over the base forecasts).

At origin t, for each (series, horizon), only base rows already REALISED at t are used:
target_month <= t - lag(series), with the publication lags of config/availability.yaml in realtime mode and the live
gap of each masked series in submission mode (0 for Schneider's CSV targets, known up to the origin by convention). Candidates are scored on the last ``window`` origin-months of those rows (pinball loss
averaged over P10/P50/P90, on the rows all eligible candidates share). Then:
- wf_best   : the candidate with the lowest trailing loss;
- wf_blend  : quantile-wise geometric average, weights proportional to loss ** -power (the submitted model);
- wf_median : quantile-wise median of the shortlist (pre-registered).
Quantiles are re-sorted. With no eligible candidate (fewer than ``min_rows`` realised rows), all three copy
the fallback model (naive).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics
from .vintage import month_add

Q = ["p10", "p50", "p90"]
META_MODELS = ("wf_best", "wf_blend", "wf_median")


@dataclass(frozen=True)
class MetaConfig:
    window: int = 36
    min_rows: int = 12
    power: float = 4.0  # chosen on DEV among {1, 2, 4} (pre-registered rule: lowest pooled dev QL)
    shortlist: tuple[str, ...] = ()
    fallback: str = "naive"


def realised(base: pd.DataFrame, t, mode: str, lags: dict[str, int] | None = None) -> pd.Series:
    t = pd.Timestamp(t)
    if mode not in ("submission", "realtime"):
        raise ValueError(mode)
    if mode == "submission":
        lags = {s: (lags or {}).get(s, 0) for s in base["series"].unique()}
    elif lags is None:
        from .availability import realtime_lags
        lags = realtime_lags()
    missing = sorted(set(base["series"]) - set(lags))
    if missing:
        raise KeyError(f"no realtime lag for series {missing}")
    cutoff = base["series"].map(lambda s: month_add(t, -int(lags[s])))
    return base["target_month"] <= cutoff


def _combine(rows: pd.DataFrame, weights: pd.Series | None) -> np.ndarray:
    lq = np.log(rows.set_index("model")[Q].astype(float))
    if weights is None:
        out = np.exp(lq.median(axis=0).to_numpy())
    else:
        w = weights.reindex(lq.index).fillna(0.0)
        out = np.exp((lq.mul(w, axis=0)).sum(axis=0).to_numpy() / w.sum())
    return np.sort(out)


def meta_pass(base: pd.DataFrame, cfg: MetaConfig, mode: str, lags: dict[str, int] | None = None,
              origins=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = metrics.score_rows(base)[["model", "series", "origin", "horizon", "target_month", "ql"]]
    origins = sorted(base["origin"].unique()) if origins is None else [pd.Timestamp(o) for o in origins]
    if mode == "realtime" and lags is None:
        from .availability import realtime_lags
        lags = realtime_lags()
    out, diag = [], []
    for (s, h), g in base.groupby(["series", "horizon"]):
        sc = scored[(scored.series == s) & (scored.horizon == h)]
        lag = int((lags or {}).get(s, 0)) if mode == "submission" else int(lags[s]) if s in lags else None
        if lag is None:
            raise KeyError(f"no realtime lag for series {[s]}")
        for t in origins:
            now = g[g.origin == t]
            if now.empty or cfg.fallback not in set(now.model):
                continue
            cutoff = month_add(t, -lag)
            r = sc[(sc.target_month <= cutoff) & (sc.origin > month_add(t, -cfg.window)) & (sc.origin <= t)]
            cand = set(now.model) - set(META_MODELS)
            counts = r[r.model.isin(cand)].groupby("model").size()
            elig = counts[counts >= cfg.min_rows].index.tolist()
            table = r[r.model.isin(elig)].pivot_table(index="origin", columns="model", values="ql").dropna()
            fb_row = now[now.model == cfg.fallback].iloc[0]
            use_fb = len(elig) == 0 or len(table) < cfg.min_rows
            if use_fb:
                q = {m: fb_row[Q].to_numpy(dtype=float) for m in META_MODELS}
                pick, weights = cfg.fallback, {}
            else:
                loss = table.mean()
                pick = loss.idxmin()
                w = loss ** -cfg.power
                w = w / w.sum()
                weights = w.round(6).to_dict()
                short = [m for m in (cfg.shortlist or sorted(cand)) if m in set(now.model)]
                q = {"wf_best": now[now.model == pick][Q].iloc[0].to_numpy(dtype=float),
                     "wf_blend": _combine(now[now.model.isin(w.index)], w),
                     "wf_median": _combine(now[now.model.isin(short)], None) if short
                     else fb_row[Q].to_numpy(dtype=float)}
            for m in META_MODELS:
                row = fb_row.copy()
                row["model"] = m
                row[Q] = np.sort(q[m])
                out.append(row)
            diag.append({"origin": t, "series": s, "horizon": h, "n_realised": len(table) if not use_fb else len(r),
                         "pick": pick, "weights": json.dumps(weights), "fallback_used": use_fb})
    return pd.DataFrame(out).reset_index(drop=True), pd.DataFrame(diag)


def pick_stability(diag: pd.DataFrame) -> pd.DataFrame:
    """Per (series, horizon): how often wf_best switches model, the most picked model and its share, and how often
    wf_best agrees with the largest wf_blend weight."""
    rows = []
    for (s, h), g in diag.sort_values("origin").groupby(["series", "horizon"]):
        g = g[~g.fallback_used]
        if g.empty:
            continue
        picks = g["pick"]
        top_w = g["weights"].map(lambda x: max(json.loads(x).items(), key=lambda kv: kv[1])[0])
        rows.append({"series": s, "horizon": h, "n": len(g), "switch_rate": float((picks != picks.shift()).iloc[1:].mean())
                     if len(g) > 1 else 0.0, "top_pick": picks.mode().iloc[0],
                     "top_pick_share": float((picks == picks.mode().iloc[0]).mean()),
                     "agree_best_blend": float((picks.to_numpy() == top_w.to_numpy()).mean())})
    return pd.DataFrame(rows)
