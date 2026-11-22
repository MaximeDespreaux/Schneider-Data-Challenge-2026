"""Leakage canaries (spec 5.4). A graded run refuses to start unless all of them pass on the current code hash.

- (a) planted leak   : a model that reads y(t+3) must show absurd skill, and the perturbation audit must flag it;
- (b) permuted target: on targets whose monthly changes are shuffled, no model may beat the naive;
- (c) truncation identity: forecasts from a panel physically cut at the origin equal, bit for bit, those from the
      full panel passed through ``information_set``, for every real model (and the derived indices);
- (d) planted off-by-one: a feature shifted by one month must FAIL (c), which proves (c) can fire.

``prepare(raw) -> panel`` is everything that turns raw columns into the backtest panel (features, derived columns).
It runs on the cut panel too, so it is inside what (c) checks.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .audit import perturb_future
from .backtest import run_backtest
from .metrics import score_rows
from .models import Naive, _frame

KEYS = ["model", "series", "origin", "horizon", "target_month", "last_obs", "step"]
Q = ["p10", "p50", "p90"]
ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------- planted models
def planted_leak_prepare(targets, shift: int = 3) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Adds ``LEAK_<s> = s.shift(-shift)``: the row at t holds the value of t+shift."""
    def prep(df):
        return df.assign(**{f"LEAK_{s}": df[s].shift(-shift) for s in targets if s in df})
    return prep


class PlantedLeak:
    """P50 = the leaked future value at the origin row (when finite), else the last y. Bands are +-1%."""

    name = "planted_leak"

    def forecast(self, y, steps, exog=None):
        col = f"LEAK_{y.name}"
        v = float(exog[col].iloc[-1]) if exog is not None and col in exog else np.nan
        p = v if np.isfinite(v) and v > 0 else float(y.iloc[-1])
        return _frame(steps, [0.99 * p] * len(steps), [p] * len(steps), [1.01 * p] * len(steps))


def off_by_one_prepare(col: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Adds ``<col>_NEXT = col.shift(-1)``: a feature deliberately one month in the future."""
    def prep(df):
        return df.assign(**{f"{col}_NEXT": df[col].shift(-1)})
    return prep


class OffByOneModel:
    """P50 = last * next-month change of ``col`` read from the shifted feature; falls back to last."""

    name = "off_by_one"

    def __init__(self, col: str):
        self.col = col

    def forecast(self, y, steps, exog=None):
        last = float(y.iloc[-1])
        r = float(exog[f"{self.col}_NEXT"].iloc[-1] / exog[self.col].iloc[-1])
        p = last * np.exp(np.log(r)) if np.isfinite(r) and r > 0 else last
        return _frame(steps, [p] * len(steps), [p] * len(steps), [p] * len(steps))


def permute_targets(panel: pd.DataFrame, cols, seed: int = 0) -> pd.DataFrame:
    """Each column of ``cols``: its monthly log changes are demeaned, permuted i.i.d. and re-cumulated from its
    first value.

    Block length 1 on purpose: with blocks of length L > 1 the autocorrelation inside each block survives, so a
    momentum model keeps part of its genuine skill and the canary could not tell that skill from a leak. With
    single months no model can legitimately beat the naive; any skill left must come from reading the future.
    Demeaned because a permutation spreads the full-sample mean change evenly over time: every origin would then
    "know" the whole sample's drift (future information), and drift models win on it without any code leak."""
    rng = np.random.default_rng(seed)
    out = panel.copy()
    for c in cols:
        s = panel[c].dropna()
        d = np.diff(np.log(s.to_numpy(dtype=float)))
        d = rng.permutation(d - d.mean())
        out.loc[s.index, c] = np.exp(np.log(s.iloc[0]) + np.concatenate([[0.0], np.cumsum(d)]))
    return out


# ----------------------------------------------------------------------------- identity checks
def _compare(a: pd.DataFrame, b: pd.DataFrame, models, origins) -> pd.DataFrame:
    """Per (origin, model): same row set and exact equality of p10/p50/p90."""
    rows = []
    for o in pd.DatetimeIndex(origins):
        for m in models:
            sa = a[(a.origin == o) & (a.model == m.name)][KEYS + Q]
            sb = b[(b.origin == o) & (b.model == m.name)][KEYS + Q]
            j = sa.merge(sb, on=KEYS, how="outer", suffixes=("", "_b"), indicator=True)
            same = bool((j["_merge"] == "both").all())
            diff = float(max((j[c] - j[c + "_b"]).abs().max() for c in Q)) if len(j) else 0.0
            diff = diff if same else np.inf
            rows.append({"origin": o, "model": m.name, "n": len(j), "max_abs_diff": diff,
                         "PASS": bool(same and diff == 0)})
    return pd.DataFrame(rows)


def truncation_identity(raw, prepare, targets, models, origins, mode, lags, derive=None) -> pd.DataFrame:
    """``run_backtest(prepare(raw))`` vs ``run_backtest(prepare(raw.loc[:o]))`` at each origin o."""
    kw = dict(mode=mode, lags=lags, derive=derive, verbose=False)
    full = run_backtest(prepare(raw), targets, models, origins=origins, **kw)
    cut = pd.concat([run_backtest(prepare(raw.loc[:o]), targets, models, origins=[o], **kw) for o in origins])
    return _compare(full, cut, models, origins)


def perturbation_audit(raw, prepare, targets, models, origins, mode, lags, derive=None) -> pd.DataFrame:
    """Every raw column (``LIC_``, ``X_`` included) is replaced by noise after o, then ``prepare`` runs:
    any changed forecast at o means something read the future."""
    kw = dict(mode=mode, lags=lags, derive=derive, verbose=False)
    full = run_backtest(prepare(raw), targets, models, origins=origins, **kw)
    pert = pd.concat([run_backtest(prepare(perturb_future(raw, o)), targets, models, origins=[o], **kw)
                      for o in origins])
    return _compare(full, pert, models, origins)


def pooled_rel_ql(fc: pd.DataFrame, horizon: int | None = None, ref: str = "naive") -> pd.Series:
    """Mean QL / naive's mean QL on the same (series, origin, horizon) rows, pooled over series (and horizons)."""
    sc = score_rows(fc if horizon is None else fc[fc.horizon == horizon])
    k = ["series", "origin", "horizon"]
    sc = sc.merge(sc[sc.model == ref][k + ["ql"]], on=k, suffixes=("", "_ref"))
    g = sc.groupby("model")
    return g["ql"].mean() / g["ql_ref"].mean()


# ----------------------------------------------------------------------------- report
def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def git_dirty() -> bool:
    return bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
                               check=True).stdout.strip())


def _fails(res: pd.DataFrame) -> list:
    return [f"{r.model}@{r.origin:%Y-%m}" for r in res[~res.PASS].itertuples()]


def _mode_checks(raw, prepare, targets, models, derive, mode, lags, origins, skill_origins=None) -> dict:
    panel = prepare(raw)
    tg = [s for s in targets if s in panel]  # (a), (b), (d) run on panel columns (no derived index)
    naive = [] if any(m.name == "naive" for m in models) else [Naive()]
    kw = dict(mode=mode, lags=lags, verbose=False)
    so = origins if skill_origins is None else skill_origins  # (a)/(b) score skill: need many origins

    leak = planted_leak_prepare(tg)
    fc = run_backtest(leak(panel), tg, [Naive(), PlantedLeak()], origins=so, **kw)
    rel = float(pooled_rel_ql(fc, horizon=3)["planted_leak"])
    aud = perturbation_audit(raw, lambda r: leak(prepare(r)), tg, [PlantedLeak()], origins, mode, lags)
    flagged = bool((aud["max_abs_diff"] > 0).any())
    a = {"rel_QL_h3": rel, "flagged": flagged, "PASS": bool(rel < 0.2 and flagged)}

    fc = run_backtest(permute_targets(panel, tg), tg, naive + list(models), origins=so, **kw)
    rels = {k: float(v) for k, v in pooled_rel_ql(fc).items() if k != "naive"}
    b = {"rel_QL": rels, "PASS": bool(all(v >= 0.97 for v in rels.values()))}

    res = truncation_identity(raw, prepare, targets, models, origins, mode, lags, derive)
    c = {"n": int(res["n"].sum()), "failures": _fails(res), "PASS": bool(res["PASS"].all())}

    # the shifted column gets lag 0: a realtime mask on it would hide the planted month and (d) could not fire
    col = tg[0]
    res = truncation_identity(raw, lambda r: off_by_one_prepare(col)(prepare(r)), [col], [OffByOneModel(col)],
                              origins, mode, {**(lags or {}), col: 0, f"{col}_NEXT": 0})
    d = {"col": col, "caught": len(_fails(res)), "PASS": bool(not res["PASS"].all())}
    return {"a_planted_leak": a, "b_permuted": b, "c_truncation": c, "d_off_by_one": d}


def run_canaries(raw, prepare, targets, models, derive, lags_by_mode: dict, origins, out_dir,
                 extra_checks: dict[str, Callable[[], bool]] | None = None, skill_origins=None) -> dict:
    """Runs (a) to (d) in each mode of ``lags_by_mode`` ({mode: lags or None}) plus ``extra_checks`` (e.g. the
    meta-pass identity), and writes ``out_dir/<git_head>.json``. PASS only if every check passes."""
    results = {mode: _mode_checks(raw, prepare, targets, models, derive, mode, lags, origins, skill_origins)
               for mode, lags in lags_by_mode.items()}
    results["extra"] = {k: bool(f()) for k, f in (extra_checks or {}).items()}
    ok = all(r["PASS"] for mode in lags_by_mode for r in results[mode].values()) and all(results["extra"].values())
    report = {"head": git_head(), "dirty": git_dirty(), "results": results, "PASS": bool(ok)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{report['head']}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
