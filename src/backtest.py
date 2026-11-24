"""Rolling-origin (expanding window) backtest, metrics, conformal intervals and DM test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from . import features as fe
from . import models as mo

MAX_LAG = 2  # largest publication lag among the targets (months), used as a cheap pre-filter
SHOCK = (pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-01"))


def _origins(y: pd.Series, h: int, start: str, require_truth: bool = True) -> list[pd.Timestamp]:
    s = y.dropna()
    idx = s.index[(s.index >= start)]
    if require_truth:
        idx = [t for t in idx if (t + pd.DateOffset(months=h)) in s.index and s.index.get_loc(t) >= 12]
    return list(idx)


def _univariate_one_series(name: str, y: pd.Series, horizons: list[int], start: str) -> list[dict]:
    rows = []
    s = y.dropna()
    origins = sorted(set().union(*[_origins(y, h, start) for h in horizons]))
    for t in origins:
        preds = mo.univariate_forecasts(np.log(s.loc[:t]), horizons)
        for h in horizons:
            th = t + pd.DateOffset(months=h)
            if th not in s.index:
                continue
            for m, p in preds.items():
                rows.append(dict(series=name, origin=t, h=h, model=m, pred=p[h]))
    return rows


def rolling_origin(targets: pd.DataFrame, exog: pd.DataFrame, horizons=(3, 6, 9), start: str = "2020-01-01",
                   models=("univariate", "dynreg", "lgbm"), n_jobs: int = -1) -> pd.DataFrame:
    """Run every model at every monthly origin from ``start``.

    Returns a long frame: series, origin, h, model, pred (log-change), [q10, q50, q90], y_origin, y_true, actual.
    """
    horizons = list(horizons)
    rows: list[dict] = []
    if "univariate" in models:
        out = Parallel(n_jobs=n_jobs)(delayed(_univariate_one_series)(c, targets[c], horizons, start)
                                      for c in targets.columns)
        rows += [r for part in out for r in part]
    feats = {h: fe.make_features(targets, exog, h) for h in horizons}
    if "dynreg" in models:
        def _dyn(c):
            r = []
            for h in horizons:
                for t in _origins(targets[c], h, start):
                    r.append(dict(series=c, origin=t, h=h, model="dynreg",
                                  pred=mo.dynreg_predict(feats[h], c, fe.DRIVERS[c], t, h)))
            return r
        out = Parallel(n_jobs=n_jobs)(delayed(_dyn)(c) for c in targets.columns)
        rows += [r for part in out for r in part]
    if "lgbm" in models:
        def _lgb(h, t):
            p = mo.lgbm_predict(feats[h], t, h)
            return [dict(series=s, origin=t, h=h, model="lgbm", pred=r["mean"], q10=r["q10"], q50=r["q50"], q90=r["q90"])
                    for s, r in p.iterrows()]
        all_origins = {h: sorted(set().union(*[_origins(targets[c], h, start) for c in targets])) for h in horizons}
        out = Parallel(n_jobs=n_jobs)(delayed(_lgb)(h, t) for h in horizons for t in all_origins[h])
        rows += [r for part in out for r in part]
    res = pd.DataFrame(rows)
    return attach_truth(res, targets)


def attach_truth(res: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    lv = targets.stack().rename("lvl")
    lv.index.names = ["month", "series"]
    res = res.copy()
    res["target_month"] = res.origin + res.h.apply(lambda h: pd.DateOffset(months=int(h)))
    res["y_origin"] = lv.reindex(list(zip(res.origin, res.series))).values
    res["y_true"] = lv.reindex(list(zip(res.target_month, res.series))).values
    res["actual"] = np.log(res.y_true / res.y_origin)
    res = res[res.y_true.notna()]  # keep only scorable rows
    return res


def add_ensemble(res: pd.DataFrame, members=("ets", "arima", "theta", "dynreg", "lgbm"), name="ensemble") -> pd.DataFrame:
    """Equal-weight combination of point forecasts (log-change space)."""
    key = ["series", "origin", "h"]
    e = res[res.model.isin(members)].groupby(key).agg(pred=("pred", "mean"), n=("pred", "size")).reset_index()
    e = e[e.n == len(members)].drop(columns="n")
    first = res.drop_duplicates(key)[key + ["target_month", "y_origin", "y_true", "actual"]]
    e = e.merge(first, on=key)
    e["model"] = name
    return pd.concat([res, e], ignore_index=True)


def add_conformal(res: pd.DataFrame, model: str, alpha: float = 0.2, min_resid: int = 12) -> pd.DataFrame:
    """Online split-conformal interval: at each origin use only residuals already realised AND published
    (``features.published_by``) for that series & horizon; pooled over series if too few."""
    r = res[res.model == model].copy().sort_values("origin")
    r["resid"] = r.actual - r.pred
    lo, hi = [], []
    for _, row in r.iterrows():
        past = r[(r.h == row.h) & (r.target_month <= row.origin + pd.DateOffset(months=MAX_LAG))]
        past = past[fe.published_by(past.target_month, past.series, row.origin, row.series)]
        own = past[past.series == row.series].resid
        pool = own if len(own) >= min_resid else past.resid
        if len(pool) < min_resid:
            lo.append(np.nan); hi.append(np.nan); continue
        lo.append(row.pred + np.quantile(pool, alpha / 2))
        hi.append(row.pred + np.quantile(pool, 1 - alpha / 2))
    r["q10"], r["q90"] = lo, hi
    res = res.copy()
    res.loc[r.index, ["q10", "q90"]] = r[["q10", "q90"]]
    return res


def _pinball(y, q, a):
    d = y - q
    return np.mean(np.maximum(a * d, (a - 1) * d))


def score(res: pd.DataFrame, by=("series", "h", "model")) -> pd.DataFrame:
    """MAPE on price level, relative MAE vs naive (<1 beats naive), directional accuracy,
    and P10–P90 coverage / pinball loss where quantiles exist."""
    r = res.copy()
    r["ape"] = np.abs(np.exp(r.pred - r.actual) - 1) * 100
    r["ae"] = np.abs(r.pred - r.actual)
    naive = r[r.model == "naive"].set_index(["series", "origin", "h"]).ae.rename("ae_naive")
    r = r.join(naive, on=["series", "origin", "h"])
    r["dir_hit"] = np.where(r.pred.abs() > 1e-12, np.sign(r.pred) == np.sign(r.actual), np.nan)

    def agg(g):
        out = {"n": len(g), "MAPE": g.ape.mean(), "relMAE_vs_naive": g.ae.sum() / g.ae_naive.sum(),
               "dir_acc": g.dir_hit.mean()}
        out["cover_80"] = out["pinball"] = np.nan
        if "q10" in g and g.q10.notna().any():
            gg = g.dropna(subset=["q10", "q90"])
            out["cover_80"] = ((gg.actual >= gg.q10) & (gg.actual <= gg.q90)).mean()
            out["pinball"] = (_pinball(gg.actual, gg.q10, 0.1) + _pinball(gg.actual, gg.q90, 0.9)) / 2
        return pd.Series(out)

    return r.groupby(list(by)).apply(agg)


def dm_test(res: pd.DataFrame, model: str, bench: str = "naive") -> pd.DataFrame:
    """Diebold–Mariano (Harvey-corrected) on squared log errors, per series × h.
    Negative stat => ``model`` more accurate than ``bench``."""
    rows = []
    for (s, h), g in res[res.model.isin([model, bench])].groupby(["series", "h"]):
        w = g.pivot_table(index="origin", columns="model", values="pred").dropna()
        if model not in w or bench not in w or len(w) < 12:
            continue
        a = g.drop_duplicates("origin").set_index("origin").actual.reindex(w.index)
        d = (a - w[model]) ** 2 - (a - w[bench]) ** 2
        n, dbar = len(d), d.mean()
        dv = d.values
        gam = [np.cov(dv[k:], dv[:n - k])[0, 1] if k else dv.var(ddof=1) for k in range(h)]
        var = (gam[0] + 2 * sum(gam[1:])) / n
        if var <= 0:
            continue
        dm = dbar / np.sqrt(var) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        rows.append(dict(series=s, h=h, DM=dm, p_value=2 * stats.t.sf(abs(dm), n - 1)))
    return pd.DataFrame(rows).set_index(["series", "h"])


def window_label(res: pd.DataFrame) -> pd.Series:
    return np.where(res.target_month.between(*SHOCK), "shock 2021-22", "calm")


def forecast_now(targets: pd.DataFrame, exog: pd.DataFrame, backtest_res: pd.DataFrame, horizons=(3, 6, 9),
                 members=("ets", "arima", "theta", "dynreg", "lgbm"), alpha: float = 0.2) -> pd.DataFrame:
    """Production forecast from each series' latest observation.

    Point = equal-weight ensemble of ``members``; interval = empirical (alpha/2, 1-alpha/2) quantiles of the
    ensemble's realised backtest residuals for that series & horizon (split-conformal).
    """
    horizons = list(horizons)
    feats = {h: fe.make_features(targets, exog, h) for h in horizons}
    resid = backtest_res[backtest_res.model == "ensemble"].assign(r=lambda d: d.actual - d.pred)
    rows = []
    for c in targets.columns:
        s = targets[c].dropna()
        t = s.index[-1]
        uni = mo.univariate_forecasts(np.log(s), horizons)
        for h in horizons:
            preds = {m: uni[m][h] for m in uni}
            preds["dynreg"] = mo.dynreg_predict(feats[h], c, fe.DRIVERS[c], t, h)
            rows.append(dict(series=c, origin=t, h=h, y_origin=s.iloc[-1], **preds))
    out = pd.DataFrame(rows)
    for h in horizons:  # global model: one fit per horizon per distinct origin
        for t in out.origin.unique():
            p = mo.lgbm_predict(feats[h], t, h)
            for sname, r in p.iterrows():
                out.loc[(out.series == sname) & (out.h == h) & (out.origin == t), "lgbm"] = r["mean"]
    out["ensemble"] = out[list(members)].mean(axis=1)
    q = resid.groupby(["series", "h"]).r.quantile([alpha / 2, 1 - alpha / 2]).unstack()
    q.columns = ["r_lo", "r_hi"]
    out = out.join(q, on=["series", "h"])
    out["target_month"] = out.origin + out.h.apply(lambda h: pd.DateOffset(months=int(h)))
    out["p50"] = out.y_origin * np.exp(out.ensemble)
    out["p10"] = out.y_origin * np.exp(out.ensemble + out.r_lo)
    out["p90"] = out.y_origin * np.exp(out.ensemble + out.r_hi)
    out["chg_%"] = 100 * (np.exp(out.ensemble) - 1)
    return out
