"""Forecast models. Each returns predicted cumulative log-changes log(y[t+h]/y[t]).

Tier 0  naive, drift                      (univariate baselines)
Tier 1  ETS damped, ARIMA(1,1,0)+drift, Theta (univariate, fitted on log prices)
Tier 2  dynamic regression (ridge, per series, per horizon) on own lags + known drivers
Tier 3  global LightGBM pooled over all series (one model per horizon, + quantiles)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

UNIVARIATE = ["naive", "drift", "ets", "arima", "theta"]


# ----------------------------------------------------------------------------- tier 0-1
def univariate_forecasts(log_hist: pd.Series, horizons: list[int]) -> dict[str, dict[int, float]]:
    """Fit each univariate model once on a log-price history and return {model: {h: log-change}}."""
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from statsmodels.tsa.forecasting.theta import ThetaModel

    s = log_hist.dropna()
    s.index = pd.DatetimeIndex(s.index, freq="MS")
    H, last = max(horizons), s.iloc[-1]
    out: dict[str, dict[int, float]] = {}
    out["naive"] = {h: 0.0 for h in horizons}
    slope = (s.iloc[-1] - s.iloc[0]) / (len(s) - 1)
    out["drift"] = {h: h * slope for h in horizons}

    def _safe(name, fn):
        try:
            path = np.asarray(fn())
            out[name] = {h: float(path[h - 1] - last) for h in horizons}
        except Exception:  # fall back to naive if a fit fails
            out[name] = {h: 0.0 for h in horizons}

    warnings.simplefilter("ignore")  # statsmodels convergence chatter inside joblib workers
    _safe("ets", lambda: ETSModel(s, error="add", trend="add", damped_trend=True).fit(disp=False).forecast(H))
    _safe("arima", lambda: ARIMA(s, order=(1, 1, 0), trend="t").fit().forecast(H))
    _safe("theta", lambda: ThetaModel(s, period=12, deseasonalize=False).fit().forecast(H))
    return out


# ----------------------------------------------------------------------------- tier 2
def dynreg_predict(feat: pd.DataFrame, series: str, drivers: list[str], origin: pd.Timestamp, h: int,
                   alpha: float = 50.0) -> float:
    """Ridge regression of the h-step log-change on own lags + series-specific drivers.

    Trains only on rows whose target was realised by the origin (t + h <= origin).
    No intercept + strong ridge = shrinkage towards the random walk (predict 0 change
    unless the features say otherwise); a historical-mean intercept behaves like the
    losing 'drift' baseline.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    f = feat.xs(series, level="series")
    cols = ["own_dlog_l0", "own_mom3", "own_dev_ma12"] + [
        c for c in f.columns if any(c.startswith(f"x_{d}_") or c == f"x_{d}" for d in drivers)]
    cols = [c for c in cols if f[c].notna().any()]
    train = f.loc[: origin - pd.DateOffset(months=h)].dropna(subset=cols + ["y"])
    x0 = f.loc[[origin], cols]
    if len(train) < 24 or x0.isna().any(axis=None):
        cols = ["own_dlog_l0", "own_mom3", "own_dev_ma12"]
        train = f.loc[: origin - pd.DateOffset(months=h)].dropna(subset=cols + ["y"])
        x0 = f.loc[[origin], cols]
        if len(train) < 24 or x0.isna().any(axis=None):
            return 0.0
    sc = StandardScaler().fit(train[cols])
    model = Ridge(alpha=alpha, fit_intercept=False).fit(sc.transform(train[cols]), train["y"])
    return float(model.predict(sc.transform(x0))[0])


# ----------------------------------------------------------------------------- tier 3
# small, heavily regularised trees; boost from 0 (not the mean) = shrink towards the random walk;
# n_jobs=1 because the backtest already parallelises over origins
LGB_PARAMS = dict(n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                  verbose=-1, n_jobs=1)


def lgbm_predict(feat: pd.DataFrame, origin: pd.Timestamp, h: int,
                 quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)) -> pd.DataFrame:
    """Global LightGBM pooled over all series. Returns one row per series observed at origin:
    columns ``mean`` and ``q{α}`` for each quantile (all in log-change space)."""
    import lightgbm as lgb

    cols = [c for c in feat.columns if c.startswith(("own_", "x_"))]
    df = feat.reset_index()
    df["series_id"] = df["series"].astype("category")
    df["family_id"] = df["family"].astype("category")
    X_cols = cols + ["series_id", "family_id"]
    train = df[(df.month <= origin - pd.DateOffset(months=h)) & df.y.notna() & df.y_level.notna()]
    test = df[(df.month == origin) & df.y_level.notna()]
    if test.empty or len(train) < 100:
        return pd.DataFrame()
    out = pd.DataFrame(index=test.series.values)
    m = lgb.LGBMRegressor(objective="regression", boost_from_average=False, random_state=0,
                          **LGB_PARAMS).fit(train[X_cols], train.y)
    out["mean"] = m.predict(test[X_cols])
    for q in quantiles:
        mq = lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=0, **LGB_PARAMS).fit(train[X_cols], train.y)
        out[f"q{int(q * 100)}"] = mq.predict(test[X_cols])
    # enforce monotone quantiles
    qc = [c for c in out if c.startswith("q")]
    out[qc] = np.sort(out[qc].values, axis=1)
    return out
