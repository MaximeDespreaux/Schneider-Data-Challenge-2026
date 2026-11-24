"""Forecast-comparison statistics on scored rows (``metrics.score_rows``), spec section 5.7.

All loss differentials are ``loss(model) - loss(ref)`` sorted by origin, so a negative stat means the model is
better. ``extra_lag`` adds the publication lag in realtime mode (step = horizon + lag, HAC uses step - 1 lags).

Overlapping h-step errors are MA(h-1) by construction. DM-type tests absorb that with HAC; the backtesting tests
that assume independence (Christoffersen, Ljung-Box) must run on ``non_overlapping`` origins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

from src.testbench.metrics import _hac_t, score_rows

GROUPS = {
    "metals": ["S01", "S02", "S03", "S04", "S05", "D01", "D02", "S07", "S08"],
    "steel": ["S09", "S10", "S12"],
    "plastics": ["S11", "S13"],
    "bom": ["RM6_NE_IQI_433520", "RMAirSeT_IIQ_24kV_IEC"],
    "steel_eur": ["L01", "L02", "L03", "L01_FR"],
    "other": ["EP01", "EL01"],
}
KEYS = ["series", "origin", "horizon"]


def _join(scored, model, ref, series=None, horizon=None) -> pd.DataFrame:
    """Model and ref rows inner-joined on (series, origin, horizon), suffixes _m/_r, sorted by origin.
    ``series`` may be one id or a list."""
    s = scored
    if series is not None:
        s = s[s.series.isin([series] if isinstance(series, str) else series)]
    if horizon is not None:
        s = s[s.horizon == horizon]
    m, r = s[s.model == model], s[s.model == ref]
    return m.merge(r, on=KEYS, suffixes=("_m", "_r")).sort_values(["origin", "series", "horizon"], kind="mergesort")


def paired(scored, model, ref, series=None, horizon=None, loss="ql") -> pd.DataFrame:
    """Loss differential ``d = loss_model - loss_ref`` per common row: columns origin, d, sorted by origin."""
    j = _join(scored, model, ref, series, horizon)
    return pd.DataFrame({"origin": j["origin"], "d": j[f"{loss}_m"] - j[f"{loss}_r"]}).reset_index(drop=True)


def dm_ordered(scored, model, ref, series, horizon, extra_lag=0, loss="ql") -> tuple[float, float]:
    """Diebold-Mariano on origin-sorted differentials: HAC (horizon + extra_lag - 1 lags), HLN, t(n-1), two-sided."""
    stat, n = _hac_t(paired(scored, model, ref, series, horizon, loss)["d"].to_numpy(), horizon + extra_lag - 1)
    return stat, float(2 * stats.t.sf(abs(stat), n - 1)) if np.isfinite(stat) else np.nan


def clark_west(scored, model, ref, series, horizon, extra_lag=0) -> tuple[float, float]:
    """Clark-West (2007) for a model nesting ``ref`` (point forecasts, log errors e = log y - log p50):
    f = e_ref^2 - (e_mod^2 - (p_ref - p_mod)^2), HAC t-stat; one-sided p for 'model better' (stat > 0)."""
    j = _join(scored, model, ref, series, horizon)
    dp = np.log(j["p50_r"]) - np.log(j["p50_m"])
    f = j["err_r"] ** 2 - (j["err_m"] ** 2 - dp ** 2)
    stat, n = _hac_t(f.to_numpy(), horizon + extra_lag - 1)
    return stat, float(stats.t.sf(stat, n - 1)) if np.isfinite(stat) else np.nan


def pooled_dm(scored, model, ref, group, horizon, extra_lag=0, loss="ql") -> tuple[float, float, int]:
    """Group DM: mean differential over the group's series available at each origin, then HAC on that series.
    ``group`` is a GROUPS key or a list of ids. Returns (stat, one-sided p for 'model better', n_origins)."""
    ids = GROUPS[group] if isinstance(group, str) else group
    d = paired(scored, model, ref, ids, horizon, loss).groupby("origin")["d"].mean()  # groupby sorts by origin
    stat, n = _hac_t(d.to_numpy(), horizon + extra_lag - 1)
    return stat, float(stats.t.cdf(stat, n - 1)) if np.isfinite(stat) else np.nan, n


def holm(pvals: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values (same index; NaN stays NaN and is not counted)."""
    p = pvals.dropna().sort_values()
    adj = (p * np.arange(len(p), 0, -1)).cummax().clip(upper=1)
    return adj.reindex(pvals.index)


def block_bootstrap_ci(scored, model, ref, horizon, stat, block=None, n_boot=2000, seed=0,
                       level=0.95) -> tuple[float, float, float]:
    """Moving-block bootstrap over origins (all series at ``horizon``, rows of an origin move together).
    stat: 'rel_mae' (sum ae_m / sum ae_r), 'rel_ql' (same with ql) or 'coverage' (model's P10-P90 hit rate).
    Block length defaults to the horizon. Returns (lo, point, hi), percentile interval."""
    j = _join(scored, model, ref, horizon=horizon)
    num, den = {"rel_mae": ("ae_m", "ae_r"), "rel_ql": ("ql_m", "ql_r"), "coverage": ("covered_m", None)}[stat]
    g = j.assign(one=1.0).groupby("origin")
    a, b = g[num].sum().to_numpy(float), (g[den].sum() if den else g["one"].sum()).to_numpy(float)
    T = len(a)
    L = min(block or horizon, T)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, T - L + 1, size=(n_boot, -(-T // L)))
    idx = (starts[:, :, None] + np.arange(L)).reshape(n_boot, -1)[:, :T]
    boot = a[idx].sum(1) / b[idx].sum(1)
    q = (1 - level) / 2
    return float(np.quantile(boot, q)), float(a.sum() / b.sum()), float(np.quantile(boot, 1 - q))


def placebo_rows(base_rows_naive, sigma, seed) -> pd.DataFrame:
    """Placebo model ``placebo_<sigma>_<seed>``: naive p10/p50/p90 x exp(N(0, sigma)), one draw per row.
    Re-scored when the input was scored."""
    rng = np.random.default_rng(seed)
    out = base_rows_naive.copy()
    out[["p10", "p50", "p90"]] = out[["p10", "p50", "p90"]].mul(np.exp(rng.normal(0, sigma, len(out))), axis=0)
    out["model"] = f"placebo_{sigma}_{seed}"
    return score_rows(out) if "ql" in out else out


def mde_table(scored_naive, sigmas=(0.005, 0.01, 0.02, 0.04, 0.08), seeds=range(20), alpha=0.05,
              power=0.8) -> pd.DataFrame:
    """Minimum detectable effect per group x horizon: the smallest mean relative-QL gap QL_placebo/QL_naive - 1
    (averaged over seeds) at which pooled_dm(naive better than placebo) rejects at ``alpha`` in >= ``power`` of
    seeds. NaN when no sigma reaches that power. Groups without rows are omitted."""
    nv = scored_naive[scored_naive.model == "naive"]
    res = []
    for sigma in sigmas:
        for seed in seeds:
            pl = placebo_rows(nv, sigma, seed)
            both = pd.concat([nv, pl], ignore_index=True)
            for grp, ids in GROUPS.items():
                for h in sorted(nv.horizon.unique()):
                    sel = nv.series.isin(ids) & (nv.horizon == h)
                    if not sel.any():
                        continue
                    _, p, n = pooled_dm(both, "naive", pl.model.iloc[0], grp, h)
                    gap = pl.loc[sel, "ql"].mean() / nv.loc[sel, "ql"].mean() - 1
                    res.append((grp, h, sigma, p < alpha, gap, n))
    r = pd.DataFrame(res, columns=["group", "horizon", "sigma", "reject", "gap", "n_origins"])
    r = r.groupby(["group", "horizon", "sigma"], as_index=False).agg(
        rate=("reject", "mean"), gap=("gap", "mean"), n_origins=("n_origins", "max"))
    rows = []
    for (grp, h), g in r.groupby(["group", "horizon"]):
        ok = g[g.rate >= power].sort_values("sigma")
        rows.append({"group": grp, "horizon": h, "mde_rel_ql": ok.gap.iloc[0] if len(ok) else np.nan,
                     "mde_sigma": ok.sigma.iloc[0] if len(ok) else np.nan, "n_origins": g.n_origins.max()})
    return pd.DataFrame(rows)


def _ll(a, b):
    """Bernoulli log-likelihood at the MLE for a ones and b zeros."""
    return xlogy(a, a / (a + b)) + xlogy(b, b / (a + b)) if a + b else 0.0


def kupiec_pof(hits, p) -> tuple[float, float]:
    """Kupiec (1995) unconditional coverage LR, chi2(1). ``hits``: boolean exceedances; ``p``: expected rate
    (0.10 for each tail of P10/P90, 0.20 for misses of the 80% band)."""
    h = np.asarray(hits, bool)
    x, n = h.sum(), len(h)
    lr = max(2 * (_ll(x, n - x) - xlogy(x, p) - xlogy(n - x, 1 - p)), 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_independence(hits) -> tuple[float, float]:
    """Christoffersen (1998) independence LR (first-order Markov vs iid), chi2(1). Run it on ``non_overlapping``
    origins: overlapping h-step hits are serially dependent by construction."""
    h = np.asarray(hits, int)
    prev, nxt = h[:-1], h[1:]
    n = {(i, j): int(((prev == i) & (nxt == j)).sum()) for i in (0, 1) for j in (0, 1)}
    lr = 2 * (_ll(n[0, 1], n[0, 0]) + _ll(n[1, 1], n[1, 0]) - _ll(n[0, 1] + n[1, 1], n[0, 0] + n[1, 0]))
    lr = max(lr, 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def ljung_box(resid, lags) -> tuple[float, float]:
    """Ljung-Box Q at ``lags`` (statsmodels). Run it on ``non_overlapping`` origins: overlapping h-step errors are
    MA(h-1) by construction and always fail it."""
    from statsmodels.stats.diagnostic import acorr_ljungbox

    r = acorr_ljungbox(np.asarray(resid, float), lags=[lags])
    return float(r["lb_stat"].iloc[0]), float(r["lb_pvalue"].iloc[0])


def non_overlapping(df, horizon) -> pd.DataFrame:
    """Keep every ``horizon``-th month from each series' first origin, so the kept h-step errors do not overlap
    (overlapping ones are MA(h-1)). Pass the step (horizon + publication lag) in realtime mode."""
    m = df["origin"].dt.year * 12 + df["origin"].dt.month
    return df[(m - m.groupby(df["series"]).transform("min")) % horizon == 0]
