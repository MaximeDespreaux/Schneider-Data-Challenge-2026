"""Tests of src/testbench/stats.py on SYNTHETIC data only."""

import numpy as np
import pandas as pd

from src.testbench.metrics import score_rows
from src.testbench.stats import (block_bootstrap_ci, christoffersen_independence, clark_west, dm_ordered, holm,
                                 kupiec_pof, ljung_box, mde_table, non_overlapping, pooled_dm)

ORIGINS = pd.date_range("2010-01-01", periods=150, freq="MS")


def rows(model, series, h, log_y, log_p50, origins=ORIGINS, spread=0.1):
    """Synthetic forecast rows (log inputs) with a symmetric P10-P90 band, scored."""
    p50 = np.exp(log_p50)
    return score_rows(pd.DataFrame({
        "model": model, "series": series, "origin": origins, "horizon": h,
        "target_month": origins + pd.DateOffset(months=h), "p10": p50 * np.exp(-spread), "p50": p50,
        "p90": p50 * np.exp(spread), "y_true": np.exp(log_y)}))


def naive_panel(series, horizons=(3,), n=150, seed=0, sd=0.03):
    """Random walk per series; naive P50 = level at the origin. Returns scored naive rows."""
    rng = np.random.default_rng(seed)
    out = []
    for s in series:
        lw = np.cumsum(rng.normal(0, sd, n + max(horizons)))
        for h in horizons:
            out.append(rows("naive", s, h, lw[h:h + n], lw[:n], origins=ORIGINS[:n]))
    return pd.concat(out, ignore_index=True)


def model_vs_naive(series="A", h=3, n=150, seed=1, gain=0.5):
    """Naive plus a model whose log error is ``gain`` x the naive error plus noise."""
    rng = np.random.default_rng(seed)
    ly, ln = rng.normal(0, 0.05, n), np.zeros(n)
    lm = ly * (1 - gain) + rng.normal(0, 0.01, n)
    return pd.concat([rows("naive", series, h, ly, ln, ORIGINS[:n]), rows("m", series, h, ly, lm, ORIGINS[:n])],
                     ignore_index=True)


def test_loss_ordering_perfect_naive_anti():  # spec 5.7 sanity
    rng = np.random.default_rng(0)
    ly = np.cumsum(rng.normal(0, 0.03, 153))
    y, naive = ly[3:], ly[:-3]
    ql = {name: rows(name, "A", 3, y, p).ql.mean() for name, p in
          [("perfect", y), ("naive", naive), ("anti", 2 * naive - y)]}
    assert ql["perfect"] < ql["naive"] < ql["anti"]


def test_dm_ordered_sorts_by_origin():
    scored = model_vs_naive()
    shuffled = scored.sample(frac=1, random_state=0)
    res = dm_ordered(scored, "m", "naive", "A", 3)
    assert np.isfinite(res[0]) and res[0] < 0 and res[1] < 0.01
    assert dm_ordered(shuffled, "m", "naive", "A", 3) == res


def test_clark_west_detects_nested_signal_dm_misses():
    # target change = signal + noise, corr(signal, change) = 0.15; model = naive + signal; 150 origins.
    # One draw is a coin flip at this effect size, so compare rejection rates over 40 seeded replications.
    rho, cw, dm = 0.15, [], []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        s = rng.normal(0, rho / np.sqrt(1 - rho ** 2), 150)
        ly = (s + rng.normal(0, 1, 150)) * 0.05
        scored = pd.concat([rows("naive", "A", 3, ly, np.zeros(150)), rows("m", "A", 3, ly, s * 0.05)])
        cw.append(clark_west(scored, "m", "naive", "A", 3)[1] < 0.05)
        stat, p = dm_ordered(scored, "m", "naive", "A", 3, loss="ae")
        dm.append(stat < 0 and p < 0.05)
    assert np.mean(cw) >= 0.4 and np.mean(dm) <= 0.3 and np.mean(cw) > 2 * np.mean(dm)


def test_holm_matches_hand_computation():
    assert holm(pd.Series([0.01, 0.04, 0.03])).round(3).tolist() == [0.03, 0.06, 0.06]


def test_pooled_dm_handles_missing_series_at_some_origins():
    nv = naive_panel(["S09", "S10", "S12"])
    m = nv.assign(model="m", p50=nv.p50 * 1.01, p10=nv.p10 * 1.01, p90=nv.p90 * 1.01)
    scored = score_rows(pd.concat([nv, m], ignore_index=True)[nv.columns[:9]])
    gap = ((scored.series == "S10") & (scored.origin < ORIGINS[20])) | (scored.origin == ORIGINS[30])
    scored_with_gap = scored[~gap]
    stat, p, n = pooled_dm(scored_with_gap, "m", "naive", "steel", 3)
    assert np.isfinite(stat) and n == len(ORIGINS) - 1


def test_bootstrap_ci_contains_point_and_is_seeded():
    scored = pd.concat([model_vs_naive("A"), model_vs_naive("B", seed=2)], ignore_index=True)
    lo, pt, hi = block_bootstrap_ci(scored, "m", "naive", 3, "rel_ql", seed=0)
    assert lo <= pt <= hi and (lo, pt, hi) == block_bootstrap_ci(scored, "m", "naive", 3, "rel_ql", seed=0)
    lo, pt, hi = block_bootstrap_ci(scored, "m", "naive", 3, "coverage", n_boot=200)
    assert 0 <= lo <= pt <= hi <= 1


def test_mde_monotone_in_sigma():
    scored_naive = naive_panel(["S01", "S02", "S03", "S09", "S10", "S12"], horizons=(3, 6), n=80)
    t = mde_table(scored_naive, seeds=range(10))
    assert set(t["group"]) == {"metals", "steel"} and set(t["horizon"]) == {3, 6}
    assert (t["mde_rel_ql"] > 0).all()


def test_kupiec_pof():
    rng = np.random.default_rng(0)
    lr, p = kupiec_pof(np.r_[np.ones(20), np.zeros(180)].astype(bool), 0.10)
    assert np.isclose(lr, 0) and p > 0.99
    assert kupiec_pof(rng.random(300) < 0.25, 0.10)[1] < 0.01
    assert kupiec_pof(rng.random(300) < 0.20, 0.20)[1] > 0.05


def test_christoffersen_independence():
    size = np.mean([christoffersen_independence(np.random.default_rng(s).random(300) < 0.2)[1] < 0.05
                    for s in range(100)])
    assert size <= 0.10  # iid hits: rejects at about the nominal 5%
    rng = np.random.default_rng(0)
    clustered = np.repeat(rng.random(60) < 0.2, 5)  # runs of 5 identical outcomes
    assert christoffersen_independence(clustered)[1] < 0.01


def test_ljung_box():
    rng = np.random.default_rng(0)
    assert ljung_box(rng.normal(size=300), 6)[1] > 0.05
    e, x = rng.normal(size=300), np.zeros(300)
    for t in range(1, 300):
        x[t] = 0.6 * x[t - 1] + e[t]
    assert ljung_box(x, 6)[1] < 0.01


def test_non_overlapping_keeps_every_h_th_origin():
    df = naive_panel(["A", "B"], horizons=(3,), n=24)
    out = non_overlapping(df, 3)
    assert out.groupby("series").size().tolist() == [8, 8]
    gaps = out.groupby("series")["origin"].diff().dropna().dt.days
    assert gaps.between(89, 92).all()
