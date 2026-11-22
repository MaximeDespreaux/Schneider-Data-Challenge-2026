"""Task 7 — total cost of ownership, RM6 (SF6) vs AirSeT (SF6-free): vectorised Monte Carlo.

    uv run python -m src.business.tco

Each input is drawn independently: ``choices`` uniformly, ``[low, high]`` uniformly, ``value`` fixed.
r = discount rate, L = lifetime, A = (1 − (1+r)^−L) / r (annuity factor), D = (1+r)^−L (end discount).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import load_business_config, selling_prices  # noqa: E402
from .paths import fig_dir, out_dir  # noqa: E402

INPUTS = ["lifetime_yrs", "sf6_charge_kg", "leak_rate_pct_yr", "eol_loss_pct", "gwp_sf6", "carbon_price_eur_t",
          "discount_rate", "airset_premium_pct", "maint_rm6_eur_yr", "maint_airset_reduction", "sf6_admin_eur_yr",
          "eol_sf6_recovery_eur"]


def draw_inputs(tco_cfg: dict, n: int, rng: np.random.Generator) -> dict:
    out = {}
    for k in INPUTS:
        spec = tco_cfg[k]
        if "choices" in spec:
            out[k] = rng.choice(np.asarray(spec["choices"], dtype=float), size=n)
        elif "low" in spec:
            out[k] = rng.uniform(float(spec["low"]), float(spec["high"]), size=n)
        else:
            out[k] = np.full(n, float(spec["value"]))
    return out


def compute(x: dict, capex: float) -> dict:
    """All TCO quantities for arrays of inputs (vectorised)."""
    r, L = x["discount_rate"], x["lifetime_yrs"]
    A = (1 - (1 + r) ** -L) / r
    D = (1 + r) ** -L
    charge, leak, eol, gwp, cprice = (x["sf6_charge_kg"], x["leak_rate_pct_yr"], x["eol_loss_pct"], x["gwp_sf6"],
                                      x["carbon_price_eur_t"])
    t_leak_yr = charge * leak / 100 * gwp / 1000
    t_eol = charge * (1 - leak / 100 * L) * eol / 100 * gwp / 1000
    sf6_lifetime = t_leak_yr * L + t_eol
    carbon_cost = cprice * t_leak_yr * A + cprice * t_eol * D
    tco_rm6 = capex + (x["maint_rm6_eur_yr"] + x["sf6_admin_eur_yr"]) * A + x["eol_sf6_recovery_eur"] * D + carbon_cost
    maint_air = x["maint_rm6_eur_yr"] * (1 - x["maint_airset_reduction"])
    tco_air = capex * (1 + x["airset_premium_pct"]) + maint_air * A
    diff = tco_air - tco_rm6
    breakeven = (tco_rm6 - maint_air * A - capex) / capex
    return {"A": A, "D": D, "t_leak_yr": t_leak_yr, "t_eol": t_eol, "sf6_lifetime_tco2e": sf6_lifetime,
            "carbon_cost": carbon_cost, "tco_rm6": tco_rm6, "maint_air": maint_air, "tco_air": tco_air, "diff": diff,
            "breakeven_premium": breakeven}


def simulate(tco_cfg: dict, capex: float, seed: int, n: int | None = None) -> pd.DataFrame:
    n = int(tco_cfg["n_draws"]) if n is None else n
    x = draw_inputs(tco_cfg, n, np.random.default_rng(seed))
    res = compute(x, capex)
    df = pd.DataFrame({**x, **res})
    cp = tco_cfg["carbon_price_eur_t"]
    df["carbon_label"] = df["carbon_price_eur_t"].map(dict(zip(map(float, cp["choices"]), cp["labels"])))
    return df


def _stats(g: pd.DataFrame) -> dict:
    out = {"n": len(g), "share_airset_cheaper": float((g["diff"] < 0).mean())}
    for c in ["diff", "breakeven_premium"]:
        out.update({f"{c}_mean": g[c].mean(), f"{c}_median": g[c].median(), f"{c}_p10": g[c].quantile(0.1),
                    f"{c}_p90": g[c].quantile(0.9)})
    return out


def summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = [{"group": "overall", **_stats(df)}]
    for lab, g in df.groupby("carbon_label", sort=False):
        rows.append({"group": f"carbon_{lab}_{g['carbon_price_eur_t'].iloc[0]:g}EUR_t", **_stats(g)})
    return pd.DataFrame(rows)


def tornado(tco_cfg: dict, capex: float) -> pd.DataFrame:
    """One-at-a-time: each input at its low / high, others at midpoint (ranges) or middle choice (choices)."""
    def mid(spec):
        if "choices" in spec:
            ch = list(spec["choices"])
            return float(ch[len(ch) // 2])
        if "low" in spec:
            return (float(spec["low"]) + float(spec["high"])) / 2
        return float(spec["value"])

    def ends(spec):
        if "choices" in spec:
            return float(min(spec["choices"])), float(max(spec["choices"]))
        return float(spec["low"]), float(spec["high"])

    base = {k: np.array([mid(tco_cfg[k])]) for k in INPUTS}
    base_diff = float(compute(base, capex)["diff"][0])
    rows = []
    for k in INPUTS:
        spec = tco_cfg[k]
        if "value" in spec and "low" not in spec and "choices" not in spec:
            continue
        lo, hi = ends(spec)
        d_lo = float(compute({**base, k: np.array([lo])}, capex)["diff"][0])
        d_hi = float(compute({**base, k: np.array([hi])}, capex)["diff"][0])
        rows.append({"input": k, "low": lo, "high": hi, "diff_at_low": d_lo, "diff_at_high": d_hi,
                     "swing": abs(d_hi - d_lo), "base_diff": base_diff})
    return pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)


def run(cfg: dict | None = None) -> dict:
    from src.testbench.targets import load_assumptions

    cfg = load_business_config() if cfg is None else cfg
    capex = selling_prices(load_assumptions(), cfg)["RM6"]
    tc = cfg["tco"]
    df = simulate(tc, capex, int(cfg["seed"]))
    out = out_dir()
    df.to_parquet(out / "tco_draws.parquet")
    summ = summary(df)
    summ.to_csv(out / "tco_summary.csv", index=False)
    tor = tornado(tc, capex)
    tor.to_csv(out / "tco_tornado.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["diff"], bins=60, color="tab:green", alpha=0.8)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("TCO AirSeT − TCO RM6 (EUR per box, present value)")
    ax.set_ylabel("Number of draws")
    ax.set_title(f"TCO difference over {len(df):,} Monte Carlo draws (< 0: AirSeT cheaper)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir() / "tco_diff_hist.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    t = tor.iloc[::-1]
    base = float(tor["base_diff"].iloc[0])
    for i, r in enumerate(t.itertuples()):
        ax.barh(i, r.diff_at_low - base, left=base, color="tab:blue", alpha=0.8, label="input at low" if i == 0 else None)
        ax.barh(i, r.diff_at_high - base, left=base, color="tab:red", alpha=0.8, label="input at high" if i == 0 else None)
    ax.set_yticks(range(len(t)), t["input"])
    ax.axvline(base, color="black", lw=1)
    ax.set_xlabel("TCO AirSeT − TCO RM6 (EUR per box)")
    ax.set_title("TCO sensitivity (one input at a time, others at mid)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir() / "tco_tornado.png", dpi=150)
    plt.close(fig)

    head = {}
    for _, r in summ.iterrows():
        g = r["group"]
        if g == "overall" or g.endswith("_0EUR_t") or g.endswith("_126EUR_t"):
            head[f"{g}_diff_mean_eur"] = round(float(r["diff_mean"]), 0)
            head[f"{g}_diff_p10_p90_eur"] = f"{r['diff_p10']:,.0f} to {r['diff_p90']:,.0f}"
            head[f"{g}_breakeven_premium_mean_pct"] = round(100 * float(r["breakeven_premium_mean"]), 2)
            head[f"{g}_share_airset_cheaper_pct"] = round(100 * float(r["share_airset_cheaper"]), 1)
    return head


if __name__ == "__main__":
    print(run())
