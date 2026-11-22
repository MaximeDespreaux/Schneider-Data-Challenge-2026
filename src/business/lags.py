"""Task 3b — pass-through lag: how many months after a market move does it show up in the slower price?

    uv run python -m src.business.lags

On monthly log changes dx, dy: cross-correlations corr(dy_t, dx_{t-k}) and a distributed-lag OLS
dy_t = a + Σ_k b_k dx_{t-k} (numpy lstsq). Cumulative pass-through C_k = Σ_{j<=k} b_j; lag_50 / lag_80 = first k
where C_k reaches 50 % / 80 % of C_max (only if C_max > 0). Full overlap, targets before 2024 and from 2024.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import load_business_config  # noqa: E402
from .paths import fig_dir, out_dir  # noqa: E402

SPLIT = pd.Timestamp("2024-01-01")
SAMPLES = {"full": (None, None), "before_2024": (None, SPLIT), "from_2024": (SPLIT, None)}
# shock-table item -> lag pair name
LAG_PAIR_FOR_ITEM = {"copper": "copper_LME_to_EU_copper_PPI", "carbon_steel": "MEPS_CR_coil_to_EU_steel_PPI",
                     "all_steel": "MEPS_CR_coil_to_EU_steel_PPI", "galvanised_steel": "MEPS_HDG_to_DE_steel_PPI",
                     "stainless_steel": "nickel_LME_to_CRU_stainless_304"}


def lag_stats(x: pd.Series, y: pd.Series, max_lag: int, start=None, end=None) -> dict:
    """Cross-correlations and distributed-lag pass-through of log changes of x into y."""
    dx = np.log(x.astype(float)).diff()
    dy = np.log(y.astype(float)).diff()
    lags = pd.DataFrame({f"dx_{k}": dx.shift(k) for k in range(max_lag + 1)})
    df = lags.assign(dy=dy).dropna()
    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index < end]
    out = {"n": int(len(df)), "start": df.index.min(), "end": df.index.max()}
    if len(df) < max_lag + 5:
        return out
    corr = [float(np.corrcoef(df["dy"], df[f"dx_{k}"])[0, 1]) for k in range(max_lag + 1)]
    X = np.column_stack([np.ones(len(df)), df[[f"dx_{k}" for k in range(max_lag + 1)]].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, df["dy"].to_numpy(), rcond=None)
    b = coef[1:]
    c = np.cumsum(b)
    cmax = float(c[-1])

    def first_reach(frac):
        if cmax <= 0:
            return np.nan
        hit = np.nonzero(c >= frac * cmax)[0]
        return int(hit[0]) if len(hit) else np.nan

    out.update({"peak_lag": int(np.nanargmax(corr)), "peak_corr": float(np.nanmax(corr)), "C_max": cmax,
                "lag_50": first_reach(0.5), "lag_80": first_reach(0.8)})
    out.update({f"corr_{k}": v for k, v in enumerate(corr)})
    out.update({f"b_{k}": float(v) for k, v in enumerate(b)})
    out.update({f"C_{k}": float(v) for k, v in enumerate(c)})
    return out


def join_lags(shock: pd.DataFrame, lag_tab: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add lag_50 / lag_80 (full sample) of the matching pair to each shock-table row (NaN if none)."""
    full = lag_tab[lag_tab["sample"] == "full"].set_index("name")
    shock = shock.drop(columns=[c for c in ["lag_pair", "lag_50", "lag_80"] if c in shock], errors="ignore")
    pair = shock["item"].map(LAG_PAIR_FOR_ITEM)
    shock["lag_pair"] = pair
    for col in ["lag_50", "lag_80"]:
        shock[col] = pair.map(full[col]) if col in full else np.nan
    return shock


def _panel() -> pd.DataFrame:
    try:
        from src.external.signals import build_panel as long_panel
        return long_panel(long=True, external=False)
    except Exception:  # noqa: BLE001  (missing public downloads)
        from src.testbench.run import build_panel
        return build_panel()


def run(cfg: dict | None = None) -> dict:
    from src import data as d

    cfg = load_business_config() if cfg is None else cfg
    la = cfg["lag_analysis"]
    max_lag = int(la["max_lag"])
    panel = _panel()
    pairs = [(p["name"], p["x"], p["y"], panel[p["x"]], panel[p["y"]]) for p in la["pairs"]
             if p["x"] in panel and p["y"] in panel]
    ep = d.load_epoxy()
    y_ep = d.load_epoxy_series()
    for ing in la["epoxy_ingredients"]:
        x = pd.to_numeric(ep[ing], errors="coerce").groupby(level=0).mean()
        short = ing.split("_")[0].lower()
        pairs.append((f"epoxy_{short}_to_EP01", ing, "EP01", x, y_ep))
    rows = []
    for name, xn, yn, x, y in pairs:
        both = pd.concat([x.rename("x"), y.rename("y")], axis=1).sort_index()
        for sample, (start, end) in SAMPLES.items():
            rows.append({"name": name, "x": xn, "y": yn, "sample": sample,
                         **lag_stats(both["x"], both["y"], max_lag, start, end)})
    tab = pd.DataFrame(rows)
    out = out_dir()
    tab.to_csv(out / "lag_table.csv", index=False)

    shock_path = out / "shock_table.csv"
    if shock_path.exists():
        join_lags(pd.read_csv(shock_path), tab, cfg).to_csv(shock_path, index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    # epoxy ingredient pairs stay in lag_table.csv but not in the figure / REPORT: their pass-through is weak
    full = tab[(tab["sample"] == "full") & ~tab["name"].str.startswith("epoxy_")]
    ks = list(range(max_lag + 1))
    for r in full.itertuples():
        ax.plot(ks, [getattr(r, f"C_{k}") for k in ks], marker="o", label=r.name)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Months after the market move (k)")
    ax.set_ylabel("Cumulative pass-through C_k (share of a 1 % move)")
    ax.set_title("Cumulative pass-through of market prices into slower prices (full sample)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir() / "lag_cumulative_passthrough.png", dpi=150)
    plt.close(fig)

    head = {}
    for r in full.itertuples():
        head[f"{r.name}_lag50_lag80"] = f"{_fmt(r.lag_50)} / {_fmt(r.lag_80)} months (C_max {r.C_max:.2f})"
    return head


def _fmt(v) -> str:
    return "n/a" if pd.isna(v) else str(int(v))


if __name__ == "__main__":
    print(run())
