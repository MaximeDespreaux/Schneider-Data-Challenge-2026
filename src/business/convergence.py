"""Task 2 — convergence: raw-material cost of one RM6 and one AirSeT in EUR per box, and their copper exposure.

    uv run python -m src.business.convergence

Constant quantities (BOM xlsx_v2), value weights from ``bom_weights()`` with the global config (electronics OUT,
like the submitted BOM index). Electronics is shown only as a flat sensitivity (low/mid/high EUR per box from
``electronics_slider``). Rule for the base month: a series not yet published at the origin (e.g. stainless L02)
carries its last value forward up to and including the origin month only; earlier gaps stay NaN.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import BOM_IDS, load_business_config, origin_ts  # noqa: E402
from .paths import fig_dir, out_dir  # noqa: E402

DATES = {"2021_01": "2021-01-01", "2025_09": "2025-09-01"}
ELEC_LEVELS = ["low", "mid", "high"]


def carry_to_base(s: pd.Series, base: pd.Timestamp) -> tuple[pd.Series, list]:
    """Fill only the trailing NaNs between the last published value and ``base`` (inclusive)."""
    v = s.loc[:base].copy()
    lv = v.last_valid_index()
    carried = []
    if lv is not None and lv < base:
        mask = v.index > lv
        carried = list(v.index[mask])
        v.loc[mask] = v.loc[lv]
    return v, carried


def monthly_costs(w: pd.DataFrame, panel: pd.DataFrame, base: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """EUR contribution of each material per box and month, and total cost per product (NaN if any input is NaN)."""
    out = {}
    carried_info = {}
    for prod, g in w.groupby("product"):
        parts = {}
        for r in g.itertuples():
            v, carried = carry_to_base(panel[r.series], base)
            for m in carried:
                carried_info.setdefault(m, set()).add(r.series)
            parts[f"{prod}_{r.material}_eur"] = r.kg * r.eur_per_kg * v / float(v.loc[:base].dropna().iloc[-1])
        parts = pd.DataFrame(parts)
        out.update(parts.to_dict("series"))
        out[f"{prod}_cost_eur"] = parts.sum(axis=1, skipna=False)
    df = pd.DataFrame(out)
    return df, carried_info


def run(cfg: dict | None = None) -> dict:
    from src.testbench.run import build_panel
    from src.testbench.targets import bom_weights, load_assumptions

    from .forecasts import get_origin_forecasts

    cfg = load_business_config() if cfg is None else cfg
    base = origin_ts(cfg)
    a = load_assumptions()
    w = bom_weights(a)
    panel = build_panel()
    costs, carried = monthly_costs(w, panel, base)
    costs = costs.loc[:base]
    costs.index.name = "month"

    monthly = pd.DataFrame(index=costs.index)
    monthly["RM6_cost_eur"] = costs["RM6_cost_eur"]
    monthly["AirSeT_cost_eur"] = costs["AirSeT_cost_eur"]
    monthly["gap_eur"] = monthly["AirSeT_cost_eur"] - monthly["RM6_cost_eur"]
    monthly["ratio"] = monthly["AirSeT_cost_eur"] / monthly["RM6_cost_eur"]
    for c in costs.columns:
        if c not in monthly:
            monthly[c] = costs[c]
    monthly["carried_to_base"] = [";".join(sorted(carried.get(m, []))) for m in monthly.index]
    out = out_dir()
    monthly.to_csv(out / "convergence_monthly.csv")

    fc = get_origin_forecasts(cfg)
    elec = cfg["electronics_slider"]["eur_per_box"]
    rows, fwd_all = [], {}
    for prod in cfg["products"]:
        cost = monthly[f"{prod}_cost_eur"]
        mats = sorted(w.loc[w["product"] == prod, "material"])
        r = {"product": prod}
        for tag, dt in {**DATES, "2026_09": base}.items():
            r[f"cost_{tag}_eur"] = float(cost.get(pd.Timestamp(dt), np.nan))
        r["pct_change_12m"] = r["cost_2026_09_eur"] / r["cost_2025_09_eur"] - 1
        r["pct_change_since_2021_01"] = r["cost_2026_09_eur"] / r["cost_2021_01_eur"] - 1
        for tag, dt in {"2021_01": DATES["2021_01"], "2026_09": base}.items():
            r[f"copper_share_{tag}"] = float(monthly.at[pd.Timestamp(dt), f"{prod}_copper_eur"] / cost[pd.Timestamp(dt)])
        shares = {m: monthly.at[base, f"{prod}_{m}_eur"] / cost[base] for m in mats}
        top = sorted(shares.items(), key=lambda kv: -kv[1])[:3]
        r["top3_materials_2026_09"] = ", ".join(f"{m} {100 * s:.1f}%" for m, s in top)
        g = fc[fc["series"] == BOM_IDS[prod]].set_index("target_month")
        fwd = {}
        for tm, row in g.iterrows():
            tag = pd.Timestamp(tm).strftime("%Y_%m")
            for q in ["p10", "p50", "p90"]:
                r[f"cost_{tag}_{q}_eur"] = r["cost_2026_09_eur"] * row[q] / 100
            fwd[pd.Timestamp(tm)] = {q: r["cost_2026_09_eur"] * row[q] / 100 for q in ["p10", "p50", "p90"]}
        fwd_all[prod] = fwd
        # electronics sensitivity: flat EUR per box added on top (not moved by any index)
        for lvl, eur in zip(ELEC_LEVELS, elec[prod]):
            r[f"electronics_{lvl}_eur_per_box"] = eur
            r[f"cost_2026_09_with_electronics_{lvl}_eur"] = r["cost_2026_09_eur"] + eur
            for tm in fwd:
                r[f"cost_{tm:%Y_%m}_p50_with_electronics_{lvl}_eur"] = fwd[tm]["p50"] + eur
        rows.append(r)
    summary = pd.DataFrame(rows)
    gap = float(monthly.at[base, "gap_eur"])
    ratio = float(monthly.at[base, "ratio"])
    summary["gap_2026_09_eur"] = gap
    summary["ratio_2026_09"] = ratio
    summary.to_csv(out / "convergence_summary.csv", index=False)

    _figures(monthly, fwd_all, elec, base, w, cfg)
    s = summary.set_index("product")
    return {**{f"{p}_pct_change_12m": round(100 * s.at[p, "pct_change_12m"], 2) for p in s.index},
            **{f"{p}_copper_share_2026_09_pct": round(100 * s.at[p, "copper_share_2026_09"], 1) for p in s.index},
            **{f"{p}_cost_2026_09_eur_per_box": round(s.at[p, "cost_2026_09_eur"], 1) for p in s.index},
            "gap_2026_09_eur": round(gap, 1), "ratio_2026_09": round(ratio, 3)}


def _figures(monthly, fwd_all, elec, base, w, cfg) -> None:
    fig_d = fig_dir()
    colors = {"RM6": "tab:blue", "AirSeT": "tab:orange"}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for prod in cfg["products"]:
        c = monthly[f"{prod}_cost_eur"].dropna()
        ax.plot(c.index, c.values, color=colors[prod], lw=2, label=f"{prod} material cost (no electronics)")
        lo, hi = elec[prod][0], elec[prod][-1]
        ax.fill_between(c.index, c + lo, c + hi, color=colors[prod], alpha=0.12,
                        label=f"{prod} + electronics {lo}-{hi} EUR/box (flat sensitivity)")
        fwd = fwd_all[prod]
        tms = [base] + sorted(fwd)
        p10 = [c[base]] + [fwd[t]["p10"] for t in sorted(fwd)]
        p50 = [c[base]] + [fwd[t]["p50"] for t in sorted(fwd)]
        p90 = [c[base]] + [fwd[t]["p90"] for t in sorted(fwd)]
        ax.fill_between(tms, p10, p90, color=colors[prod], alpha=0.25)
        ax.plot(tms, p50, color=colors[prod], ls="--", lw=1.5, label=f"{prod} forecast P50 (band P10-P90)")
    ax.axvline(base, color="grey", lw=0.8, ls=":")
    ax.set_title("Raw-material cost per box at constant BOM (xlsx_v2)")
    ax.set_xlabel("Month")
    ax.set_ylabel("EUR per box")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_d / "convergence_cost_per_box.png", dpi=150)
    plt.close(fig)

    mats = sorted(w["material"].unique())
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels, x = [], 0
    for prod in cfg["products"]:
        for dt in [pd.Timestamp(DATES["2021_01"]), base]:
            bottom = 0.0
            cost = monthly.at[dt, f"{prod}_cost_eur"]
            for i, m in enumerate(mats):
                col = f"{prod}_{m}_eur"
                share = monthly.at[dt, col] / cost if col in monthly else 0.0
                ax.bar(x, share * 100, bottom=bottom, color=cmap(i % 10), label=m if x == 0 else None)
                bottom += share * 100
            labels.append(f"{prod}\n{dt:%Y-%m}")
            x += 1
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("Share of material cost (%)")
    ax.set_title("Value share of each material in the cost per box")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_d / "convergence_value_shares.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print(run())
