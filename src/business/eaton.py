"""Task 4 — Eaton price gap: is the SF6-free brand pricing differently from the SF6 brands?

    uv run python -m src.business.eaton

The price benchmark is SYNTHETIC (yearly P50 per brand, 2017-2026): a signal, not proof.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import load_business_config  # noqa: E402
from .paths import fig_dir, out_dir  # noqa: E402

# price-benchmark brand name -> teardown brand name
BENCH_TO_TEARDOWN = {"Schneider Electric": "Schneider", "SwitchGear Company": "SGC"}
SF6_MIN_KG = 0.1


def sf6_flags(td: pd.DataFrame) -> pd.Series:
    """sf6_free per teardown brand: True if no 'sf6' material above 0.1 kg; NaN if the teardown is broken."""
    out = {}
    for brand, g in td.groupby("brand"):
        if bool(g["broken"].any()):
            out[brand] = np.nan
        else:
            out[brand] = not bool(((g["group"] == "sf6") & (g["mass_kg"] > SF6_MIN_KG)).any())
    return pd.Series(out, name="sf6_free", dtype=object)


def run(cfg: dict | None = None) -> dict:
    from src import data as d

    cfg = load_business_config() if cfg is None else cfg
    bench = d.load_rmu_price_benchmark().copy()
    bench["brand"] = bench["Brand"].replace(BENCH_TO_TEARDOWN)
    flags = sf6_flags(d.load_competitor_teardowns())
    if flags.get("Eaton") is not True:
        raise ValueError(f"Eaton is not flagged SF6-free from its teardown (sf6_free = {flags.get('Eaton')}).")
    bench["sf6_free"] = bench["brand"].map(flags)

    p50 = bench.pivot_table(index="Year", columns="brand", values="P50")
    yoy = bench.pivot_table(index="Year", columns="brand", values="YoY P50")
    rel = p50.div(p50["Schneider"], axis=0)
    sf6_brands = [b for b in p50.columns if flags.get(b) is not True]  # SF6 or unknown (broken teardown)

    tab = pd.DataFrame(index=p50.index)
    tab["eaton_p50"] = p50["Eaton"]
    tab["schneider_p50"] = p50["Schneider"]
    tab["sf6_brands"] = ", ".join(sf6_brands)
    tab["sf6_avg_p50"] = p50[sf6_brands].mean(axis=1)
    tab["eaton_vs_sf6_avg"] = tab["eaton_p50"] / tab["sf6_avg_p50"]
    tab["eaton_rel_schneider"] = rel["Eaton"]
    tab["eaton_yoy_p50"] = yoy["Eaton"]
    tab["sf6_avg_yoy_p50"] = yoy[sf6_brands].mean(axis=1)
    for b in p50.columns:
        tab[f"rel_schneider_{b}"] = rel[b]
        tab[f"yoy_p50_{b}"] = yoy[b]
        tab[f"sf6_free_{b}"] = flags.get(b, np.nan)
    below = tab.index[tab["eaton_p50"] < tab["schneider_p50"]]
    first_below = int(below.min()) if len(below) else None
    tab["eaton_first_year_below_schneider"] = first_below
    tab.to_csv(out_dir() / "eaton_price_gap.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    for b in rel.columns:
        lw, alpha = (3, 1.0) if b == "Eaton" else (1.3, 0.7)
        ax.plot(rel.index, rel[b], lw=lw, alpha=alpha, marker="o" if b == "Eaton" else None,
                label=f"{b} ({'SF6-free' if flags.get(b) is True else 'SF6' if flags.get(b) is False else 'unknown'})")
    ax.axhline(1.0, color="grey", ls=":")
    ax.set_xlabel("Year")
    ax.set_ylabel("P50 price relative to Schneider (Schneider = 1.0)")
    ax.set_title("RMU price benchmark (synthetic): Eaton vs SF6 brands")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir() / "eaton_vs_sf6_brands.png", dpi=150)
    plt.close(fig)

    y0, y1 = int(p50.index.min()), int(p50.index.max())
    return {f"eaton_rel_schneider_{y0}": round(float(rel.at[y0, "Eaton"]), 3),
            f"eaton_rel_schneider_{y1}": round(float(rel.at[y1, "Eaton"]), 3),
            f"eaton_yoy_{y1}_pct": round(100 * float(yoy.at[y1, "Eaton"]), 1),
            f"sf6_avg_yoy_{y1}_pct": round(100 * float(tab.at[y1, "sf6_avg_yoy_p50"]), 1),
            "eaton_first_year_below_schneider": first_below if first_below is not None else "never"}


if __name__ == "__main__":
    print(run())
