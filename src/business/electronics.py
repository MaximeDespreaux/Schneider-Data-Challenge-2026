"""Task 5 — electronics slider: how much would electronics weigh in the BOM index at different EUR per box?

    uv run python -m src.business.electronics

Electronics is NOT in Schneider's BOM and is kept out of the submitted index; this is the only module that sets
``extras.electronics.include = true`` (on a copy of the assumptions, never in the config files).
The P90 change at h=6 adds up the marginal P90s of each material, which overstates the joint P90.
"""

from __future__ import annotations

import copy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import PRODUCT_KEY, load_business_config  # noqa: E402
from .paths import fig_dir, out_dir  # noqa: E402

H = 6
Q = "p90"


def material_change(fc: pd.DataFrame, material: str, series: str, cfg: dict) -> float:
    """P90 change at h=6 of one material (0 if no forecast)."""
    from .forecasts import pct_change

    have = set(fc["series"])
    if material == "electronics":
        series = "EL01"
    if material == "brass":
        mix = cfg["competitor_lever"]["brass_mix"]
        if {"D01", "S05"} <= have:
            return mix["copper"] * pct_change(fc, "D01", H, Q) + mix["zinc"] * pct_change(fc, "S05", H, Q)
        return 0.0
    return pct_change(fc, series, H, Q) if series in have else 0.0


def run(cfg: dict | None = None) -> dict:
    from src.testbench.targets import bom_weights, load_assumptions

    from .forecasts import get_origin_forecasts

    cfg = load_business_config() if cfg is None else cfg
    a = load_assumptions()
    fc = get_origin_forecasts(cfg)
    rows = []
    for prod, values in cfg["electronics_slider"]["eur_per_box"].items():
        for eur in values:
            a2 = copy.deepcopy(a)
            a2["extras"]["electronics"]["include"] = True
            a2["extras"]["electronics"]["eur_per_box_at_base"][PRODUCT_KEY[prod]]["value"] = eur
            w = bom_weights(a2)
            w = w[w["product"] == prod].set_index("material")
            total = float(w["value_eur"].sum())
            el_val = float(w.at["electronics", "value_eur"])
            p90_change = sum(float(r.weight) * material_change(fc, m, r.series, cfg) for m, r in w.iterrows())
            rows.append({"product": prod, "eur_per_box": eur, "total_bom_value_eur": total,
                         "electronics_value_share": el_val / total, "el01_plus10_eur": 0.10 * el_val,
                         "el01_plus10_pct_of_bom": 0.10 * el_val / total,
                         "copper_share_of_index": float(w.at["copper", "weight"]),
                         "index_p90_change_h6": p90_change,
                         "note": "sum of marginal P90s: overstates the joint P90"})
    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir() / "electronics_slider.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for prod, g in tab.groupby("product", sort=False):
        axes[0].plot(g["eur_per_box"], 100 * g["electronics_value_share"], marker="o", label=prod)
        axes[1].plot(g["eur_per_box"], 100 * g["index_p90_change_h6"], marker="o", label=prod)
    axes[0].set_title("Electronics share of BOM value")
    axes[0].set_ylabel("Share of index weight (%)")
    axes[1].set_title("BOM index P90 cost change at 6 months")
    axes[1].set_ylabel("Change vs origin (%)")
    for ax in axes:
        ax.set_xlabel("Electronics value (EUR per box)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Electronics slider (not in Schneider's BOM; sensitivity only)")
    fig.tight_layout()
    fig.savefig(fig_dir() / "electronics_slider.png", dpi=150)
    plt.close(fig)

    head = {}
    for prod, g in tab.groupby("product", sort=False):
        head[f"{prod}_electronics_share_range_pct"] = (f"{100 * g['electronics_value_share'].min():.1f} - "
                                                       f"{100 * g['electronics_value_share'].max():.1f} "
                                                       f"(at {g['eur_per_box'].min()}-{g['eur_per_box'].max()} EUR/box)")
    return head


if __name__ == "__main__":
    print(run())
