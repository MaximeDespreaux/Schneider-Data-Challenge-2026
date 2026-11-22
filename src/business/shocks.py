"""Task 3a — shock table: "if material X moves by Y %, how many EUR more per box?"

    uv run python -m src.business.shocks

Value weights from ``bom_weights()`` (global config: electronics out, so the electronics row is skipped as absent from
the BOM). Two cost bases: ``bom_value`` (priced BOM materials) and ``material_share`` (share of selling price, columns
suffixed ``_ms``). The pass-through lags of Task 3b are joined on by ``lags.run``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import load_business_config, selling_prices  # noqa: E402
from .paths import fig_dir, out_dir  # noqa: E402


def shock_table(w: pd.DataFrame, prices: dict, trigger_pct: float, material_share: float, materials: list,
                combos: dict, shocks_pct: list) -> pd.DataFrame:
    """Long table: one row per product × item (material or combo) × shock."""
    rows = []
    for prod, g in w.groupby("product"):
        vals = g.groupby("material")["value_eur"].sum()
        total = float(vals.sum())
        price = float(prices[prod])
        cost_ms = material_share * price
        items = [(m, "material", [m]) for m in materials if m in vals.index]
        items += [(c, "combo", [m for m in ms if m in vals.index]) for c, ms in combos.items()]
        for name, kind, ms in items:
            if not ms:
                continue
            base_val = float(vals[ms].sum())
            for sh in shocks_pct:
                x = sh / 100.0
                delta = base_val * x
                pct_cost = delta / total
                delta_ms = cost_ms * base_val / total * x
                rows.append({"product": prod, "item": name, "kind": kind, "materials": "+".join(ms),
                             "shock_pct": sh, "base_value_eur": base_val, "total_bom_value_eur": total,
                             "delta_eur": delta, "pct_of_material_cost": pct_cost, "pct_of_price": delta / price,
                             "margin_pts": -delta / price * 100,
                             "crosses_indexation_trigger": bool(abs(pct_cost) >= trigger_pct / 100),
                             "cost_base_ms_eur": cost_ms, "delta_eur_ms": delta_ms, "pct_of_price_ms": delta_ms / price,
                             "margin_pts_ms": -delta_ms / price * 100,
                             "crosses_indexation_trigger_ms": bool(abs(delta_ms / cost_ms) >= trigger_pct / 100)})
    return pd.DataFrame(rows)


def run(cfg: dict | None = None) -> dict:
    from src.testbench.targets import bom_weights, load_assumptions

    cfg = load_business_config() if cfg is None else cfg
    a = load_assumptions()
    w = bom_weights(a)
    prices = selling_prices(a, cfg)
    st = cfg["shock_table"]
    tab = shock_table(w, prices, float(a["quote_rules"]["indexation_trigger_pct"]),
                      float(a["pricing"]["material_share_of_price"]["value"]), st["materials"], st["combos"],
                      st["shocks_pct"])
    out = out_dir()
    lag_path = out / "lag_table.csv"
    if lag_path.exists():  # standalone re-run after lags: keep the lag columns
        from .lags import join_lags
        tab = join_lags(tab, pd.read_csv(lag_path), cfg)
    tab.to_csv(out / "shock_table.csv", index=False)
    wide = tab.pivot_table(index="item", columns=["product", "shock_pct"], values="delta_eur")
    wide.columns = [f"{p}_{s:+d}pct" for p, s in wide.columns]
    wide.to_csv(out / "shock_table_wide.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    sel = tab[tab["item"].isin(["copper", "all_steel"]) & tab["shock_pct"].isin([10, 20, 30])]
    groups = [(p, i) for p in cfg["products"] for i in ["copper", "all_steel"]]
    width = 0.2
    for k, (p, i) in enumerate(groups):
        s = sel[(sel["product"] == p) & (sel["item"] == i)].set_index("shock_pct")["delta_eur"]
        xs = np.arange(3) + (k - 1.5) * width
        ax.bar(xs, [s.get(v, np.nan) for v in [10, 20, 30]], width, label=f"{p} - {i}")
    ax.set_xticks(range(3), ["+10%", "+20%", "+30%"])
    ax.set_xlabel("Price shock")
    ax.set_ylabel("Extra material cost (EUR per box)")
    ax.set_title("Extra cost per box for copper and steel shocks")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir() / "shock_copper_steel.png", dpi=150)
    plt.close(fig)

    t30 = tab[tab["shock_pct"] == 30].set_index(["product", "item"])["delta_eur"]
    return {f"{p}_{i}_plus30_eur_per_box": round(float(t30.get((p, i), np.nan)), 1)
            for p in cfg["products"] for i in ["copper", "all_steel"]}


if __name__ == "__main__":
    print(run())
