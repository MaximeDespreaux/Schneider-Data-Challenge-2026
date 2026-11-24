"""Gap 2 — how sensitive are the results to the unidentified BOM mass and to the proxies?

    uv run python -m src.business.clarifications.sensitivity

Per product (RM6, AirSeT), four numbers:
- ``bom_value_eur``     : priced BOM value per box at the base month (``bom_weights()``: kg × EUR/kg);
- ``copper_share``      : copper value share of the BOM, INCLUDING the copper contained in brass (so that a higher
                          brass copper share shows up as more copper exposure);
- ``p90_change_h9_pct`` : P90 cost change of the BOM at h = 9, sum over materials of w_i × (p90_i / last_i - 1), same
                          logic as ``electronics.py`` (brass = copper share × D01 change + zinc share × S05 change;
                          a material without forecast counts 0). Adding marginal P90s overstates the joint P90;
- ``margin_pts_lost``   : bom_value × P90 change / selling price × 100 (``bom_value`` cost basis).

Base case = current setup (BOM from config/assumptions.yaml, unidentified mass NOT priced). Scenarios, one at a time:
1. ``unidentified_priced``: unidentified kg = implied total kg - priced kg (``coverage_table()``), priced as each
   material of ``unidentified_price_as`` (its base-month EUR/kg and its forecast series);
2. ``brass_copper_share``: copper share of brass (rest zinc), for both the EUR value and the forecast change;
   zinc EUR/kg at base is backed out of the brass value in assumptions.yaml (brass = 0.63 copper + 0.37 zinc);
3. ``plastics_eur_per_kg``: plastics proxy price (forecast series unchanged, S11);
4. ``bom_version`` (setting "BOM v1 (data note)"): the previous BOM xlsx_v1 (if data/raw/RM_Prices_XHEC_Challenge.xlsx exists), plus
   bom_version_diff.csv (kg per product × material family in v1 vs v2). It is a data note, not a sensitivity: it
   stays in sensitivity.csv and the tornado but is excluded from the ``largest_swing`` headline.
Outputs in data/outputs/business/clarifications/ (gitignored) and figure sensitivity_tornado.png.
"""

from __future__ import annotations

import copy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import data as d  # noqa: E402

from ..config import load_business_config, selling_prices  # noqa: E402
from ..paths import fig_dir  # noqa: E402
from . import out_dir  # noqa: E402

H = 9
Q = "p90"
METRICS = ["bom_value_eur", "copper_share", "p90_change_h9_pct", "margin_pts_lost"]
BOM_V1_LABEL = "BOM v1 (data note)"  # data-quality check, not a sensitivity: kept out of the largest_swing headline
NOT_IN_SWING = {"bom_version"}


def material_params(m: str, ctx: dict, brass_share: float, plastics_eur: float | None = None) -> tuple:
    """(EUR/kg, copper EUR contained per kg, P90 change at h=9) of one material at the base month."""
    if m == "brass":
        s = brass_share
        return (s * ctx["eur"]["copper"] + (1 - s) * ctx["zinc_eur"], s * ctx["eur"]["copper"],
                s * ctx["change"]["copper"] + (1 - s) * ctx["zinc_change"])
    if m == "copper":
        return ctx["eur"]["copper"], ctx["eur"]["copper"], ctx["change"]["copper"]
    eur = plastics_eur if (m == "plastics" and plastics_eur is not None) else ctx["eur"][m]
    return eur, 0.0, ctx["change"].get(m, 0.0)


def build_lines(kg: pd.Series, ctx: dict, brass_share: float, plastics_eur: float | None = None,
                extra: tuple | None = None) -> pd.DataFrame:
    """One row per material of one product: kg, eur_per_kg, copper_eur_per_kg, change. ``extra`` = (material, kg)
    adds the unidentified mass priced as that material."""
    items = list(kg.items()) + ([(f"unidentified_as_{extra[0]}", extra[1])] if extra else [])
    rows = []
    for name, k in items:
        m = name.replace("unidentified_as_", "")
        eur, cu, ch = material_params(m, ctx, brass_share, plastics_eur)
        rows.append({"material": name, "kg": float(k), "eur_per_kg": eur, "copper_eur_per_kg": cu, "change": ch})
    return pd.DataFrame(rows)


def evaluate(lines: pd.DataFrame, price: float) -> dict:
    value = lines["kg"] * lines["eur_per_kg"]
    total = float(value.sum())
    p90 = float((value * lines["change"]).sum() / total)
    return {"bom_value_eur": total, "copper_share": float((lines["kg"] * lines["copper_eur_per_kg"]).sum() / total),
            "p90_change_h9_pct": 100 * p90, "margin_pts_lost": total * p90 / price * 100}


def context(a: dict, cfg: dict, fc: pd.DataFrame) -> dict:
    """Base-month EUR/kg (assumptions.yaml) and P90 change at h=9 (origin forecasts) of every material."""
    from src.testbench.targets import _series_for

    from ..forecasts import pct_change

    have = set(fc["series"])

    def change(series):
        return pct_change(fc, series, H, Q) if series in have else 0.0

    eur = {k: float(v["value"]) for k, v in a["eur_per_kg_at_base"].items()}
    mix = cfg["competitor_lever"]["brass_mix"]
    ch = {k: change(_series_for(a, k)) for k in eur if k != "brass"}
    ch["copper"] = change("D01")
    return {"eur": eur, "change": ch, "brass_share_base": float(mix["copper"]),
            "zinc_eur": (eur["brass"] - mix["copper"] * eur["copper"]) / mix["zinc"], "zinc_change": change("S05")}


def unidentified_kg(a: dict) -> pd.Series:
    """Implied total kg - priced kg per product, for the BOM version in ``a``."""
    from src.testbench.targets import coverage_table

    cov = coverage_table(a).set_index("product")
    return (cov["implied_total_kg"] - cov["priced_kg"]).clip(lower=0)


def scenarios(kg_by_prod: dict, unid: pd.Series, ctx: dict, prices: dict, sens: dict,
              kg_v1: dict | None = None) -> pd.DataFrame:
    """Long table: base case and every scenario setting, with deltas vs base."""
    b0 = ctx["brass_share_base"]
    rows = []
    for prod, kg in kg_by_prod.items():
        price = float(prices[prod])
        base = evaluate(build_lines(kg, ctx, b0), price)
        runs = [("base", "current setup", base)]
        for lvl, m in sens["unidentified_price_as"].items():
            runs.append(("unidentified_priced", f"{lvl} ({m})",
                         evaluate(build_lines(kg, ctx, b0, extra=(m, float(unid[prod]))), price)))
        for s in sens["brass_copper_share"]:
            runs.append(("brass_copper_share", f"{s:.2f}", evaluate(build_lines(kg, ctx, float(s)), price)))
        for p in sens["plastics_eur_per_kg"]:
            runs.append(("plastics_eur_per_kg", f"{p:.2f}",
                         evaluate(build_lines(kg, ctx, b0, plastics_eur=float(p)), price)))
        if kg_v1 is not None and prod in kg_v1:
            runs.append(("bom_version", BOM_V1_LABEL, evaluate(build_lines(kg_v1[prod], ctx, b0), price)))
        for scen, setting, res in runs:
            rows.append({"product": prod, "scenario": scen, "setting": setting,
                         "unidentified_kg": float(unid[prod]), **res,
                         **{f"delta_{k}": res[k] - base[k] for k in METRICS}})
    return pd.DataFrame(rows)


def bom_version_diff(rm6_carbon_share: float) -> pd.DataFrame:
    """kg per product × material family in BOM v1 (previous) vs v2 (official), with implied totals."""
    v1 = d.load_bom("root", rm6_carbon_share=rm6_carbon_share)
    v2 = d.load_bom("v2", rm6_carbon_share=rm6_carbon_share)
    frames = []
    for tag, b in {"kg_v1": v1, "kg_v2": v2}.items():
        k = b.groupby(["product", "material_family"])["mass_kg"].sum()
        tot = b.groupby("product")["implied_total_kg"].first()
        unl = tot - b.groupby("product")["mass_kg"].sum()
        extra = pd.concat([tot.to_frame("mass_kg").assign(material_family="TOTAL (implied)"),
                           unl.clip(lower=0).to_frame("mass_kg").assign(material_family="unlabelled remainder")])
        k = pd.concat([k, extra.reset_index().set_index(["product", "material_family"])["mass_kg"]])
        frames.append(k.rename(tag))
    out = pd.concat(frames, axis=1).fillna(0.0).reset_index()
    out["diff_kg_v2_minus_v1"] = out["kg_v2"] - out["kg_v1"]
    out["note"] = np.where((out["product"] == "RM6") & (out["material_family"] == "Carbon steel"),
                           f"v1 RM6 carbon steel = assumed {rm6_carbon_share:.0%} of implied total (unlabelled slice)",
                           "")
    out["_last"] = out["material_family"].isin(["TOTAL (implied)", "unlabelled remainder"])
    return out.sort_values(["product", "_last", "material_family"]).drop(columns="_last").reset_index(drop=True)


def _figure(tab: pd.DataFrame, products: list) -> None:
    fig, axes = plt.subplots(1, len(products), figsize=(6 * len(products), 4.5), squeeze=False)
    for ax, prod in zip(axes[0], products):
        g = tab[tab["product"] == prod]
        base = float(g.loc[g["scenario"] == "base", "margin_pts_lost"].iloc[0])
        scen = [s for s in g["scenario"].unique() if s != "base"]
        for i, s in enumerate(scen):
            v = g.loc[g["scenario"] == s, "margin_pts_lost"]
            lo, hi = float(v.min()), float(v.max())
            ax.barh(i, lo - base, left=base, color="tab:blue", alpha=0.8, label="low setting" if i == 0 else None)
            ax.barh(i, hi - base, left=base, color="tab:orange", alpha=0.8, label="high setting" if i == 0 else None)
        ax.axvline(base, color="black", lw=1)
        ax.set_yticks(range(len(scen)), [BOM_V1_LABEL if s == "bom_version" else s for s in scen])
        ax.set_xlabel("Margin points lost at P90, h = 9 (bom_value basis)")
        ax.set_title(f"{prod} (base = {base:.2f} pts)")
        ax.grid(axis="x", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Sensitivity to unidentified mass and proxies (one scenario at a time)")
    fig.tight_layout()
    fig.savefig(fig_dir() / "sensitivity_tornado.png", dpi=150)
    plt.close(fig)


def run(cfg: dict | None = None) -> dict:
    from src.testbench.targets import bom_weights, load_assumptions

    from ..forecasts import get_origin_forecasts

    cfg = load_business_config() if cfg is None else cfg
    sens = cfg["clarifications"]["sensitivity"]
    a = load_assumptions()
    prices = selling_prices(a, cfg)
    ctx = context(a, cfg, get_origin_forecasts(cfg))
    w = bom_weights(a)
    kg = {p: w[w["product"] == p].set_index("material")["kg"] for p in cfg["products"]}
    unid = unidentified_kg(a)
    print(f"    BOM {a['bom']['version']}: unidentified kg priced in scenario 1: "
          + ", ".join(f"{p} {unid[p]:.1f} kg" for p in cfg["products"]))

    kg_v1, out = None, out_dir()
    if (d.RAW / "RM_Prices_XHEC_Challenge.xlsx").exists():
        a1 = copy.deepcopy(a)
        a1["bom"]["version"] = "xlsx_v1"
        w1 = bom_weights(a1)
        kg_v1 = {p: w1[w1["product"] == p].set_index("material")["kg"] for p in cfg["products"]}
        bom_version_diff(float(a["bom"]["rm6_big_unlabelled_share"])).to_csv(out / "bom_version_diff.csv",
                                                                             index=False)
    else:
        print("    previous BOM (RM_Prices_XHEC_Challenge.xlsx) not found: bom_version scenario skipped")

    tab = scenarios(kg, unid, ctx, prices, sens, kg_v1)
    tab.to_csv(out / "sensitivity.csv", index=False)
    _figure(tab, cfg["products"])

    head = {}
    for prod, g in tab.groupby("product", sort=False):
        base = float(g.loc[g["scenario"] == "base", "margin_pts_lost"].iloc[0])
        sw = {s: float(max(v.max(), base) - min(v.min(), base))
              for s, v in g[~g["scenario"].isin({"base"} | NOT_IN_SWING)].groupby("scenario")["margin_pts_lost"]}
        top = max(sw, key=sw.get)
        head[f"{prod}_unidentified_kg"] = round(float(unid[prod]), 1)
        head[f"{prod}_base_margin_pts_lost_p90_h9"] = round(base, 2)
        head[f"{prod}_largest_swing"] = f"{top}: {sw[top]:.2f} margin pts"
    return head


if __name__ == "__main__":
    for k, v in run().items():
        print(f"{k}: {v}")
