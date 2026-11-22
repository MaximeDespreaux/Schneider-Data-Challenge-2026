"""Build the NDA-safe figure set for the public reports.

    uv run python public/reports/make_public_figures.py

- Figures that only show public data, rebased indices (=100) or statistics are copied from the private
  ``reports/figures/`` (re-run the private notebooks first so they are current).
- Figures that showed licensed EUR/t levels or Schneider € figures are redrawn here as % changes or indices.
- Teardown should-cost and supplier revenue charts are not published at all.
Every output name is on ``ALLOWED_FIGS`` in ``public/scan_leaks.py``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.data_loading as dl  # noqa: E402

PROC = ROOT / "data" / "processed"
PRIVATE_FIGS = ROOT / "reports" / "figures"
OUT = Path(__file__).resolve().parent / "figures"
BOM_V1 = ROOT / "data" / "RM_Prices_XHEC_Challenge_v1.xlsx"

COPY_AS_IS = [
    "00_rm_rebased", "00_rm_missingness", "00_energy_shock", "00_lme_spread", "01_missingness", "02_rm_indexed",
    "03_rolling_vol", "04_momentum", "05_corr_returns", "06_energy_ccf", "07_steel_market_vs_proxy",
    "11_baseline_mape", "12_leaderboard_heatmap", "13_interval_coverage", "14_dynreg_s09_coefs", "17_corr_levels",
    "18_corr_dlog_pearson", "18b_corr_dlog_spearman", "19_corr_family_blocks", "20_rolling_corr",
    "21_leadlag_ccf", "22_feature_target_rho", "23_feature_selection",
]

# Same chart system as the notebooks.
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
mpl.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.prop_cycle": mpl.cycler(color=PAL), "lines.linewidth": 1.6, "legend.frameon": False,
    "legend.fontsize": 8, "font.family": "sans-serif", "figure.dpi": 110,
})


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("drawn ", name)


# ------------------------------------------------------------------------------------------- BOM cost index
PLASTIC_EUR_T_2021 = 2500  # our assumption (no EUR/t plastics series), moved with the S11 PPI
BRASS_CU, BRASS_ZN = 0.65, 0.35  # brass proxy: typical CuZn35 composition


def material_prices(P: pd.DataFrame) -> pd.DataFrame:
    """EUR/t price per BOM material family (private levels: used only to build indices, never plotted)."""
    plast = PLASTIC_EUR_T_2021 * P.S11 / P.S11.loc["2021"].mean()
    zinc_eur = P.S05 / P.S06
    return pd.DataFrame({
        "Carbon steel": P.CRU_CR_DE, "Galvanised steel": P.CRU_HDG_DE, "Stainless steel": P.CRU_SS304_DE,
        "Copper": P.D01, "Aluminium": P.D02, "Epoxy": P.EPOXY_RESIN, "Other plastics": plast,
        "Brass": BRASS_CU * P.D01 + BRASS_ZN * zinc_eur,
    })


def bom_v1_masses() -> pd.DataFrame:
    """Latest BOM (v1): mass per product × priceable material family. 'Other' / unidentified are excluded."""
    b = pd.read_excel(BOM_V1, sheet_name="BOM_Materials")
    b = b[b.plant_code.notna() & b.mass_kg.notna()].copy()
    fam = b.material_family.replace({"Engineering plastic": "Other plastics", "Plastic": "Other plastics"})
    fam[b.material_label_source.str.contains("Epoxy", case=False)] = "Epoxy"
    b["family"] = fam
    b["product"] = np.where(b.product_ref.str.startswith("RM6"), "RM6 NE-IQI", "RM AirSeT")
    b = b[~b.family.isin(["Other", "To confirm"])]
    return b.groupby(["product", "family"]).mass_kg.sum().unstack(fill_value=0.0)


def bom_cost(P: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.Timestamp]:
    """Cost-weighted material cost (price × mass) per product and family, and the base month."""
    prices = material_prices(P)
    masses = bom_v1_masses()
    prices = prices[masses.columns].dropna(how="all").ffill()
    base = prices.dropna().index.max()
    costs = {prod: prices.loc[:base].mul(masses.loc[prod] / 1000, axis=1).dropna() for prod in masses.index}
    return costs, base


def fig_bom_cost_index(P: pd.DataFrame) -> None:
    costs, base = bom_cost(P)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    for i, (prod, c) in enumerate(costs.items()):
        tot = c.sum(axis=1).loc["2017":]
        a1.plot(tot.index, 100 * tot / tot.loc[base], color=PAL[i], label=prod)
    a1.axhline(100, color=AXIS, lw=0.8)
    a1.set_title(f"Priced material cost index, cost-weighted (latest BOM), {base:%b-%Y} = 100")
    a1.legend(loc="upper left")
    fams = list(next(iter(costs.values())).columns)
    left = np.zeros(len(costs))
    for j, f in enumerate(fams):
        sh = np.array([100 * c.loc[base, f] / c.loc[base].sum() for c in costs.values()])
        a2.barh(list(costs), sh, left=left, color=PAL[j % len(PAL)], label=f, height=0.55)
        left += sh
    a2.set_xlim(0, 100); a2.set_xlabel("% of priced material cost"); a2.grid(axis="x"); a2.grid(axis="y", visible=False)
    a2.set_title(f"Cost shares by material, {base:%b-%Y}")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4)
    fig.tight_layout(); save(fig, "08_bom_cost_index")


def fig_price_vs_cost_index(P: pd.DataFrame) -> None:
    costs, _ = bom_cost(P)
    rm6 = costs["RM6 NE-IQI"].sum(axis=1)
    cost_y = rm6.groupby(rm6.index.year).mean()
    pb = pd.read_parquet(PROC / "price_benchmark.parquet")
    price_y = pb[pb.brand.str.contains("Schneider")].set_index("year").p50.sort_index()
    yrs = [y for y in price_y.index if y in cost_y.index and y >= 2017]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(yrs, 100 * price_y.loc[yrs] / price_y.loc[yrs[0]], marker="o", color=PAL[0], label="Schneider selling price (P50, synthetic)")
    ax.plot(yrs, 100 * cost_y.loc[yrs] / cost_y.loc[yrs[0]], marker="o", color=PAL[1], label="RM6 priced material cost (latest BOM)")
    ax.axhline(100, color=AXIS, lw=0.8)
    ax.set_title(f"Price vs material cost, yearly means, {yrs[0]} = 100"); ax.legend(loc="upper left")
    fig.tight_layout(); save(fig, "16_price_vs_cost_index")


# ------------------------------------------------------------------------------------------- other redraws
def fig_stainless_alloy_share(P: pd.DataFrame) -> None:
    s = (100 * P.CRU_SS304_DE_ALLOY / (P.CRU_SS304_DE_BASE + P.CRU_SS304_DE_ALLOY)).dropna()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.fill_between(s.index, s, color=PAL[3], alpha=0.8, lw=0)
    ax.set_ylim(0, 100); ax.set_ylabel("% of total 304 price")
    ax.set_title("Stainless 304 Germany: alloy surcharge as a share of the transaction price")
    fig.tight_layout(); save(fig, "00_stainless_alloy_share")


def fig_epoxy_typo_pct() -> None:
    ep = dl.load_epoxy()[["Epychlorhydrine", "Epoxy_resin"]].rename(columns={"Epychlorhydrine": "ECH", "Epoxy_resin": "Epoxy resin"})
    ch = 100 * ep.pct_change().loc["2024":]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    w = 12
    for i, c in enumerate(ch.columns):
        ax.bar(ch.index + pd.Timedelta(days=(i - 0.5) * w), ch[c], width=w, color=PAL[i], label=c)
    ax.set_yscale("symlog", linthresh=10); ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel("monthly change, % (symlog)")
    ax.set_title("Epoxy chain, monthly % change: the May-2026 ECH data-entry error (≈ −90 % then ≈ +900 %)")
    ax.legend(loc="upper left"); fig.tight_layout(); save(fig, "00_epoxy_typo_pct")


def fig_forecast_fans_pct() -> None:
    fc = pd.read_parquet(PROC / "forecast_fans.parquet")
    T = pd.read_parquet(PROC / "panel_monthly_filled.parquet")
    show = [("D01", "Copper"), ("CRU_CR_DE", "CR coil DE"), ("CRU_SS304_DE", "Stainless 304 DE"), ("S09", "EU steel PPI")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), sharey=False)
    for ax, (s, lab) in zip(axes, show):
        f = fc[fc.series == s].sort_values("h")
        y0, t0 = f.y0.iat[0], f.origin.iat[0]
        hist = T[s].loc["2023-01":t0].dropna()
        pct = lambda v: 100 * (np.asarray(v, float) / y0 - 1)  # noqa: E731
        xs = [t0] + list(f.target_month)
        ax.fill_between(xs, [0] + list(pct(f.p10)), [0] + list(pct(f.p90)), color=PAL[0], alpha=0.18, lw=0, label="P10–P90")
        ax.plot(hist.index, pct(hist), color=INK2, lw=1.3, label="history")
        ax.plot(xs, [0] + list(pct(f.p50)), color=PAL[0], lw=1.8, marker="o", ms=4, label="ensemble P50")
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_title(f"{lab} (% vs latest)", fontsize=9.5); ax.tick_params(axis="x", rotation=30)
    axes[0].legend(loc="upper left")
    fig.suptitle("Forecast fans from the latest observation, % change (h = 3, 6, 9 months)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); save(fig, "15_forecast_fans_pct")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    for name in COPY_AS_IS:
        shutil.copy2(PRIVATE_FIGS / f"{name}.png", OUT / f"{name}.png")
        print("copied", name)
    P = pd.read_parquet(PROC / "panel_monthly_filled.parquet")
    fig_stainless_alloy_share(P)
    fig_epoxy_typo_pct()
    fig_bom_cost_index(P)
    fig_forecast_fans_pct()
    fig_price_vs_cost_index(P)
