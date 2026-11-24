"""Gap 3 — procurement levers: one table of numbers that already exist, for the procurement slide. No new modelling.

    uv run python -m src.business.clarifications.procurement_summary

Reads (re-running the producing module's ``run()`` if a file is missing):
- shock_table.csv, convergence_summary.csv (copper exposure to hedge);
- clarifications/lme_reference_summary.csv, lme_spread_stats.csv (hedge reference, back-to-back indexation);
- lag_table.csv (negotiation timing);
- supplier_estimates.csv (supplier engagement order).
Output (gitignored): data/outputs/business/clarifications/procurement_levers_summary.csv
(columns lever, metric, product_or_pair, value, unit, source_file).
"""

from __future__ import annotations

import importlib

import pandas as pd

from .. import paths
from ..config import load_business_config
from . import OUT

BUS_OUT = paths.OUT  # monkeypatched in the tests
OG_OUT = OUT
LAG_PAIRS = {"MEPS_CR_coil_to_EU_steel_PPI": "MEPS CR coil -> EU steel PPI",
             "MEPS_HDG_to_DE_steel_PPI": "MEPS HDG -> DE steel PPI",
             "aluminium_LME_to_EU_aluminium_PPI": "LME aluminium -> EU aluminium PPI",
             "nickel_LME_to_CRU_stainless_304": "LME nickel -> CRU stainless 304"}
HORIZONS = [3, 6, 9]


def sources() -> dict:
    """file name -> (path, module that produces it)."""
    return {"shock_table.csv": (BUS_OUT / "shock_table.csv", "shocks"),
            "convergence_summary.csv": (BUS_OUT / "convergence_summary.csv", "convergence"),
            "lag_table.csv": (BUS_OUT / "lag_table.csv", "lags"),
            "supplier_estimates.csv": (BUS_OUT / "supplier_estimates.csv", "suppliers"),
            "clarifications/lme_reference_summary.csv": (OG_OUT / "lme_reference_summary.csv",
                                                       "clarifications.lme_reference"),
            "clarifications/lme_spread_stats.csv": (OG_OUT / "lme_spread_stats.csv", "clarifications.lme_reference")}


def ensure_inputs(cfg: dict) -> dict:
    """Load every input; run the producing module once if its file is missing."""
    src, done, data = sources(), set(), {}
    for name, (path, mod) in src.items():
        if not path.exists() and mod not in done:
            print(f"    {path.name} missing: running src.business.{mod}")
            importlib.import_module(f"src.business.{mod}").run(cfg)
            done.add(mod)
        data[name] = pd.read_csv(path)
    return data


def build(data: dict, products: list) -> pd.DataFrame:
    rows = []

    def add(lever, metric, who, value, unit, source):
        rows.append({"lever": lever, "metric": metric, "product_or_pair": who, "value": value, "unit": unit,
                     "source_file": source})

    st = data["shock_table.csv"]
    cop = st[(st["item"] == "copper") & (st["kind"] == "material")]
    conv = data["convergence_summary.csv"].set_index("product")
    for p in products:
        for sh in [10, 30]:
            v = cop.loc[(cop["product"] == p) & (cop["shock_pct"] == sh), "delta_eur"]
            if not v.empty:
                add("copper exposure to hedge", f"extra cost for copper +{sh}%", p, float(v.iloc[0]),
                    "EUR per box", "shock_table.csv")
        if p in conv.index:
            add("copper exposure to hedge", "copper value share at 2026-09", p,
                100 * float(conv.at[p, "copper_share_2026_09"]), "%", "convergence_summary.csv")

    s = data["clarifications/lme_reference_summary.csv"]
    s = s[(s["metal"] == "copper") & (s["period"] == "all")].set_index(["horizon", "mechanism"])["p90_err_pct"]
    src = "clarifications/lme_reference_summary.csv"
    for h in HORIZONS:
        for m, label in [("B_fixed_3m", "P90 margin loss, price locked at LME 3-month (B_fixed_3m)"),
                         ("A_fixed_cash", "P90 margin loss, price fixed at LME cash (A_fixed_cash)")]:
            if (h, m) in s.index:
                add("hedge reference", label, f"copper h={h}", float(s[(h, m)]), "% of metal cost", src)
    sp = data["clarifications/lme_spread_stats.csv"].set_index("metal")
    if "copper" in sp.index:
        add("hedge reference", "mean (3m - cash) / cash spread", "copper", float(sp.at["copper", "mean_spread_pct"]),
            "%", "clarifications/lme_spread_stats.csv")
    for h in HORIZONS:
        if (h, "E_fixed_3m_plus_trigger") in s.index:
            add("back-to-back indexation", "P90 margin loss left with Schneider, 3m base + trigger clause "
                "(E_fixed_3m_plus_trigger)", f"copper h={h}", float(s[(h, "E_fixed_3m_plus_trigger")]),
                "% of metal cost", src)

    lag = data["lag_table.csv"]
    lag = lag[lag["sample"] == "full"].set_index("name")
    for key, label in LAG_PAIRS.items():
        if key in lag.index:
            for col in ["lag_50", "lag_80"]:
                add("negotiation timing", f"{col} (months until {col[-2:]}% of the move is passed through)", label,
                    float(lag.at[key, col]), "months", "lag_table.csv")

    sup = data["supplier_estimates.csv"].sort_values("priority_rank").head(3)
    for r in sup.itertuples():
        add("supplier engagement order", f"priority rank (confidence: {r.confidence})", r.name,
            float(r.priority_rank), "rank", "supplier_estimates.csv")
    return pd.DataFrame(rows, columns=["lever", "metric", "product_or_pair", "value", "unit", "source_file"])


def run(cfg: dict | None = None) -> dict:
    cfg = load_business_config() if cfg is None else cfg
    tab = build(ensure_inputs(cfg), cfg["products"])
    OG_OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OG_OUT / "procurement_levers_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        print(tab.round(2).to_string(index=False))
    head = {}
    for r in tab.itertuples():
        v = f"{r.value:.2f}" if r.unit != "rank" else f"{int(r.value)}"
        head[f"{r.lever}: {r.metric} ({r.product_or_pair})"] = f"{v} {r.unit}"  # no "|": REPORT.md table
    return head


if __name__ == "__main__":
    run()
