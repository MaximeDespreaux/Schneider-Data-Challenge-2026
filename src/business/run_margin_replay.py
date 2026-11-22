"""Margin-protection replay on real data (Goal A core result).

    uv run python -m src.business.run_margin_replay            # base pass on the BOM indices + meta pass + replay
    uv run python -m src.business.run_margin_replay --cached   # reuse data/outputs/margin/base_bom.parquet

Steps
1. Walk-forward base pass on the two BOM cost indices (origins 2010-01 .. 2026-09, submission mode with the live
   ragged edge, causal BOM weights), then the Strategy A meta pass (wf_blend = submitted model).
2. One quote per month and horizon: realised cost change of the basket, forecast P50/P90, historical P90 (naive),
   copper share and copper forward at the quote date.
3. Policies scored on the DEV window (origins 2010-01..2021-12, targets < 2022-01) and, separately, on the
   GRADED window (origins >= 2022-01). The rule (lowest markup with breach rate <= 5% and 5%-worst margin >=
   floor) picks the clause threshold on DEV only; GRADED is reported as is.
4. Sensitivity: material share of the selling price {0.20 (config), 0.40, 0.68 (teardown: material ~80% of
   should-cost)}.
Outputs (confidential, gitignored): data/outputs/margin/.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src import data as d
from src.business.margin_replay import DEFAULT_POLICIES, Economics, build_quotes, choose_policy, summarize_policies
from src.external.signals import build_panel
from src.testbench.availability import ragged_edge_lags
from src.testbench.backtest import run_backtest
from src.testbench.driver_models import DriverRidge
from src.testbench.meta import MetaConfig, meta_pass
from src.testbench.models import (AR1Point, Conformal, DriftPoint, MeanReversionPoint, Naive, NaiveGaussian,
                                  NaiveVolScaled)
from src.testbench.run import DEFAULT_TARGETS, bom_deriver
from src.testbench.targets import TARGET_IDS, load_assumptions

OUT = d.OUT / "margin"
PURGE = pd.Timestamp("2022-01-01")
PRODUCT_OF = {v: k for k, v in TARGET_IDS.items()}
SHARES = (0.20, 0.40, 0.68)


def candidates() -> list:
    return [Naive(), NaiveGaussian(), NaiveVolScaled(halflife=3, name="naive_vol_hl3"),
            NaiveVolScaled(halflife=6, name="naive_vol_hl6"), NaiveVolScaled(halflife=12, name="naive_vol_hl12"),
            Conformal(DriftPoint()), Conformal(AR1Point()), Conformal(MeanReversionPoint()), DriverRidge()]


def economics(a: dict, product: str, share: float | None = None) -> Economics:
    pr = a["pricing"]
    key = product.lower()
    return Economics(material_share=float(share or pr["material_share_of_price"]["value"]),
                     target_margin=float(pr[f"profit_margin_{key}"]["value"]),
                     floor=float(pr["margin_floor"][key]["value"]))


def base_forecasts(panel, der, cached: bool) -> pd.DataFrame:
    path = OUT / "base_bom.parquet"
    if cached and path.exists():
        return pd.read_parquet(path)
    exempt = set(DEFAULT_TARGETS) | {"L01", "L02", "L03", "L01_FR", "EP01", "BR01", "EL01"}
    edge = ragged_edge_lags(panel, "2026-09-01", exempt=exempt)
    fc = run_backtest(panel, list(TARGET_IDS.values()), candidates(),
                      origins=pd.date_range("2010-01-01", "2026-09-01", freq="MS"), derive=der, lags=edge, verbose=False)
    meta, diag = meta_pass(fc, MetaConfig(), "submission")
    fc = pd.concat([fc, meta], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(path)
    diag.to_parquet(OUT / "meta_diag_bom.parquet")
    return fc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true")
    args = ap.parse_args(argv)
    a = load_assumptions()
    panel = build_panel()
    der = bom_deriver(panel)
    fc = base_forecasts(panel, der, args.cached)

    def w_cu(series, origin):
        w = der.weights_at(panel.loc[:origin], origin)
        w = w[(w["product"] == PRODUCT_OF[series]) & (w["material"] == "copper")]
        return float(w["weight"].sum())

    tables, picks = [], []
    for series, product in PRODUCT_OF.items():
        for model in ("wf_blend", "naive_vol_hl3"):
            q = build_quotes(fc, panel, series, model, w_cu)
            for share in SHARES:
                eco = economics(a, product, share)
                for h, qh in q.groupby("horizon"):
                    dev = qh[(qh.origin < PURGE) & (qh.origin + pd.DateOffset(months=int(h)) < PURGE)]
                    grd = qh[qh.origin >= PURGE]
                    dev_tab = None
                    for window, part in (("dev", dev), ("graded", grd)):
                        if part.empty:
                            continue
                        t = summarize_policies(part, eco, DEFAULT_POLICIES)
                        dev_tab = t if window == "dev" else dev_tab
                        t = t.assign(product=product, model=model, material_share=share, horizon=h, window=window)
                        tables.append(t.rename_axis("policy").reset_index())
                    if dev_tab is not None:  # the rule only ever looks at DEV
                        picks.append({"product": product, "model": model, "material_share": share, "horizon": h,
                                      "rule_pick_dev": choose_policy(dev_tab, max_breach=0.05, floor=eco.floor)})
    res = pd.concat(tables, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "policy_scores.csv", index=False)
    pk = pd.DataFrame(picks)
    pk.to_csv(OUT / "policy_picks.csv", index=False)
    show = res[(res.model == "wf_blend") & (res.material_share == SHARES[0])]
    pd.set_option("display.width", 200)
    print(show.pivot_table(index=["product", "horizon", "policy"], columns="window",
                           values=["breach_rate", "mean_markup", "p5_margin"]).round(3).to_string())
    print(pk[pk.model == "wf_blend"].to_string(index=False))
    print(f"-> {OUT}")
    return res, pk


if __name__ == "__main__":
    main()
