"""Sanity check of DriverRidge with the new drivers (MEPS/CRU steel L01-L03, epoxy EP01, electronics EL01).

    uv run python -m src.checks.verify_drivers

THIS IS A SANITY CHECK, NOT MODEL SELECTION: it only shows that DriverRidge runs on the new series and gives ordered
quantiles. The official model comparison is Julia's holdout selection (choose on 2020-23, evaluate on 2024-26,
``metrics.holdout_selection`` in the full backtest).

Runs ONE backtest (naive, naive_vol, driver_ridge; origins 2022-01 -> 2026-09). Local only (needs data/raw/).
Output (confidential, gitignored): data/outputs/checks/verify_drivers_scores.csv.
"""

from __future__ import annotations

import pandas as pd

from src import data as d
from src.testbench import metrics
from src.testbench.backtest import compute_footprint, run_backtest
from src.testbench.driver_models import DriverRidge
from src.testbench.models import Naive, NaiveVolScaled
from src.testbench.run import bom_deriver, build_panel

ORIGIN = pd.Timestamp("2026-09-01")


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    panel = build_panel()
    targets = [s for s in ["L01", "L02", "L03", "EP01", "EL01", "S09", "S10"] if s in panel and panel[s].notna().any()] \
        + ["RM6_NE_IQI_433520", "RMAirSeT_IIQ_24kV_IEC"]
    print(f"Targets: {targets}")
    fc = run_backtest(panel, targets, [Naive(), NaiveVolScaled(), DriverRidge()],
                      origins=pd.date_range("2022-01-01", "2026-09-01", freq="MS"), derive=bom_deriver(panel),
                      verbose=False)

    scored = metrics.score_rows(fc)
    table = metrics.summarize(scored)
    cols = [c for c in ["n", "MAE_pct", "rel_MAE", "coverage", "DM_p"] if c in table.columns]
    view = table[cols].reorder_levels(["series", "horizon", "model"]).sort_index()
    print("\nScores (series x horizon x model; rel_MAE and DM_p vs naive):")
    print(view.to_string(float_format=lambda x: f"{x:.3f}"))
    print()
    print(compute_footprint(fc.attrs["runtime_s"]).to_string())

    out = d.OUT / "checks"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "verify_drivers_scores.csv")
    print(f"\n-> {out / 'verify_drivers_scores.csv'}")

    last = fc[(fc["origin"] == ORIGIN) & (fc["model"] == "driver_ridge")]
    for s in targets:
        g = last[last["series"] == s]
        assert {3, 6, 9} <= set(g["horizon"]), f"{s}: missing DriverRidge forecasts at h=3/6/9 for origin 2026-09"
        assert ((g["p10"] <= g["p50"]) & (g["p50"] <= g["p90"])).all(), f"{s}: quantiles not ordered"
    print("DRIVERS CHECK PASSED")


if __name__ == "__main__":
    main()
