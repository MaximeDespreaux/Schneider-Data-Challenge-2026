"""Une commande pour : criblage des signaux -> vérification 2024-26 du champion -> CSV de soumission.

    uv run python -m src.external.run_champion

Pré-requis : données du challenge dans data/raw/, données publiques téléchargées
(`uv run python -m src.external.fetch_public`), config/assumptions.yaml rempli en local.
Sorties (confidentielles, gitignorées) : data/outputs/external/ et data/outputs/testbench/.
"""

from __future__ import annotations

import pandas as pd

from src import data as d
from src.external.models import Champion
from src.external.screen import screen, selected
from src.external.signals import build_panel
from src.testbench import metrics
from src.testbench.backtest import run_backtest
from src.testbench.models import Naive, NaiveVolScaled
from src.testbench.run import DEFAULT_TARGETS, bom_deriver
from src.testbench.submission import from_forecasts, validate_submission

BOM_IDS = ["RM6_NE_IQI_433520", "RMAirSeT_IIQ_24kV_IEC"]
ORIGIN = "2026-09"


def main(team_id: str = "team_TBD"):
    out = d.OUT / "external"
    out.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    derive = bom_deriver(panel)

    print("1/3 criblage des signaux (cibles < 2024)")
    res = screen(panel)
    res.to_csv(out / "signal_screen.csv", index=False)
    sel = selected(res)
    champ = Champion(selected=sel)

    print("2/3 vérification sur 2024-26 (jamais utilisée pour choisir)")
    fc = run_backtest(panel, DEFAULT_TARGETS, [Naive(), NaiveVolScaled(), champ],
                      origins=pd.date_range("2024-01-01", f"{ORIGIN}-01", freq="MS"), derive=derive, verbose=False)
    fc.to_parquet(out / "champion_holdout.parquet")
    scored = metrics.score_rows(fc)
    tab = metrics.summarize(scored)
    tab.to_csv(out / "champion_holdout_scores.csv")
    agg = scored.groupby(["model", "horizon"]).agg(MAE=("ae", "mean"), QL=("ql", "mean"), coverage=("covered", "mean"))
    rel = (agg[["MAE", "QL"]] / agg.xs("naive", level="model")[["MAE", "QL"]]).join(agg["coverage"])
    print(rel.round(3).to_string())

    print("3/3 CSV de soumission")
    cat, _ = d.load_catalogs()
    fc0 = run_backtest(panel, DEFAULT_TARGETS, [champ], origins=[f"{ORIGIN}-01"], derive=derive, verbose=False)
    region = {s: ("EU" if s in ["S07", "S08", "S09", "S10", "S11"] else "WW") for s in d.RM_TARGETS}
    unit = {s: cat.loc[s, "unit"] for s in d.RM_TARGETS}
    ai = "claude-code; public data: FRED/IMF/BLS, Eurostat, NY Fed GSCPI"
    name = "champion_signals_ridge+naive_vol"
    a = from_forecasts(fc0[~fc0.series.isin(BOM_IDS)], ORIGIN, team_id, name, ai, region, unit)
    b = from_forecasts(fc0[fc0.series.isin(BOM_IDS)], ORIGIN, team_id, name, ai, {s: "EU" for s in BOM_IDS},
                       {s: f"index ({ORIGIN}=100)" for s in BOM_IDS}, target_type="bom_index")
    sub = pd.concat([a, b], ignore_index=True)
    problems = validate_submission(sub)
    path = d.OUT / "testbench" / f"draft_forecast_{ORIGIN.replace('-', '')}_champion.csv"
    sub.to_csv(path, index=False)
    print("problèmes de format :", problems or "aucun", "->", path)
    return rel, sub


if __name__ == "__main__":
    main()
