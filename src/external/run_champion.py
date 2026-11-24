"""Une commande pour : criblage des signaux -> vérification 2024-26 du champion -> CSV de soumission.

    uv run python -m src.external.run_champion                    # v1 (défaut, inchangé)
    uv run python -m src.external.run_champion --candidates v2    # option v2 : + séries énergie fournies (CROSS_V2)

Pré-requis : données du challenge dans data/raw/, données publiques téléchargées
(`uv run python -m src.external.fetch_public`), config/assumptions.yaml rempli en local.
Sorties (confidentielles, gitignorées) : data/outputs/external/ et data/outputs/testbench/.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from src import data as d
from src.external.features import CROSS_V2
from src.external.models import Champion
from src.external.screen import screen, selected
from src.external.signals import build_panel
from src.testbench import metrics
from src.testbench.backtest import compute_footprint, run_backtest
from src.testbench.driver_models import DriverRidge
from src.testbench.models import Naive, NaiveVolScaled
from src.testbench.run import bom_deriver, default_targets
from src.testbench.submission import from_forecasts, validate_submission

BOM_IDS = ["RM6_NE_IQI_433520", "RMAirSeT_IIQ_24kV_IEC"]
ORIGIN = "2026-09"


CANDIDATES = {"v1": {"cross": None, "model": "champion", "suffix": ""},
              "v2": {"cross": tuple(CROSS_V2), "model": "champion_v2", "suffix": "_v2"}}


TEAM_ID = "team_01"  # X-HEC DSAIB 2025-27, team 1


def main(team_id: str = TEAM_ID, candidates: str = "v1"):
    opt = CANDIDATES[candidates]
    sfx = opt["suffix"]
    t_start = time.perf_counter()
    out = d.OUT / "external"
    out.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    derive = bom_deriver(panel)
    targets = default_targets(panel)  # 14 séries Schneider + séries ajoutées présentes + 2 indices BOM

    print(f"1/3 criblage des signaux (cibles < 2024), candidats {candidates}")
    t0 = time.perf_counter()
    res = screen(panel, cross=opt["cross"])
    t_screen = time.perf_counter() - t0
    res.to_csv(out / f"signal_screen{sfx}.csv", index=False)
    sel = selected(res)
    champ = Champion(selected=sel, name=opt["model"], cross=opt["cross"])

    print("2/3 vérification sur 2024-26 (jamais utilisée pour choisir)")
    fc = run_backtest(panel, targets, [Naive(), NaiveVolScaled(), champ, DriverRidge()],
                      origins=pd.date_range("2024-01-01", f"{ORIGIN}-01", freq="MS"), derive=derive, verbose=False)
    runtime_holdout = dict(fc.attrs["runtime_s"])
    fc.to_parquet(out / f"champion_holdout{sfx}.parquet")
    scored = metrics.score_rows(fc)
    tab = metrics.summarize(scored)
    tab.to_csv(out / f"champion_holdout_scores{sfx}.csv")
    agg = scored.groupby(["model", "horizon"]).agg(MAE=("ae", "mean"), QL=("ql", "mean"), coverage=("covered", "mean"))
    rel = (agg[["MAE", "QL"]] / agg.xs("naive", level="model")[["MAE", "QL"]]).join(agg["coverage"])
    print(rel.round(3).to_string())

    print("3/3 CSV de soumission")
    cat, _ = d.load_catalogs()
    fc0 = run_backtest(panel, targets, [champ], origins=[f"{ORIGIN}-01"], derive=derive, verbose=False)
    runtime_sub = fc0.attrs["runtime_s"][champ.name]
    # le CSV ne contient que les 14 séries d'origine de Schneider + les 2 indices BOM
    region = {s: ("EU" if s in ["S07", "S08", "S09", "S10", "S11"] else "WW") for s in d.SCHNEIDER_RM_TARGETS}
    unit = {s: cat.loc[s, "unit"] for s in d.SCHNEIDER_RM_TARGETS}
    ai = "claude-code; public data: FRED/IMF/BLS, Eurostat, NY Fed GSCPI"
    name = f"{opt['model']}_signals_ridge+naive_vol"
    a = from_forecasts(fc0[fc0.series.isin(d.SCHNEIDER_RM_TARGETS)], ORIGIN, team_id, name, ai, region, unit)
    b = from_forecasts(fc0[fc0.series.isin(BOM_IDS)], ORIGIN, team_id, name, ai, {s: "EU" for s in BOM_IDS},
                       {s: f"index ({ORIGIN}=100)" for s in BOM_IDS}, target_type="bom_index")
    sub = pd.concat([a, b], ignore_index=True)
    problems = validate_submission(sub)
    path = d.OUT / "testbench" / f"draft_forecast_{ORIGIN.replace('-', '')}_champion{sfx}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    print("problèmes de format :", problems or "aucun", "->", path)

    runtimes = {"screening": t_screen, **{f"holdout_{k}": v for k, v in runtime_holdout.items()},
                f"submission_{champ.name}": runtime_sub}
    fp = compute_footprint(runtimes)
    print(f"\nCalcul ({candidates}) — temps des modèles, hors chargement des données :")
    print(fp.to_string())
    print(f"Temps total de la commande : {time.perf_counter() - t_start:.0f} s")
    return rel, sub


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="v1", choices=["v1", "v2"])
    main(candidates=ap.parse_args().candidates)
