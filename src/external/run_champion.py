"""Une commande pour : criblage des signaux (exploration) -> vérification 2024-26 du champion -> CSV de soumission.

Le champion noté et soumis est ``ChampionWF`` : criblage refait à chaque origine sur les seules cibles réalisées.
Le criblage global (``screen``) n'est écrit que pour l'exploration. Les séries absentes au bord du live (origine
2026-09) sont masquées du même nombre de mois à chaque origine, comme dans ``testbench.pipeline``.

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
from src.external.models import ChampionWF
from src.external.screen import screen
from src.external.signals import build_panel
from src.testbench import metrics
from src.testbench.backtest import compute_footprint, run_backtest
from src.testbench.driver_models import DriverRidge
from src.testbench.ml_models import LGBMQuantile
from src.testbench.models import Naive, NaiveVolScaled
from src.testbench.pipeline import lags_for
from src.testbench.run import bom_deriver, default_targets
from src.testbench.submission import from_forecasts, validate_submission

BOM_IDS = ["RM6_NE_IQI_433520", "RMAirSeT_IIQ_24kV_IEC"]
ORIGIN = "2026-09"


CANDIDATES = {"v1": {"cross": None, "model": "champion", "suffix": ""},
              "v2": {"cross": tuple(CROSS_V2), "model": "champion_v2", "suffix": "_v2"}}


TEAM_ID = "team_01"  # X-HEC DSAIB 2025-27, team 1


def ml_replaces_champion(tab: pd.DataFrame, champion: str, ml: str, tol: float = 0.02) -> dict:
    """Règle pré-enregistrée (identique à la règle champion v1/v2) sur la vérification 2024-26 :
    adopter le modèle ML seulement si sa rel_QL médiane sur (série, horizon) est plus basse que celle du champion
    ET sa couverture médiane n'est pas inférieure à celle du champion moins ``tol``."""
    t = tab.reset_index()
    med = t[t.model.isin([champion, ml])].groupby("model")[["rel_QL", "coverage"]].median()
    q_c, q_m = med.at[champion, "rel_QL"], med.at[ml, "rel_QL"]
    c_c, c_m = med.at[champion, "coverage"], med.at[ml, "coverage"]
    adopt = bool(q_m < q_c and c_m >= c_c - tol)
    return {"adopt": adopt, "verdict": f"ADOPT {ml}" if adopt else f"KEEP {champion}",
            "detail": f"rel_QL médiane {ml} {q_m:.3f} vs {champion} {q_c:.3f} ; couverture {c_m:.3f} vs {c_c:.3f}",
            "ml_rel_QL": q_m, "champion_rel_QL": q_c, "ml_coverage": c_m, "champion_coverage": c_c}


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
    champ = ChampionWF(name=opt["model"], cross=opt["cross"])
    lags = lags_for("submission", panel, targets, f"{ORIGIN}-01")  # bord irrégulier du live, appliqué à chaque origine

    print("2/3 vérification sur 2024-26 (sert aussi aux règles v1/v2 et champion/LightGBM : ce n'est pas un test vierge)")
    ml = LGBMQuantile()
    fc = run_backtest(panel, targets, [Naive(), NaiveVolScaled(), champ, DriverRidge(), ml],
                      origins=pd.date_range("2024-01-01", f"{ORIGIN}-01", freq="MS"), derive=derive, lags=lags, verbose=False)
    runtime_holdout = dict(fc.attrs["runtime_s"])
    scored = metrics.score_rows(fc)
    tab = metrics.summarize(scored)
    tab.to_csv(out / f"champion_holdout_scores{sfx}.csv")
    agg = scored.groupby(["model", "horizon"]).agg(MAE=("ae", "mean"), QL=("ql", "mean"), coverage=("covered", "mean"))
    rel = (agg[["MAE", "QL"]] / agg.xs("naive", level="model")[["MAE", "QL"]]).join(agg["coverage"])
    print(rel.round(3).to_string())

    # Ljung-Box sur les erreurs (origines non chevauchantes) : reste-t-il de la structure exploitable ?
    lb = metrics.ljung_box_table(scored, [champ.name, "naive_vol", ml.name])
    lb.to_csv(out / f"champion_ljungbox{sfx}.csv", index=False)
    print("\nLjung-Box (lag 1, origines non chevauchantes) : part des séries avec p < 0,05")
    print(lb.groupby(["model", "horizon"])["p_value"].apply(lambda p: (p < 0.05).mean()).round(2).to_string())

    # Règle FIXÉE AVANT de voir les résultats (même règle que champion v1/v2) : le modèle ML remplace le champion
    # seulement si sa perte quantile relative médiane est plus basse ET sa couverture médiane >= celle du champion - 0,02
    decision = ml_replaces_champion(tab, champ.name, ml.name)
    pd.DataFrame([decision]).to_csv(out / f"champion_vs_{ml.name}{sfx}.csv", index=False)
    print(f"\nRègle de remplacement : {decision['verdict']} ({decision['detail']})")
    submitted = ml if decision["adopt"] else champ
    if decision["adopt"]:
        # le business lit les lignes « champion » : on y met le modèle retenu pour que tout suive automatiquement
        fc.loc[fc.model == champ.name, "model"] = f"{champ.name}_signals"
        fc.loc[fc.model == ml.name, "model"] = champ.name
    fc.to_parquet(out / f"champion_holdout{sfx}.parquet")

    print("3/3 CSV de soumission")
    cat, _ = d.load_catalogs()
    fc0 = run_backtest(panel, targets, [submitted], origins=[f"{ORIGIN}-01"], derive=derive, lags=lags, verbose=False)
    runtime_sub = fc0.attrs["runtime_s"][submitted.name]
    # le CSV ne contient que les 14 séries d'origine de Schneider + les 2 indices BOM
    region = {s: ("EU" if s in ["S07", "S08", "S09", "S10", "S11"] else "WW") for s in d.SCHNEIDER_RM_TARGETS}
    unit = {s: cat.loc[s, "unit"] for s in d.SCHNEIDER_RM_TARGETS}
    ai = "claude-code; public data: FRED/IMF/BLS, Eurostat, NY Fed GSCPI"
    name = ("lgbm_quantile (adopted by pre-registered rule)" if submitted is ml
            else f"{opt['model']}_wf_signals_ridge+naive_vol")
    if submitted is ml:
        ai += "; lightgbm"
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
                f"submission_{submitted.name}": runtime_sub}
    fp = compute_footprint(runtimes)
    print(f"\nCalcul ({candidates}) — temps des modèles, hors chargement des données :")
    print(fp.to_string())
    print(f"Temps total de la commande : {time.perf_counter() - t_start:.0f} s")
    return rel, sub


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="v1", choices=["v1", "v2"])
    main(candidates=ap.parse_args().candidates)
