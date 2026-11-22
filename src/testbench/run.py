"""Point d'entrée du banc de test.

    uv run python -m src.testbench.run                     # baselines, mode submission, origines 2020-01 →
    uv run python -m src.testbench.run --mode realtime     # avec délais de publication
    uv run python -m src.testbench.run --start 2017-11     # échantillon complet

Sorties (CONFIDENTIELLES, gitignorées) dans data/outputs/testbench/<mode>/ :
    forecasts.parquet  toutes les prévisions (format long)
    scores.csv         scores par modèle × série × horizon
    overview.csv       vue d'ensemble par modèle × horizon
    regimes.csv        scores séparés choc 2021-22 / période calme
"""

from __future__ import annotations

import argparse
import warnings

import pandas as pd

from src import data as d

from . import metrics
from .backtest import compute_footprint, default_origins, run_backtest
from .models import baseline_models


def build_panel(with_bom: bool = False):
    """Panel matières + énergie. Les indices BOM ne sont PAS ajoutés ici par défaut : dans le backtest ils sont
    reconstruits à chaque origine (``bom_deriver``) pour ne jamais utiliser des poids du futur."""
    rm = d.load_rm_wide()
    en = d.load_energy_wide()
    panel = rm.join(en, how="left")
    # laiton dérivé (EUR/t) : ≈ 63 % cuivre + 37 % zinc (S05 en USD/t, S06 = USD par EUR). Pas une cible.
    panel["BR01"] = 0.63 * panel["D01"] + 0.37 * panel["S05"] / panel["S06"]
    try:  # acier sous licence MEPS/CRU -> colonnes L01-L03 (vides dans le panel du challenge)
        steel = d.load_licensed_steel()
        for c in ["L01", "L02", "L03"]:
            panel[c] = steel[c].reindex(panel.index)
    except FileNotFoundError as e:
        warnings.warn(f"{e} -> l'acier retombe sur les indices Eurostat (S09/S10).", stacklevel=2)
    try:
        panel["EP01"] = d.load_epoxy_series().reindex(panel.index)
    except FileNotFoundError as e:
        warnings.warn(f"{e} -> pas de série epoxy EP01.", stacklevel=2)
    ppi = d.RAW / "external" / "eurostat_ppi_long.csv"
    eu = pd.read_csv(ppi, index_col=0, parse_dates=True) if ppi.exists() else pd.DataFrame()
    for c in ["EL01", "EL02", "EL03"]:
        if c in eu:
            panel[c] = eu[c].reindex(panel.index)
    if "EL01" not in eu:
        warnings.warn("EL01 absent : run `uv run python -m src.external.fetch_public` to download the electronics "
                      "price index", stacklevel=2)
    if with_bom:
        from .targets import add_bom_indices
        panel, _ = add_bom_indices(panel)
    return panel


def bom_deriver(panel):
    from .targets import make_bom_deriver
    return make_bom_deriver(panel)


# constante rétro-compatible : les 14 séries de Schneider + les 2 indices BOM
DEFAULT_TARGETS = d.SCHNEIDER_RM_TARGETS + ["RM6_NE_IQI_433520", "RMAirSeT_IIQ_24kV_IEC"]


def default_targets(panel) -> list[str]:
    """Cibles par défaut : seulement les séries présentes et non vides dans le panel, + les 2 indices BOM."""
    return [s for s in d.RM_TARGETS if s in panel and panel[s].notna().any()] + ["RM6_NE_IQI_433520",
                                                                                   "RMAirSeT_IIQ_24kV_IEC"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="submission", choices=["submission", "realtime"])
    ap.add_argument("--start", default="2020-01")
    ap.add_argument("--targets", default=None)
    args = ap.parse_args(argv)

    panel = build_panel()
    targets = args.targets.split(",") if args.targets else default_targets(panel)
    models = baseline_models()
    print(f"Banc de test : {len(models)} modèles × {len(targets)} séries, mode={args.mode}, origines dès {args.start}")
    fc = run_backtest(panel, targets, models, origins=default_origins(panel, args.start), mode=args.mode,
                      derive=bom_deriver(panel))

    out = d.OUT / "testbench" / args.mode
    out.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(out / "forecasts.parquet")
    scored = metrics.score_rows(fc)
    table = metrics.summarize(scored)
    table.to_csv(out / "scores.csv")
    ov = metrics.overview(table)
    ov.to_csv(out / "overview.csv")
    metrics.summarize(scored, by=("model", "regime", "horizon")).to_csv(out / "regimes.csv")
    sel = metrics.holdout_selection(scored)
    sel.to_csv(out / "holdout_selection.csv")
    fp = compute_footprint(fc.attrs["runtime_s"])
    fp.to_csv(out / "compute_footprint.csv")
    print(ov.to_string())
    print("\nSélection sur 2020-23, évaluée sur 2024-26 (médiane par horizon) :")
    print(sel.groupby("horizon")[["test_rel_MAE", "test_rel_QL", "test_coverage"]].median().to_string())
    print(f"\nCalcul : {fp.loc['TOTAL', 'runtime_s']:.0f} s ≈ {fp.loc['TOTAL', 'gCO2e']:.3f} gCO2e (estimation)")
    print(f"\nÉcrit dans {out}")
    return fc, scored, table, ov


if __name__ == "__main__":
    main()
