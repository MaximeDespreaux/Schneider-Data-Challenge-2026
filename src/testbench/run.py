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
    if with_bom:
        from .targets import add_bom_indices
        panel, _ = add_bom_indices(panel)
    return panel


def bom_deriver(panel):
    from .targets import make_bom_deriver
    return make_bom_deriver(panel)


DEFAULT_TARGETS = d.RM_TARGETS + ["RM6_NE_IQI_433520", "RMAirSeT_IIQ_24kV_IEC"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="submission", choices=["submission", "realtime"])
    ap.add_argument("--start", default="2020-01")
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    args = ap.parse_args(argv)

    panel = build_panel()
    targets = args.targets.split(",")
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
