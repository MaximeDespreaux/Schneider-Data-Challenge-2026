"""Audit anti-fuite sur les VRAIES données (à lancer en local, pas en CI : data/ est vide en CI).

    uv run python -m src.testbench.audit

Principe (test de perturbation) : pour plusieurs origines t, on remplace TOUTES les valeurs de TOUTES les séries
après t par du bruit, puis on relance le backtest à l'origine t. Si une seule prévision change, c'est qu'un modèle
(ou un indice dérivé) a lu le futur d'une matière — la sienne ou une autre. Les deux modes sont audités.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .models import baseline_models
from .run import DEFAULT_TARGETS, bom_deriver, build_panel

COLS = ["model", "series", "origin", "horizon", "step", "p10", "p50", "p90"]


def perturb_future(panel: pd.DataFrame, origin, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = panel.copy()
    fut = p.index > pd.Timestamp(origin)
    p.loc[fut] = p.loc[fut] * rng.uniform(0.2, 5.0, size=p.loc[fut].shape)
    return p


def audit(origins=("2020-06-01", "2022-03-01", "2024-09-01"), modes=("submission", "realtime")) -> pd.DataFrame:
    panel = build_panel()
    derive = bom_deriver(panel)  # constantes de calibrage figées une fois (cf. targets.make_bom_deriver)
    models = baseline_models()
    rows = []
    for mode in modes:
        for o in origins:
            a = run_backtest(panel, DEFAULT_TARGETS, models, origins=[o], mode=mode, derive=derive, verbose=False)
            b = run_backtest(perturb_future(panel, o), DEFAULT_TARGETS, models, origins=[o], mode=mode,
                             derive=derive, verbose=False)
            m = a[COLS].merge(b[COLS], on=COLS[:5], suffixes=("", "_pert"))
            diff = max((m[c] - m[c + "_pert"]).abs().max() for c in ["p10", "p50", "p90"])
            rows.append({"mode": mode, "origin": o, "n_forecasts": len(m), "same_rows": len(a) == len(b) == len(m),
                         "max_abs_diff": diff, "PASS": bool(diff == 0 and len(a) == len(b) == len(m))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    res = audit()
    print(res.to_string(index=False))
    print("\nRÉSULTAT :", "AUCUNE FUITE ✅" if res["PASS"].all() else "FUITE DÉTECTÉE ❌")
