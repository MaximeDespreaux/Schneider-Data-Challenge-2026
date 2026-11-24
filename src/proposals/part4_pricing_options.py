"""PROPOSITION pour la partie 4 (à reprendre/valider par l'équipe) : rejouer les options de prix d'un devis ETO.

Question : un devis signé à l'origine t fixe le prix pour h mois. Le coût matières évolue pendant ce temps.
Quelle règle de prix aurait le mieux protégé la marge, sans rendre l'offre trop chère ?

Pour chaque origine passée t (2020 → 2026), chaque horizon h et chaque produit, on compare :
  A. prix fixe, rien ajouté
  B. prix fixe + ajustement au P50 prévu (on facture la hausse attendue)
  C. prix fixe + marge de sécurité jusqu'au P90 (on facture le scénario défavorable)
  D. clause d'indexation (le prix suit l'indice BOM : 0 risque pour Schneider, tout le risque chez le client)
  E. (RM6) prix fixe + couverture du cuivre (on fige la part cuivre à l'origine via le marché à terme)
  F. hybride : P50 + indexation au-delà d'un seuil (le client paie ce qui dépasse le P90)

Tout est exprimé en POINTS DE MARGE (% du prix de vente) : érosion = part matières × (hausse réelle − hausse facturée).
Hypothèses lues dans config/assumptions.yaml (part matières dans le prix, marge, plancher de marge).
Prévisions : celles du banc de test (modèle `naive_vol` par défaut), donc SANS FUITE.

    uv run python -m src.proposals.part4_pricing_options
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import data as d
from src.testbench.backtest import default_origins, run_backtest
from src.testbench.models import NaiveVolScaled
from src.testbench.run import bom_deriver, build_panel
from src.testbench.targets import TARGET_IDS, load_assumptions

PROD_OF = {v: k for k, v in TARGET_IDS.items()}


def simulate(model=None, start: str = "2020-01", material_share: float | None = None) -> pd.DataFrame:
    a = load_assumptions()
    ms = a["pricing"]["material_share_of_price"]["value"] if material_share is None else material_share
    margin = {"RM6": a["pricing"]["profit_margin_rm6"]["value"], "AirSeT": a["pricing"]["profit_margin_airset"]["value"]}
    floor = {"RM6": a["pricing"]["margin_floor"]["rm6"]["value"], "AirSeT": a["pricing"]["margin_floor"]["airset"]["value"]}

    panel = build_panel()
    derive = bom_deriver(panel)
    model = model or NaiveVolScaled()
    fc = run_backtest(panel, list(TARGET_IDS.values()), [model], origins=default_origins(panel, start),
                      derive=derive, verbose=False).dropna(subset=["y_true"])

    rows = []
    for r in fc.itertuples():
        prod = PROD_OF[r.series]
        base = 100.0  # l'indice vaut 100 à l'origine (poids et bases fixés à l'origine)
        real = r.y_true / base - 1
        p50, p90 = r.p50 / base - 1, r.p90 / base - 1
        # couverture cuivre : part cuivre (poids connu à l'origine) × variation réelle du cuivre
        hedged = 0.0
        if prod == "RM6":
            w = derive.weights_at(panel.loc[:r.origin], r.origin)
            w_cu = float(w[(w["product"] == "RM6") & (w["material"] == "copper")]["weight"].sum())
            cu = panel["D01"]
            hedged = w_cu * (cu.get(r.target_month, np.nan) / cu.loc[:r.origin].dropna().iloc[-1] - 1)
        charged = {"A_fixe": 0.0, "B_fixe+P50": p50, "C_fixe+P90": p90, "D_indexé": real,
                   "F_P50+index>P90": p50 + max(real - p90, 0.0)}
        if prod == "RM6":
            charged["E_fixe+couv_cuivre"] = hedged  # gain de la couverture = hausse cuivre réalisée
        for opt, c in charged.items():
            erosion = ms * (real - c)  # en part du prix de vente
            rows.append({"product": prod, "origin": r.origin, "horizon": r.horizon, "option": opt,
                         "cost_change_pct": 100 * real, "uplift_pct_price": 100 * ms * (c if opt != "E_fixe+couv_cuivre" else 0.0),
                         "margin_loss_pts": 100 * erosion, "margin_after": margin[prod] - erosion,
                         "below_floor": (margin[prod] - erosion) < floor[prod]})
    return pd.DataFrame(rows)


def summarize(sim: pd.DataFrame) -> pd.DataFrame:
    g = sim.groupby(["product", "horizon", "option"])
    return pd.DataFrame({
        "n": g.size(),
        "hausse_facturée_moy_pts": g["uplift_pct_price"].mean(),
        "perte_marge_moy_pts": g["margin_loss_pts"].mean(),
        "perte_marge_P95_pts": g["margin_loss_pts"].quantile(0.95),
        "pire_cas_pts": g["margin_loss_pts"].max(),
        "part_devis_sous_plancher": g["below_floor"].mean(),
    }).round(2)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    sim = simulate()
    out = d.OUT / "part4_proposal"
    out.mkdir(parents=True, exist_ok=True)
    sim.to_parquet(out / "pricing_options_sim.parquet")
    tab = summarize(sim)
    tab.to_csv(out / "pricing_options_summary.csv")
    print(tab.to_string())
    print(f"\nÉcrit dans {out}")
