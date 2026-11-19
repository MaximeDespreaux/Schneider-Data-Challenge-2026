"""Cibles dérivées : indices de coût BOM (lignes ``bom_index`` du CSV).

Indice à quantités fixes (Laspeyres) : on fige les kg de chaque matière et leur prix au mois de base
(2026-09, cf. config/assumptions.yaml), puis on fait évoluer chaque prix avec sa série :

    indice(t) = 100 × Σ_i kg_i × prix_base_i × P_i(t)/P_i(base)  /  Σ_i kg_i × prix_base_i

C'est exactement « coût matières(t) / coût matières(base) » à BOM constante. Pondération par la VALEUR
(kg × €/kg) et non par la masse : le cuivre pèse peu dans la masse du RM6 mais l'essentiel de son coût.
Matières absentes du panel (epoxy, électronique, « Other ») : exclues ici, la couverture est indiquée.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src import data as d

CONFIG = d.ROOT / "config" / "assumptions.yaml"

FAMILY_TO_KEY = {
    "Carbon steel": "carbon_steel", "Stainless steel": "stainless_steel", "Galvanised steel": "galvanised_steel",
    "Copper": "copper", "Aluminium": "aluminium", "Engineering plastic": "plastics", "Plastic": "plastics",
}
TARGET_IDS = {"RM6": "RM6_NE_IQI_433520", "AirSeT": "RMAirSeT_IIQ_24kV_IEC"}
BOM_VERSION_MAP = {"csv_v2": "csv", "xlsx_v1": "root"}


def load_assumptions(path: Path = CONFIG) -> dict:
    """Lit config/assumptions.yaml (local, gitignoré : contient des prix MEPS/CRU sous licence)."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} introuvable. Copier config/assumptions.example.yaml vers config/assumptions.yaml, puis remplir "
            "les valeurs FILL_LOCALLY (prix MEPS/CRU sous licence, données du challenge). Ne jamais le commiter.")
    with open(path, encoding="utf-8") as f:
        a = yaml.safe_load(f)
    missing = [k for k, v in a.get("eur_per_kg_at_base", {}).items()
               if k in set(FAMILY_TO_KEY.values()) and v.get("value") is None]
    if missing:
        raise ValueError(f"Valeurs à remplir dans {path} (eur_per_kg_at_base) : {missing}")
    if a["bom"]["version"] == "csv_v2":
        import warnings
        warnings.warn("BOM csv_v2 obsolète : l'équipe a validé xlsx_v1 (RM_Prices_XHEC_Challenge.xlsx).", stacklevel=2)
    return a


def bom_weights(assumptions: dict | None = None) -> pd.DataFrame:
    """Poids en valeur de chaque matière au mois de base, par produit."""
    a = load_assumptions() if assumptions is None else assumptions
    bom = d.load_bom(BOM_VERSION_MAP[a["bom"]["version"]], rm6_carbon_share=a["bom"]["rm6_big_unlabelled_share"])
    rows = []
    for _, r in bom.iterrows():
        key = FAMILY_TO_KEY.get(r["material_family"])
        if key is None:
            continue
        eur_kg = float(a["eur_per_kg_at_base"][key]["value"])
        series = a["material_series"][key]["series"]
        rows.append({"product": r["product"], "material": key, "series": series, "kg": r["mass_kg"],
                     "eur_per_kg": eur_kg, "value_eur": r["mass_kg"] * eur_kg})
    w = pd.DataFrame(rows).groupby(["product", "material", "series"], as_index=False).sum(numeric_only=True)
    w["eur_per_kg"] = w["value_eur"] / w["kg"]
    w["weight"] = w["value_eur"] / w.groupby("product")["value_eur"].transform("sum")
    return w


DIRECT_EUR_T = {"D01", "D02"}  # séries déjà en EUR/t ; les autres sont des indices sans unité de prix


def make_bom_deriver(panel_full: pd.DataFrame, assumptions: dict | None = None):
    """Fabrique la fonction qui construit les indices BOM **à une origine donnée, sans regarder le futur**.

    À l'origine t, pour chaque matière i :
      - prix de référence p_i(t) en EUR/kg = valeur connue la plus récente (≤ t) :
          * cuivre / alu (D01, D02) : directement le prix EUR/t ÷ 1000 ;
          * acier / plastiques (indices) : k_i × indice_i(t), avec k_i = EUR/kg d'ancrage ÷ indice au mois d'ancrage.
            k_i est une CONSTANTE de calibrage de niveau (hypothèse, comme les EUR/kg du fichier d'hypothèses),
            pas une donnée de prévision : elle fixe le niveau absolu, pas les variations.
      - poids_i(t) = kg_i × p_i(t) / Σ_j kg_j × p_j(t)   (connus à t)
      - indice(s) = 100 × Σ_i poids_i(t) × P_i(s) / P_i(base_i(t))   où base_i(t) = dernière date connue de i
    => indice = 100 au mois d'origine, comme dans le CSV (« index (origin = 100) »).

    Retourne derive(values_df, bases_df, origin) -> DataFrame (une colonne par produit).
      - bases_df : ce qui est connu à l'origine (sert à fixer les poids et les bases) ;
      - values_df: les valeurs à indexer (l'info tronquée pour l'historique, le panel complet pour la vérité).
    """
    a = load_assumptions() if assumptions is None else assumptions
    anchor_month = pd.Timestamp(a["cost_index"]["base_month"] + "-01")
    w = bom_weights(a)
    k = {}
    for r in w.itertuples():
        if r.series not in DIRECT_EUR_T:
            k[r.material] = r.eur_per_kg / panel_full.at[anchor_month, r.series]

    def derive(values_df: pd.DataFrame, bases_df: pd.DataFrame, origin) -> pd.DataFrame:
        origin = pd.Timestamp(origin)
        out = {}
        for prod, g in w.groupby("product"):
            parts, total = [], 0.0
            for r in g.itertuples():
                known = bases_df[r.series].loc[:origin].dropna()
                if known.empty:
                    break
                pb = float(known.iloc[-1])
                eur_kg = pb / 1000.0 if r.series in DIRECT_EUR_T else k[r.material] * pb
                val = r.kg * eur_kg
                parts.append((val, r.series, pb))
                total += val
            else:
                out[TARGET_IDS[prod]] = 100 * sum(v / total * values_df[s] / pb for v, s, pb in parts)
        return pd.DataFrame(out, index=values_df.index)

    def weights_at(bases_df: pd.DataFrame, origin) -> pd.DataFrame:
        """Poids en valeur de chaque matière connus à l'origine (pour les analyses de la partie 4)."""
        origin = pd.Timestamp(origin)
        rows = []
        for r in w.itertuples():
            pb = float(bases_df[r.series].loc[:origin].dropna().iloc[-1])
            eur_kg = pb / 1000.0 if r.series in DIRECT_EUR_T else k[r.material] * pb
            rows.append({"product": r.product, "material": r.material, "series": r.series, "value": r.kg * eur_kg})
        df = pd.DataFrame(rows)
        df["weight"] = df["value"] / df.groupby("product")["value"].transform("sum")
        return df

    derive.weights_at = weights_at
    return derive


def add_bom_indices(panel: pd.DataFrame, assumptions: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ajoute au panel une colonne par produit (indice base 100 au mois de base). Retourne (panel, poids).

    ⚠️ Pour l'ANALYSE descriptive et pour le CSV final (origine = mois de base) uniquement. Dans le backtest,
    utiliser ``make_bom_deriver`` : ici les poids sont ceux du mois de base, donc postérieurs aux origines passées."""
    a = load_assumptions() if assumptions is None else assumptions
    base = pd.Timestamp(a["cost_index"]["base_month"] + "-01")
    w = bom_weights(a)
    out = panel.copy()
    for prod, g in w.groupby("product"):
        rel = sum(row.weight * panel[row.series] / panel.at[base, row.series] for row in g.itertuples())
        out[TARGET_IDS[prod]] = 100 * rel
    return out, w
