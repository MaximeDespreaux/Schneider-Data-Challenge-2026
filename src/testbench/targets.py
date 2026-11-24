"""Cibles dérivées : indices de coût BOM (lignes ``bom_index`` du CSV).

Indice à quantités fixes (Laspeyres) : on fige les kg de chaque matière et leur prix au mois de base
(2026-09, cf. config/assumptions.yaml), puis on fait évoluer chaque prix avec sa série :

    indice(t) = 100 × Σ_i kg_i × prix_base_i × P_i(t)/P_i(base)  /  Σ_i kg_i × prix_base_i

C'est exactement « coût matières(t) / coût matières(base) » à BOM constante. Pondération par la VALEUR
(kg × €/kg) et non par la masse : le cuivre pèse peu dans la masse du RM6 mais l'essentiel de son coût.
Acier = prix MEPS/CRU sous licence (L01-L03) ou indices Eurostat (``steel_price_source``), epoxy = EP01, laiton = BR01
(dérivé), électronique = indice EL01 (ligne « une boîte » en option). « Other » et parts non étiquetées : exclues,
la couverture est donnée par ``coverage_table``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import data as d

CONFIG = d.ROOT / "config" / "assumptions.yaml"

FAMILY_TO_KEY = {
    "Carbon steel": "carbon_steel", "Stainless steel": "stainless_steel", "Galvanised steel": "galvanised_steel",
    "Copper": "copper", "Aluminium": "aluminium", "Engineering plastic": "plastics", "Plastic": "plastics",
    "Epoxy": "epoxy", "Brass": "brass",
}
TARGET_IDS = {"RM6": "RM6_NE_IQI_433520", "AirSeT": "RMAirSeT_IIQ_24kV_IEC"}
BOM_VERSION_MAP = {"csv_v2": "csv", "xlsx_v1": "root", "xlsx_v2": "v2"}
STEELS = {"carbon_steel", "stainless_steel", "galvanised_steel"}
SERIES_FIX = {
    **{s: "mettre le classeur sous licence Copia_de_Europe_Steel_INDEXES.xlsx dans data/raw/, ou mettre "
          "steel_price_source: eurostat dans config/assumptions.yaml" for s in ["L01", "L02", "L03"]},
    "EL01": "lancer `uv run python -m src.external.fetch_public`",
    "EP01": "mettre le fichier epoxy ('Epoxy for student.xlsx') dans data/raw/",
}


def load_assumptions(path: Path = CONFIG) -> dict:
    """Lit config/assumptions.yaml (local, gitignoré : contient des prix MEPS/CRU sous licence)."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} introuvable. Copier config/assumptions.example.yaml vers config/assumptions.yaml, puis remplir "
            "les valeurs FILL_LOCALLY (prix MEPS/CRU sous licence, données du challenge). Ne jamais le commiter.")
    with open(path, encoding="utf-8") as f:
        a = yaml.safe_load(f)
    base = a.get("eur_per_kg_at_base") or {}
    missing = [k for k in sorted(set(FAMILY_TO_KEY.values())) if (base.get(k) or {}).get("value") is None]
    if missing:
        raise ValueError(f"Valeurs à remplir dans {path} (eur_per_kg_at_base) : {missing}")
    if a["bom"]["version"] == "csv_v2":
        import warnings
        warnings.warn("BOM csv_v2 obsolète : l'équipe a validé xlsx_v1 (RM_Prices_XHEC_Challenge.xlsx).", stacklevel=2)
    return a


def _series_for(a: dict, key: str, product: str | None = None) -> str:
    """Série de prix d'une matière ; acier sur indices Eurostat si ``steel_price_source: eurostat``.
    ``material_series_by_product`` permet un indice régional par produit (ex. L01_FR)."""
    if a.get("steel_price_source") == "eurostat" and key in STEELS:
        return a["material_series_eurostat"][key]["series"]
    override = ((a.get("material_series_by_product") or {}).get(product) or {}).get(key)
    return override or a["material_series"][key]["series"]


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
        series = _series_for(a, key, r["product"])
        rows.append({"product": r["product"], "material": key, "series": series, "kg": r["mass_kg"],
                     "eur_per_kg": eur_kg, "value_eur": r["mass_kg"] * eur_kg})
    extras = a.get("extras") or {}
    if (extras.get("epoxy") or {}).get("include"):  # anciennes versions de BOM seulement
        if (bom["material_family"] == "Epoxy").any():
            raise ValueError("extras.epoxy.include = true mais la BOM contient déjà une ligne Epoxy : l'epoxy serait "
                             "compté deux fois. Mettre extras.epoxy.include: false.")
        eur_kg = float(a["eur_per_kg_at_base"]["epoxy"]["value"])
        for prod, g in bom.groupby("product"):
            kg = float(extras["epoxy"]["share_of_mass"]) * float(g["implied_total_kg"].iloc[0])
            rows.append({"product": prod, "material": "epoxy", "series": _series_for(a, "epoxy"), "kg": kg,
                         "eur_per_kg": eur_kg, "value_eur": kg * eur_kg})
    if (extras.get("electronics") or {}).get("include"):  # une « boîte » d'électronique par produit
        for prod in bom["product"].unique():
            eur_box = float(extras["electronics"]["eur_per_box_at_base"][prod.lower()]["value"])
            rows.append({"product": prod, "material": "electronics", "series": _series_for(a, "electronics"),
                         "kg": 1.0, "eur_per_kg": eur_box, "value_eur": eur_box})
    w = pd.DataFrame(rows).groupby(["product", "material", "series"], as_index=False).sum(numeric_only=True)
    w["eur_per_kg"] = w["value_eur"] / w["kg"]
    w["weight"] = w["value_eur"] / w.groupby("product")["value_eur"].transform("sum")
    return w


def _check_series(panel: pd.DataFrame, series) -> None:
    """Erreur claire si une série nécessaire aux poids manque au panel ou est entièrement vide."""
    for s in series:
        if s not in panel or panel[s].isna().all():
            fix = SERIES_FIX.get(s, "vérifier le panel et config/assumptions.yaml (material_series)")
            raise ValueError(f"Série {s} absente ou vide dans le panel (nécessaire à l'indice BOM) : {fix}.")


def last_valid_at(s: pd.Series, month) -> float:
    """Dernière valeur non vide à ou avant ``month`` (L02 / EL01 peuvent ne pas être publiés au mois de base)."""
    known = s.loc[:pd.Timestamp(month)].dropna()
    if known.empty:
        raise ValueError(f"Série {s.name} : aucune valeur à ou avant {pd.Timestamp(month):%Y-%m}.")
    return float(known.iloc[-1])


DIRECT_EUR_T = {"D01", "D02", "L01", "L02", "L03", "EP01", "BR01"}  # séries déjà en EUR/t ; les autres sont des indices sans unité de prix


def _chain_back(idx: pd.Series, v: pd.DataFrame, w: pd.Series, origin: pd.Timestamp) -> pd.Series:
    """Before the first month where the full basket exists (up to the origin), extend the index backwards with the
    weighted monthly growth of the components available in both months (chain-linking: no jump when a component
    starts later). Months after the origin are left as is (truth always needs the full basket)."""
    hist = idx.loc[:origin]
    gaps = hist.index[hist.isna()]
    if len(gaps) == 0 or hist.isna().all():
        return idx
    first = hist.index[hist.index > gaps.max()].min() if gaps.max() < hist.index.max() else None
    if first is None:
        return idx
    growth = v / v.shift(1)
    ok = growth.notna()
    g = (growth.where(ok).mul(w).sum(axis=1) / ok.mul(w).sum(axis=1)).where(ok.any(axis=1))
    out = idx.copy()
    pos = list(out.index).index(first)
    for i in range(pos, 0, -1):
        gi = g.iloc[i]
        if not np.isfinite(gi) or gi <= 0:
            break
        out.iloc[i - 1] = out.iloc[i] / gi
    return out


def make_bom_deriver(panel_full: pd.DataFrame | None = None, assumptions: dict | None = None):
    """Fabrique la fonction qui construit les indices BOM **à une origine donnée, sans regarder le futur**.

    À l'origine t, pour chaque matière i (pondération en coût = prix × masse, recommandée par Schneider) :
      - valeur_i(t) en EUR par boîte = kg_i × prix connu le plus récent (≤ t) en EUR/kg pour les séries en EUR/t
        (cuivre, alu, acier MEPS/CRU, epoxy, laiton) ; pour les matières suivies par un indice sans niveau
        (plastiques S11, électronique EL01) : kg_i × EUR/kg de config/assumptions.yaml (hypothèse de niveau, la
        même à toute origine : aucune valeur du panel postérieure à t n'entre dans les poids) ;
      - une matière dont la série n'a encore aucune valeur à t est exclue et les poids sont renormalisés ;
      - indice(s) = 100 × Σ_i poids_i(t) × P_i(s) / P_i(t), chaque P_i prolongé par sa dernière valeur connue
        jusqu'à t (bord irrégulier, ex. inox publié avec un mois de retard) => indice = 100 au mois d'origine.

    ``panel_full`` sert seulement à vérifier que les séries existent (et à retomber sur la série par défaut si
    une série régionale manque). Retourne derive(values_df, bases_df, origin) -> DataFrame (une colonne par produit).
    """
    a = load_assumptions() if assumptions is None else assumptions
    w = bom_weights(a)
    if panel_full is not None:
        for i, r in w.iterrows():  # série régionale absente (ex. acier sous licence désactivé) -> série par défaut
            if r.series not in panel_full or panel_full[r.series].isna().all():
                w.at[i, "series"] = _series_for(a, r.material)
        _check_series(panel_full, w["series"].unique())

    def _parts(bases_df: pd.DataFrame, origin: pd.Timestamp, g: pd.DataFrame) -> list[tuple]:
        parts = []
        for r in g.itertuples():
            known = bases_df[r.series].loc[:origin].dropna() if r.series in bases_df else pd.Series(dtype=float)
            if known.empty:
                continue
            pb = float(known.iloc[-1])
            eur_kg = pb / 1000.0 if r.series in DIRECT_EUR_T else float(r.eur_per_kg)
            parts.append((r.material, r.series, eur_kg, r.kg * eur_kg, pb))
        return parts

    def derive(values_df: pd.DataFrame, bases_df: pd.DataFrame, origin) -> pd.DataFrame:
        origin = pd.Timestamp(origin)
        out = {}
        for prod, g in w.groupby("product"):
            parts = _parts(bases_df, origin, g)
            if not parts:
                continue
            total = sum(p[3] for p in parts)
            w_ = pd.Series({s_: val / total for _, s_, _, val, _ in parts})
            v = pd.DataFrame({s_: values_df[s_] for _, s_, _, _, _ in parts}, index=values_df.index)
            v.loc[:origin] = v.loc[:origin].ffill()
            pb = pd.Series({s_: b for _, s_, _, _, b in parts})
            idx = 100 * (v / pb).mul(w_).sum(axis=1, min_count=len(w_))   # full basket only (NaN otherwise)
            out[TARGET_IDS[prod]] = _chain_back(idx, v, w_, origin)
        return pd.DataFrame(out, index=values_df.index)

    def weights_at(bases_df: pd.DataFrame, origin) -> pd.DataFrame:
        """Poids en coût de chaque matière connus à l'origine (pour la marge à risque / partie 4)."""
        origin = pd.Timestamp(origin)
        rows = []
        for prod, g in w.groupby("product"):
            for mat, s_, eur_kg, val, _ in _parts(bases_df, origin, g):
                rows.append({"product": prod, "material": mat, "series": s_, "eur_per_kg": eur_kg, "value": val})
        df = pd.DataFrame(rows)
        df["weight"] = df["value"] / df.groupby("product")["value"].transform("sum")
        return df

    derive.weights_at = weights_at
    derive.weights = w
    return derive


def add_bom_indices(panel: pd.DataFrame, assumptions: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ajoute au panel une colonne par produit (indice base 100 au mois de base). Retourne (panel, poids).

    ⚠️ Pour l'ANALYSE descriptive et pour le CSV final (origine = mois de base) uniquement. Dans le backtest,
    utiliser ``make_bom_deriver`` : ici les poids sont ceux du mois de base, donc postérieurs aux origines passées."""
    a = load_assumptions() if assumptions is None else assumptions
    base = pd.Timestamp(a["cost_index"]["base_month"] + "-01")
    w = bom_weights(a)
    _check_series(panel, w["series"].unique())
    out = panel.copy()
    for prod, g in w.groupby("product"):
        rel = 0.0
        for row in g.itertuples():
            v = panel[row.series].copy()
            v.loc[:base] = v.loc[:base].ffill()  # série pas encore publiée au mois de base : dernière valeur connue
            rel = rel + row.weight * v / last_valid_at(panel[row.series], base)
        out[TARGET_IDS[prod]] = 100 * rel
    return out, w


def coverage_table(assumptions: dict | None = None) -> pd.DataFrame:
    """Part de la masse BOM réellement pricée, par produit (epoxy compris ; électronique exclue : pas des kg).

    priced_share = kg des matières pricées ÷ masse totale implicite. ``excluded`` liste les familles non pricées
    (« Other »…) et la part non étiquetée (masse totale − masse de toutes les lignes gardées).
    """
    a = load_assumptions() if assumptions is None else assumptions
    version = a["bom"]["version"]
    bom = d.load_bom(BOM_VERSION_MAP[version], rm6_carbon_share=a["bom"]["rm6_big_unlabelled_share"])
    epoxy_extra = (a.get("extras") or {}).get("epoxy") or {}
    rows = []
    for prod, g in bom.groupby("product"):
        total = float(g["implied_total_kg"].iloc[0])
        priced = g[g["material_family"].isin(FAMILY_TO_KEY)]
        priced_kg = float(priced["mass_kg"].sum())
        if epoxy_extra.get("include"):
            priced_kg += float(epoxy_extra["share_of_mass"]) * total
        excl = [f"{f} ({m:.1f} kg)" for f, m in g.loc[~g["material_family"].isin(FAMILY_TO_KEY),
                                                       ["material_family", "mass_kg"]].itertuples(index=False)]
        unlabelled = total - float(g["mass_kg"].sum())
        if unlabelled > 0.05:
            excl.append(f"unlabelled ({unlabelled:.1f} kg)")
        rows.append({"product": prod, "bom_version": version, "implied_total_kg": total, "priced_kg": priced_kg,
                     "priced_share": priced_kg / total, "excluded": ", ".join(excl) or "-"})
    return pd.DataFrame(rows)
