"""Chargement des données du challenge Schneider × X-HEC 2026.

Toutes les données vivent dans ``data/raw/`` (gitignoré, NDA). Ce module ne contient
aucune valeur : seulement des chemins, des schémas et des hypothèses documentées.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PANEL = RAW / "panel"
OUT = ROOT / "data" / "outputs"  # sorties confidentielles (figures, tables), gitignorées

PRODUCTS = {
    "RM6": "RM6_NE_IQI_433520",
    "AirSeT": "RMAirSeT IIQ 24kV IEC",
}

# Séries à prévoir (proxys gratuits). Les séries L sont vides par construction.
RM_TARGETS = ["S01", "S02", "S03", "S04", "S05", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "D01", "D02"]

# Délais de publication (mois) — ce qui est connu à la date d'origine.
PUBLICATION_LAG = {
    **{s: 1 for s in ["S01", "S02", "S03", "S04", "S05", "D01", "D02"]},  # IMF via FRED
    "S06": 0,
    **{s: 2 for s in ["S07", "S08", "S09", "S10", "S11"]},  # Eurostat
    **{s: 1 for s in ["S12", "S13"]},  # BLS
}

# Prix d'ancrage (EUR/kg, moyenne 2021) pour les matières suivies par un INDICE (base 2021=100).
# HYPOTHÈSES à remplacer par les niveaux CRU/MEPS du workbook acier dès que possible.
ANCHOR_EUR_PER_KG_2021 = {
    "Carbon steel": 0.95,  # CR coil Europe ~950 EUR/t en 2021
    "Stainless steel": 3.00,  # 304 CR 2 mm, base + surcharge
    "Galvanised steel": 1.00,  # HDG coil
    "Engineering plastic": 2.50,  # PBT / ETP
    "Plastic": 1.50,  # PET
}

# Matière BOM -> série de prix utilisée pour valoriser le coût (Europe)
FAMILY_TO_SERIES = {
    "Carbon steel": "S09",
    "Stainless steel": "S09",
    "Galvanised steel": "S10",
    "Engineering plastic": "S11",
    "Plastic": "S11",
    "Copper": "D01",  # EUR/t
    "Aluminium": "D02",  # EUR/t
}
DIRECT_PRICE_SERIES = {"D01", "D02"}  # déjà en EUR/t


def load_rm_wide() -> pd.DataFrame:
    df = pd.read_csv(PANEL / "rm_prices_monthly_wide.csv", parse_dates=["month"], index_col="month")
    return df.drop(columns=["year", "month_num"])


def load_energy_wide() -> pd.DataFrame:
    return pd.read_csv(PANEL / "energy_prices_monthly_wide.csv", parse_dates=["month"], index_col="month")


def load_catalogs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rm = pd.read_csv(PANEL / "source_catalog.csv").set_index("series_id")
    en = pd.read_csv(PANEL / "energy_source_catalog.csv").set_index("series_id")
    return rm, en


def load_bom(version: str = "csv", rm6_carbon_share: float = 0.45) -> pd.DataFrame:
    """BOM nettoyée, une ligne par (produit, matière).

    version="csv"  -> panel/bom_materials.csv (data.zip, documentée)
    version="root" -> RM_Prices_XHEC_Challenge.xlsx à la racine (plus récente, en conflit)
    La grosse part non étiquetée du RM6 est supposée en acier carbone : masse = rm6_carbon_share × masse totale implicite.
    """
    if version == "csv":
        bom = pd.read_csv(PANEL / "bom_materials.csv")
    elif version == "root":
        bom = pd.read_excel(RAW / "RM_Prices_XHEC_Challenge.xlsx", sheet_name="BOM_Materials")
    else:
        raise ValueError(version)
    bom = bom[bom["product_ref"].isin(PRODUCTS.values())].copy()
    bom["implied_total_kg"] = bom["mass_kg"] / bom["share_pct"]
    rows = []
    for prod, g in bom.groupby("product_ref"):
        total = g["implied_total_kg"].median()
        for _, r in g.iterrows():
            fam = r["material_family"]
            mass = r["mass_kg"]
            if fam == "Carbon steel (to confirm)":
                fam, mass = "Carbon steel", rm6_carbon_share * total
            if pd.isna(mass) or fam in ("To confirm",):
                continue
            rows.append({"product_ref": prod, "material_family": fam, "mass_kg": mass,
                         "share_pct": mass / total, "implied_total_kg": total})
    out = pd.DataFrame(rows)
    out["product"] = out["product_ref"].map({v: k for k, v in PRODUCTS.items()})
    out["bom_version"] = version
    return out.groupby(["product", "material_family", "bom_version"], as_index=False).agg(
        mass_kg=("mass_kg", "sum"), share_pct=("share_pct", "sum"), implied_total_kg=("implied_total_kg", "first"))


def material_price_eur_per_kg(rm: pd.DataFrame) -> pd.DataFrame:
    """Prix mensuel en EUR/kg par famille de matière (prix direct ou indice × ancrage 2021)."""
    out = {}
    for fam, s in FAMILY_TO_SERIES.items():
        if s in DIRECT_PRICE_SERIES:
            out[fam] = rm[s] / 1000.0
        else:
            base = rm.loc["2021", s].mean()
            out[fam] = rm[s] / base * ANCHOR_EUR_PER_KG_2021[fam]
    return pd.DataFrame(out)


def bom_cost(rm: pd.DataFrame, bom: pd.DataFrame) -> pd.DataFrame:
    """Coût matières « pricé » mensuel par produit et matière (EUR/unité). « Other » est exclu."""
    px = material_price_eur_per_kg(rm)
    frames = []
    for _, r in bom.iterrows():
        if r["material_family"] not in px:
            continue
        frames.append(pd.DataFrame({"product": r["product"], "material_family": r["material_family"],
                                    "cost_eur": px[r["material_family"]] * r["mass_kg"]}))
    return pd.concat(frames).reset_index(names="month")


def load_rmu_price_benchmark() -> pd.DataFrame:
    """Prix de vente annuels synthétiques (P10/P50/P90, EUR) d'un RMU 3 fonctions par marque."""
    df = pd.read_excel(RAW / "RMU_IQI_EU_Price_Benchmark_2017-2026_Synthetic.xlsx", sheet_name="ML_Trend", header=3)
    df = df.rename(columns=lambda c: str(c).replace(" (ML)", "").strip())
    df = df.dropna(subset=["Brand"])
    df["Year"] = df["Year"].astype(int)
    df["YoY P50"] = pd.to_numeric(df["YoY P50"], errors="coerce")
    return df


def load_epoxy() -> pd.DataFrame:
    df = pd.read_excel(RAW / "Epoxy for student.xlsx", sheet_name="Feuil1")
    df = df.rename(columns={df.columns[0]: "date"})
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    return df.set_index("month").drop(columns="date")


def dlog(df: pd.DataFrame | pd.Series, k: int = 1):
    return np.log(df).diff(k)
