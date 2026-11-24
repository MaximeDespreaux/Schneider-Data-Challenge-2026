"""Chargement des données du challenge Schneider × X-HEC 2026.

Toutes les données vivent dans ``data/raw/`` (gitignoré, NDA). Ce module ne contient
aucune valeur : seulement des chemins, des schémas et des hypothèses documentées.
"""

from __future__ import annotations

import re
from datetime import datetime
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

# Séries à prévoir. Les 14 séries d'origine de Schneider (proxys gratuits) : seules celles-ci vont dans le CSV.
SCHNEIDER_RM_TARGETS = ["S01", "S02", "S03", "S04", "S05", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "D01", "D02"]
# Séries ajoutées : acier sous licence MEPS/CRU (L01-L03), epoxy (EP01), électronique Eurostat (EL01).
EXTRA_TARGETS = ["L01", "L02", "L03", "EP01", "EL01"]
RM_TARGETS = SCHNEIDER_RM_TARGETS + EXTRA_TARGETS

# Publication lags live in config/availability.yaml (single source of truth).

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


def load_bom(version: str = "v2", rm6_carbon_share: float = 0.45, allow_deprecated: bool = False) -> pd.DataFrame:
    """BOM nettoyée, une ligne par (produit, matière).

    version="v2"   -> RM_Prices_XHEC_Challenge_v1.xlsx : **nouvelle BOM officielle** (envoyée par Schneider ~20 nov. ;
                      = ``xlsx_v2`` dans config/assumptions.yaml)
    version="root" -> RM_Prices_XHEC_Challenge.xlsx : version officielle précédente, gardée pour les tests de
                      sensibilité (= ``xlsx_v1``)
    version="csv"  -> panel/bom_materials.csv (data.zip) : **obsolète**
    Les lignes epoxy (``material_label_source`` contenant « Epoxy ») passent en famille « Epoxy ».
    Anciennes versions seulement : la grosse part non étiquetée du RM6 est supposée en acier carbone
    (masse = rm6_carbon_share × masse totale implicite).
    """
    if version == "csv" and not allow_deprecated:
        raise ValueError("BOM csv is deprecated (see data/Data/BOM_SOURCE_OF_TRUTH.md); "
                         "pass allow_deprecated=True for sensitivity only")
    if version == "csv":
        bom = pd.read_csv(PANEL / "bom_materials.csv")
    elif version == "root":
        bom = pd.read_excel(RAW / "RM_Prices_XHEC_Challenge.xlsx", sheet_name="BOM_Materials")
    elif version == "v2":
        bom = pd.read_excel(RAW / "RM_Prices_XHEC_Challenge_v1.xlsx", sheet_name="BOM_Materials")
    else:
        raise ValueError(version)
    bom = bom[bom["product_ref"].isin(PRODUCTS.values())].copy()
    epoxy = bom["material_label_source"].astype(str).str.contains("Epoxy", case=False, na=False)
    bom.loc[epoxy, "material_family"] = "Epoxy"
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


EPOXY_COLUMNS = ["EPX_BENZENE", "EPX_ACETONE", "EPX_ECH", "EPX_OTHER", "EPX_RESIN"]  # file order
# Data corrections applied on load (documented, reversible with correct=False).
EPOXY_CORRECTIONS = [{"month": "2026-05", "column": "EPX_ECH", "factor": 10,
                      "reason": "decimal typo, resin left as delivered and flagged"}]


def correct_epoxy(df: pd.DataFrame) -> pd.DataFrame:
    """Apply EPOXY_CORRECTIONS to a copy of ``df``; the list of applied corrections goes in ``attrs``."""
    out = df.copy()
    applied = []
    for c in EPOXY_CORRECTIONS:
        m = pd.Timestamp(c["month"] + "-01")
        if c["column"] in out and m in out.index:
            out.at[m, c["column"]] = out.at[m, c["column"]] * c["factor"]
            applied.append(dict(c))
    out.attrs["corrections"] = applied
    return out


def load_epoxy(correct: bool = True) -> pd.DataFrame:
    """Epoxy cost build-up (EUR/t), month-start index, columns EPX_*. ECH May-2026 typo corrected (x10)."""
    path = RAW / "Epoxy for student.xlsx"
    if not path.exists():
        path = RAW / "Epoxy_for_student.xlsx"
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier epoxy introuvable : mettre 'Epoxy for student.xlsx' (ou 'Epoxy_for_student.xlsx') dans {RAW}")
    df = pd.read_excel(path, sheet_name="Feuil1")
    df = df.rename(columns={df.columns[0]: "date"})
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    df = df.set_index("month").drop(columns="date")
    if len(df.columns) != len(EPOXY_COLUMNS):
        raise ValueError(f"epoxy file: expected {len(EPOXY_COLUMNS)} columns, got {list(df.columns)}")
    df.columns = EPOXY_COLUMNS
    return correct_epoxy(df) if correct else df


def load_epoxy_series() -> pd.Series:
    """Prix mensuel de la résine epoxy (EUR/t), série ``EP01`` indexée au début de mois."""
    s = pd.to_numeric(load_epoxy()["EPX_RESIN"], errors="coerce").astype(float)
    s = s.groupby(level=0).mean().sort_index()
    s.index.name = "month"
    return s.rename("EP01")


def _num(v) -> float:
    """Cellule Excel -> float (NaN si non numérique : '-', None, texte)."""
    if isinstance(v, bool):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return np.nan


def load_licensed_steel() -> pd.DataFrame:
    """Prix acier sous licence (MEPS / CRU), EUR/t : L01 = MEPS DE CR coil, L02 = CRU DE 304 CR-2mm (base +
    surcharge), L03 = MEPS DE HDG coil, L01_FR = MEPS FR CR coil (indice régional). Lecture par en-têtes
    (``src.licensed``). Fichier sous licence : jamais commité, jamais dans le CSV."""
    from src.licensed import load_licensed
    return load_licensed()


TEARDOWN_LINE = re.compile(r"^\s{2}(?P<material>[^—]+?)\s—\s(?P<grams>[\d\s]+)\sg")
TEARDOWN_BRANDS = {"ormazabal": "Ormazabal", "eaton": "Eaton", "SGC": "SGC", "Schneider": "Schneider", "ABB": "ABB",
                   "Siemens": "Siemens"}
TEARDOWN_GROUPS = {
    "copper": ["Copper"],
    "carbon_steel": ["Steel"],
    "aluminium": ["Aluminium"],
    "brass": ["Brass"],
    "silver": ["Silver"],
    "epoxy": ["Epoxy", "Cast", "Thermoset"],
    "plastics": ["Thermoplastic", "Polyester", "PA66", "ABS", "Polyamide", "Polycarbonate", "Polyethylene", "PTFE",
                 "Glass-reinforced", "Laminated"],
    "electronics": ["PCB", "Printed", "FR-4"],
    "sf6": ["SF6", "Sulphur"],
    "packaging": ["Wood", "Softwood", "Cardboard", "Paper"],
}
_TEARDOWN_GROUP_OF = {m.lower(): g for g, ms in TEARDOWN_GROUPS.items() for m in ms}


def load_competitor_teardowns() -> pd.DataFrame:
    """Masses matières des teardowns de RMU concurrents (estimations d'une agence externe).

    Une ligne par (marque, produit, matière) : brand, product, material, group, mass_kg, total_kg, broken
    (``broken`` = total < 1 kg, teardown inexploitable).
    """
    import openpyxl

    path = RAW / "competitor_teardown_rm6_class.xlsx"
    if not path.exists():
        alt = sorted(RAW.glob("benchmark-modules-fonctionnels-rm6*.xlsx"))
        if not alt:
            raise FileNotFoundError(f"Teardowns concurrents introuvables : {path}")
        path = alt[0]
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = []
    try:
        for name in wb.sheetnames:
            if not name.startswith("BOM "):
                continue
            col_a = [r[0] if r else None for r in wb[name].iter_rows(max_col=1, values_only=True)]
            title = str(col_a[0]) if col_a and col_a[0] is not None else ""
            product = title.split("—", 1)[1].strip() if "—" in title else title.strip()
            first = name[len("BOM "):].split()[0]
            brand = TEARDOWN_BRANDS.get(first, first)
            for v in col_a[1:]:
                if not isinstance(v, str):
                    continue
                m = TEARDOWN_LINE.match(v)
                if not m:
                    continue
                material = m.group("material").strip().rstrip(",").strip()
                grams = float(re.sub(r"\s", "", m.group("grams")))
                rows.append({"brand": brand, "product": product, "material": material,
                             "group": _TEARDOWN_GROUP_OF.get(material.lower(), "other"), "mass_kg": grams / 1000.0})
    finally:
        wb.close()
    df = pd.DataFrame(rows, columns=["brand", "product", "material", "group", "mass_kg"])
    df = df.groupby(["brand", "product", "material", "group"], as_index=False, sort=False)["mass_kg"].sum()
    df["total_kg"] = df.groupby(["brand", "product"])["mass_kg"].transform("sum")
    df["broken"] = df["total_kg"] < 1.0
    return df


def dlog(df: pd.DataFrame | pd.Series, k: int = 1):
    return np.log(df).diff(k)
