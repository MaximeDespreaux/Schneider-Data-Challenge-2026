"""Format du CSV de soumission (slide « Recommended submission ») et validateur.

Partagé par tout le monde : la partie 1 l'utilise pour écrire le fichier final, la partie 2 pour vérifier
que n'importe quelle sortie de modèle respecte le format avant de la scorer.
"""

from __future__ import annotations

import pandas as pd

COLUMNS = ["team_id", "target_type", "target_id", "region", "origin_month", "horizon_m", "target_month", "unit",
           "p10", "p50", "p90", "model_name", "ai_tools"]
TARGET_TYPES = {"material", "bom_index"}
REGIONS = {"EU", "WW"}
HORIZONS = {3, 6, 9}


def from_forecasts(fc: pd.DataFrame, origin: str, team_id: str, model_name: str, ai_tools: str,
                   region: dict[str, str], unit: dict[str, str], target_type: str = "material") -> pd.DataFrame:
    """Convertit une sortie du banc (model, series, origin, horizon, target_month, p10, p50, p90) au format CSV."""
    d = fc[fc["origin"] == pd.Timestamp(origin)].copy()
    out = pd.DataFrame({
        "team_id": team_id, "target_type": target_type, "target_id": d["series"], "region": d["series"].map(region),
        "origin_month": pd.Timestamp(origin).strftime("%Y-%m"), "horizon_m": d["horizon"].astype(int),
        "target_month": d["target_month"].dt.strftime("%Y-%m"), "unit": d["series"].map(unit),
        "p10": d["p10"], "p50": d["p50"], "p90": d["p90"], "model_name": model_name, "ai_tools": ai_tools,
    })
    return out[COLUMNS].reset_index(drop=True)


def validate_submission(df: pd.DataFrame) -> list[str]:
    """Liste des problèmes trouvés (vide = OK)."""
    issues = []
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        return [f"colonnes manquantes : {missing}"]
    if list(df.columns) != COLUMNS:
        issues.append("ordre des colonnes différent du gabarit")
    if df[COLUMNS].isna().any().any():
        issues.append(f"valeurs manquantes dans : {df.columns[df.isna().any()].tolist()}")
    if not set(df["target_type"]) <= TARGET_TYPES:
        issues.append(f"target_type invalide : {set(df['target_type']) - TARGET_TYPES}")
    if not set(df["region"]) <= REGIONS:
        issues.append(f"region invalide : {set(df['region']) - REGIONS}")
    if not set(df["horizon_m"]) <= HORIZONS:
        issues.append(f"horizon_m invalide : {set(df['horizon_m']) - HORIZONS}")
    try:
        o = pd.PeriodIndex(df["origin_month"], freq="M")
        t = pd.PeriodIndex(df["target_month"], freq="M")
        bad = [(a, b, h) for a, b, h in zip(o, t, df["horizon_m"]) if (b - a).n != h]
        if bad:
            issues.append(f"target_month ≠ origin_month + horizon_m sur {len(bad)} lignes")
    except Exception as e:  # noqa: BLE001
        issues.append(f"dates au mauvais format (attendu YYYY-MM) : {e}")
    if not ((df["p10"] <= df["p50"]) & (df["p50"] <= df["p90"])).all():
        issues.append("p10 <= p50 <= p90 non respecté")
    dup = df.duplicated(["target_type", "target_id", "region", "horizon_m"]).sum()
    if dup:
        issues.append(f"{dup} doublons (target_type, target_id, region, horizon_m)")
    return issues
