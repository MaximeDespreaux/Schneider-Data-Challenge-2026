"""Signaux candidats pour prévoir la variation d'une série à h mois.

Chaque signal à la date t n'utilise QUE des valeurs ≤ t (diff, moyennes glissantes passées) : il peut donc être
calculé sur l'information tronquée d'une origine sans aucune fuite.

Familles :
- own_*      : dynamique propre (momentum 1/3/12 mois, accélération, écart à la moyenne 12 / 60 mois)
- cross_*    : autres matières du panel (minerai, nickel, zinc, cuivre, alu) et énergie (Brent, gaz, électricité)
- X_*        : signaux externes publics (chaîne des prix US, demande, nervosité, dollar, supply chain, LME)
- Créatifs   :
    * breadth3       : part des 14 matières en hausse sur 3 mois (« la marée monte-t-elle pour tout le monde ? »)
    * overext36      : distance au plus haut des 36 derniers mois (surchauffe -> risque de repli)
    * real_dev120    : prix réel (déflaté par le PPI US global) vs sa moyenne 10 ans (retour vers le coût marginal)
    * twin_gap       : écart indice Eurostat / prix de marché jumeau vs sa moyenne 24 mois (l'indice rattrape le marché)
    * upstream3      : variation 3 mois des matières brutes US moins celle des biens intermédiaires (la vague arrive)
    * scrap_spread3  : ferraille d'acier vs produits sidérurgiques US (la ferraille mène l'acier électrique)
    * cu_term / al_term : structure à terme LME (backwardation = tension physique)
    * vol_regime     : volatilité 3 mois / volatilité 36 mois de la série (marché agité ?)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MATERIALS = ["S01", "S02", "S03", "S04", "S05", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "D01", "D02"]
PRICE_LIKE_EXT = ["X_US_PPI_ALL", "X_US_PPI_CRUDE", "X_US_PPI_INTERMED", "X_US_PPI_STEEL", "X_US_STEEL_SCRAP",
                  "X_US_PPI_NONFERROUS", "X_FREIGHT", "X_US_INDPRO", "X_US_IP_METALS", "X_US_HOUSING", "X_US_NEW_ORDERS",
                  "X_USD_BROAD", "X_IMF_METALS", "X_IMF_ALLCOMMOD"]
LEVEL_EXT = ["X_VIX", "X_OVX", "X_GLOBAL_EPU", "X_GSCPI", "X_US_CAPUTIL", "X_US_BREAKEVEN", "X_US_10Y"]
CROSS = ["S03", "S04", "S05", "D01", "D02", "ED01", "ED02", "E09", "E11", "E13"]
# Option « champion v2 » : v1 + les autres séries énergie fournies (charbon, gaz US, GNL Japon, électricité FR/JP,
# PPI énergie DE/FR, imports russes, indice énergie FMI). Le défaut reste CROSS (v1).
CROSS_V2 = CROSS + ["E08", "E06", "E07", "ED03", "ED04", "E14", "E15", "E10", "E12", "ED05", "E20", "E21", "ED06",
                    "E22"]
TWIN = {"S07": "D01", "S08": "D02", "S09": "X_US_PPI_STEEL", "S10": "X_US_PPI_STEEL", "S11": "ED01", "S13": "E01"}


def _dlog(s: pd.Series, k: int) -> pd.Series:
    return np.log(s.where(s > 0)).diff(k)


# Extended set (champion_wf2): licensed market price vs index gap (Report 3's strongest signal: the EU index catches
# up with coil prices), and regime-gated momentum / mean reversion.
LIC_TWIN = {"S09": "L01", "S10": "L03", "S12": "L01", "RM6_NE_IQI_433520": "L01", "RMAirSeT_IIQ_24kV_IEC": "L01",
            "L01": "S09", "L03": "S10", "L02": "S09"}


def _extended(info: pd.DataFrame, target: str, ly: pd.Series) -> dict:
    f = {}
    twin = LIC_TWIN.get(target)
    if twin and twin in info and info[twin].notna().any():
        gap = ly - np.log(info[twin].where(info[twin] > 0)).reindex(ly.index)
        f["lic_gap"] = gap
        f["lic_gap_dev24"] = gap - gap.rolling(24, min_periods=18).mean()
    d3 = ly.diff(3)
    dev24 = ly - ly.rolling(24, min_periods=18).mean()
    # "overextended" = |dev24| above its own 80th percentile over the past 10 years (causal, rolling)
    hot = (dev24.abs() > dev24.abs().rolling(120, min_periods=60).quantile(0.8)).astype(float).where(dev24.notna())
    f["mom_calm"] = d3 * (1 - hot)
    f["dev_hot"] = dev24 * hot
    return f


def make_features(info: pd.DataFrame, target: str, cross=None, extended: bool = False) -> pd.DataFrame:
    """Toutes les features candidates pour ``target``, indexées comme ``info`` (valeurs à t calculées sur ≤ t).

    ``cross`` : liste des séries « cross_* » candidates (None -> CROSS, la v1). ``extended`` : ajoute l'écart
    prix de marché sous licence / indice et les features de régime (champion_wf2)."""
    cross = CROSS if cross is None else cross
    y = info[target]
    ly = np.log(y.where(y > 0))
    f = {
        "own_d1": ly.diff(1), "own_d3": ly.diff(3), "own_d12": ly.diff(12),
        "own_accel": ly.diff(1) - ly.diff(3) / 3,
        "own_dev12": ly - ly.rolling(12).mean(), "own_dev60": ly - ly.rolling(60, min_periods=36).mean(),
        "overext36": ly - ly.rolling(36, min_periods=24).max(),
        "vol_regime": ly.diff().rolling(3).std() / ly.diff().rolling(36, min_periods=24).std(),
    }
    for c in cross:
        if c in info and c != target:
            f[f"cross_{c}_d1"] = _dlog(info[c], 1)
            f[f"cross_{c}_d3"] = _dlog(info[c], 3)
    for c in PRICE_LIKE_EXT:
        if c in info:
            f[f"{c}_d1"] = _dlog(info[c], 1)
            f[f"{c}_d3"] = _dlog(info[c], 3)
    for c in LEVEL_EXT:
        if c in info:
            s = info[c]
            f[f"{c}_z"] = (s - s.rolling(36, min_periods=24).mean()) / s.rolling(36, min_periods=24).std()
            f[f"{c}_ch3"] = s.diff(3)
    mats = [m for m in MATERIALS if m in info]
    f["breadth3"] = (np.log(info[mats]).diff(3) > 0).sum(axis=1).where(info[mats].notna().sum(axis=1) >= 8) / len(mats)
    if "X_US_PPI_ALL" in info:
        real = ly - np.log(info["X_US_PPI_ALL"])
        f["real_dev120"] = real - real.rolling(120, min_periods=60).mean()
    twin = TWIN.get(target)
    if twin and twin in info:
        gap = ly - np.log(info[twin].where(info[twin] > 0))
        f["twin_gap"] = gap - gap.rolling(24, min_periods=18).mean()
    if {"X_US_PPI_CRUDE", "X_US_PPI_INTERMED"} <= set(info):
        f["upstream3"] = _dlog(info["X_US_PPI_CRUDE"], 3) - _dlog(info["X_US_PPI_INTERMED"], 3)
    if {"X_US_STEEL_SCRAP", "X_US_PPI_STEEL"} <= set(info):
        f["scrap_spread3"] = _dlog(info["X_US_STEEL_SCRAP"], 3) - _dlog(info["X_US_PPI_STEEL"], 3)
    for c in ["X_CU_TERM", "X_AL_TERM"]:
        if c in info:
            f[c.lower().replace("x_", "")] = info[c]
    for c in ["X_CU_STOCKS", "X_AL_STOCKS"]:
        if c in info:
            f[c.lower().replace("x_", "") + "_d3"] = _dlog(info[c], 3)
    if extended:
        f.update(_extended(info, target, ly))
    return pd.DataFrame(f, index=info.index)
