"""Ce qui était connu à une date d'origine donnée.

Deux modes :
- ``submission`` (défaut) : l'origine est « le dernier mois de données utilisé », comme dans le CSV de Schneider
  (origin_month = 2026-09). Toutes les séries du panel sont connues jusqu'à l'origine incluse.
- ``realtime`` : on se place à la date de décision ``origin`` et on retire à chaque série son délai de
  publication (ex. Eurostat ≈ 2 mois). Plus réaliste pour un acheteur ; sert de test de robustesse.

Dans les deux cas, AUCUNE ligne postérieure à l'origine n'est transmise au modèle.
"""

from __future__ import annotations

import pandas as pd

# Délais de publication en mois, mesurés le 12/11/2026 (date de téléchargement) : la plupart des séries
# s'arrêtent en sept. 2026 => ~2 mois ; FX, électricité, BoJ en oct. => ~1 mois ; imports Eurostat juil. => 4.
REALTIME_LAG_MONTHS: dict[str, int] = {
    **{s: 2 for s in ["S01", "S02", "S03", "S04", "S05", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "D01", "D02"]},
    "S06": 1,
    **{f"E{i:02d}": 2 for i in range(1, 25)},
    **{e: 1 for e in ["E04", "E09", "E10", "E11", "E12", "E19", "E23", "E24", "ED05"]},
    **{e: 4 for e in ["E20", "E21", "ED06"]},
    **{e: 2 for e in ["ED01", "ED02", "ED03", "ED04"]},
}


def month_add(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    return (pd.Timestamp(ts).to_period("M") + n).to_timestamp()


def months_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """Nombre de mois de a vers b (b - a)."""
    return (pd.Timestamp(b).to_period("M") - pd.Timestamp(a).to_period("M")).n


def information_set(panel: pd.DataFrame, origin: pd.Timestamp, mode: str = "submission",
                    lags: dict[str, int] | None = None) -> pd.DataFrame:
    """Copie du panel telle qu'elle était connue à ``origin``.

    Les lignes après l'origine sont supprimées ; en mode realtime, les ``lag`` derniers mois de chaque
    série sont en plus masqués (NaN).
    """
    origin = pd.Timestamp(origin)
    info = panel.loc[:origin].copy()
    if mode == "submission":
        return info
    if mode != "realtime":
        raise ValueError(f"mode inconnu : {mode}")
    lags = REALTIME_LAG_MONTHS if lags is None else lags
    for col in info.columns:
        lag = lags.get(col, 0)
        if lag > 0:
            info.loc[info.index > month_add(origin, -lag), col] = float("nan")
    return info
