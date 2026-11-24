"""Ce qui était connu à une date d'origine donnée.

Deux modes :
- ``submission`` (défaut) : l'origine est « le dernier mois de données utilisé », comme dans le CSV de Schneider
  (origin_month = 2026-09). Toutes les séries du panel sont connues jusqu'à l'origine incluse.
- ``realtime`` : on se place à la date de décision ``origin`` et on retire à chaque série son délai de
  publication (ex. Eurostat ≈ 2 mois). Plus réaliste pour un acheteur ; sert de test de robustesse.
Les délais viennent tous de ``config/availability.yaml`` (cf. ``availability.py``).

Dans les deux cas, AUCUNE ligne postérieure à l'origine n'est transmise au modèle.
"""

from __future__ import annotations

import pandas as pd

def month_add(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    return (pd.Timestamp(ts).to_period("M") + n).to_timestamp()


def months_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """Nombre de mois de a vers b (b - a)."""
    return (pd.Timestamp(b).to_period("M") - pd.Timestamp(a).to_period("M")).n


def information_set(panel: pd.DataFrame, origin: pd.Timestamp, mode: str = "submission",
                    lags: dict[str, int] | None = None) -> pd.DataFrame:
    """Copie du panel telle qu'elle était connue à ``origin``.

    Les lignes après l'origine sont supprimées, puis les ``lags[col]`` derniers mois de chaque colonne sont
    masqués (NaN), dans les deux modes :
    - submission : ``lags`` = bord irrégulier du live (``availability.ragged_edge_lags``) ; None = aucun masque ;
    - realtime   : ``lags`` = délais de publication ; None = table ``config/availability.yaml`` (toute colonne
      absente de la table lève une erreur, jamais de délai 0 par défaut).
    """
    origin = pd.Timestamp(origin)
    info = panel.loc[:origin].copy()
    if mode not in ("submission", "realtime"):
        raise ValueError(f"mode inconnu : {mode}")
    if lags is None:
        if mode == "submission":
            return info
        from .availability import check_coverage, realtime_lags
        check_coverage(info.columns)
        lags = realtime_lags()
    for col in info.columns:
        lag = lags.get(col, 0)
        if lag > 0:
            info.loc[info.index > month_add(origin, -lag), col] = float("nan")
    return info
