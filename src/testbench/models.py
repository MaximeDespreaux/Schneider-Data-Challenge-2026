"""Interface commune des modèles + baselines.

Contrat (à respecter par TOUS les modèles, y compris ceux de la partie 3) :

    model.forecast(y, steps, exog=None) -> DataFrame
        y     : pd.Series mensuelle (DatetimeIndex début de mois), SANS NaN, qui se termine à la dernière
                observation connue. Valeurs en niveau (EUR/t ou indice), strictement positives.
        steps : liste d'entiers >= 1 = nombre de mois après la dernière observation de y.
        exog  : (optionnel) tout le panel tel que connu à l'origine (déjà tronqué, sans fuite possible).
        retour: DataFrame indexé par steps, colonnes "p10", "p50", "p90" (niveaux), p10 <= p50 <= p90.

Pour brancher un modèle « point » (une seule valeur), il suffit d'écrire ``point(y, h) -> float`` et de
l'envelopper dans ``Conformal`` : les P10/P90 sont alors tirés de ses propres erreurs passées.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

Z90 = 1.2815515655446004  # quantile 90 % de la loi normale


def _frame(steps, p10, p50, p90) -> pd.DataFrame:
    df = pd.DataFrame({"p10": p10, "p50": p50, "p90": p90}, index=pd.Index(steps, name="step"))
    # garantit l'ordre des quantiles
    df["p10"] = np.minimum(df["p10"], df["p50"])
    df["p90"] = np.maximum(df["p90"], df["p50"])
    return df


def h_log_changes(y: pd.Series, h: int) -> np.ndarray:
    ly = np.log(y.to_numpy(dtype=float))
    return ly[h:] - ly[:-h] if len(ly) > h else np.array([])


# ----------------------------------------------------------------------------- baselines
@dataclass
class Naive:
    """« Le prix reste celui d'aujourd'hui ». Bandes = quantiles des variations passées à h mois, recentrées."""

    name: str = "naive"
    window: int | None = None  # None = tout l'historique

    def forecast(self, y, steps, exog=None):
        yy = y if self.window is None else y.iloc[-(self.window + max(steps)):]
        last = float(y.iloc[-1])
        p10, p90 = [], []
        for h in steps:
            d = h_log_changes(yy, h)
            if len(d) < 8:
                d = h_log_changes(y, 1) * np.sqrt(h)
            med = np.median(d)
            p10.append(last * np.exp(np.quantile(d, 0.10) - med))
            p90.append(last * np.exp(np.quantile(d, 0.90) - med))
        return _frame(steps, p10, [last] * len(steps), p90)


@dataclass
class NaiveGaussian:
    """Naïf avec bandes « marche aléatoire gaussienne » : sigma mensuel × racine(h). Sert de contre-exemple
    (sous-estime les queues épaisses)."""

    name: str = "naive_gauss"
    window: int = 60

    def forecast(self, y, steps, exog=None):
        last = float(y.iloc[-1])
        s = np.std(h_log_changes(y.iloc[-(self.window + 1):], 1), ddof=1)
        return _frame(steps, [last * np.exp(-Z90 * s * np.sqrt(h)) for h in steps], [last] * len(steps),
                      [last * np.exp(Z90 * s * np.sqrt(h)) for h in steps])


@dataclass
class NaiveVolScaled:
    """Naïf dont les bandes s'élargissent quand le marché s'agite.

    On standardise les variations passées à h mois par la volatilité (EWMA) du moment où elles ont eu lieu,
    on prend leurs quantiles, puis on les remet à l'échelle de la volatilité actuelle. En période calme les
    bandes se resserrent, en période de choc elles s'élargissent tout de suite (au lieu d'attendre que le choc
    entre dans l'historique)."""

    name: str = "naive_vol"
    # demi-vie EWMA en mois. Choisie parmi {3, 6, 12} en ne regardant QUE les prévisions dont la cible est
    # avant 2024 (perte quantile la plus basse) ; confirmée ensuite sur 2024-26 (encore la meilleure).
    halflife: float = 3.0
    floor: float = 0.5  # la vol actuelle ne descend pas sous 50 % de la vol longue (évite des bandes trop fines)
    # recenter=True : bandes symétriques autour du prix actuel (on retire la médiane des variations passées).
    # recenter=False : on garde l'asymétrie historique (les prix ont plus souvent monté que baissé) — corrige
    # le P90 trop optimiste, le P50 reste le prix actuel.
    recenter: bool = True

    def __post_init__(self):
        if not self.recenter and self.name == "naive_vol":
            self.name = "naive_vol_skew"

    def forecast(self, y, steps, exog=None):
        last = float(y.iloc[-1])
        d1 = pd.Series(h_log_changes(y, 1), index=y.index[1:])
        vol = np.sqrt((d1 ** 2).ewm(halflife=self.halflife, min_periods=6).mean())
        long_vol = float(d1.std(ddof=1))
        cur = max(float(vol.iloc[-1]), self.floor * long_vol)
        p10, p90 = [], []
        for h in steps:
            ly = np.log(y)
            dh = (ly.shift(-h) - ly).dropna()
            v = vol.reindex(dh.index).dropna()
            z = (dh.reindex(v.index) / (v * np.sqrt(h))).to_numpy()
            if len(z) < 12:
                z = np.array([-Z90, 0.0, Z90])
            med = np.median(z) if self.recenter else 0.0
            p10.append(last * np.exp((np.quantile(z, 0.10) - med) * cur * np.sqrt(h)))
            p90.append(last * np.exp((np.quantile(z, 0.90) - med) * cur * np.sqrt(h)))
        return _frame(steps, p10, [last] * len(steps), p90)


# ----------------------------------------------------------------------------- modèles « point »
@dataclass
class DriftPoint:
    """Naïf + tendance : prolonge la variation mensuelle moyenne des ``window`` derniers mois."""

    name: str = "drift"
    window: int = 60

    def point(self, y: pd.Series, h: int) -> float:
        d = h_log_changes(y.iloc[-(self.window + 1):], 1)
        return float(y.iloc[-1] * np.exp(h * d.mean()))


@dataclass
class AR1Point:
    """Momentum : AR(1) sur les log-variations mensuelles, itéré h pas (capte l'inertie des indices PPI)."""

    name: str = "ar1"
    window: int = 60

    def point(self, y: pd.Series, h: int) -> float:
        d = h_log_changes(y.iloc[-(self.window + 1):], 1)
        if len(d) < 12:
            return float(y.iloc[-1])
        X = np.column_stack([np.ones(len(d) - 1), d[:-1]])
        c, phi = np.linalg.lstsq(X, d[1:], rcond=None)[0]
        phi = float(np.clip(phi, -0.95, 0.95))
        cum, last_d = 0.0, d[-1]
        for _ in range(h):
            last_d = c + phi * last_d
            cum += last_d
        return float(y.iloc[-1] * np.exp(cum))


@dataclass
class MeanReversionPoint:
    """Retour vers la moyenne mobile 12 mois (contre-hypothèse du momentum)."""

    name: str = "ma12"
    window: int = 12

    def point(self, y: pd.Series, h: int) -> float:
        return float(np.exp(np.log(y.iloc[-self.window:]).mean()))


@dataclass
class Conformal:
    """Transforme un modèle point en modèle probabiliste.

    À chaque appel, on rejoue le modèle sur ``n_cal`` pseudo-origines passées (toutes antérieures à la
    dernière observation, donc sans fuite), on mesure ses erreurs en log à h mois et on prend leurs
    quantiles 10 % / 90 %. Coût = n_cal appels de ``point`` par horizon : à réserver aux modèles rapides.
    """

    base: object
    n_cal: int = 48
    min_cal: int = 18
    name: str = field(default="")

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.base.name}_conf"

    def forecast(self, y, steps, exog=None):
        p10, p50, p90 = [], [], []
        for h in steps:
            pt = self.base.point(y, h)
            errs = []
            last_tau = len(y) - 1 - h  # position de la dernière pseudo-origine dont la cible est connue
            for k in range(last_tau, max(last_tau - self.n_cal, 24) - 1, -1):
                hist = y.iloc[: k + 1]
                errs.append(np.log(y.iloc[k + h]) - np.log(self.base.point(hist, h)))
            if len(errs) < self.min_cal:
                s = np.std(h_log_changes(y, 1), ddof=1) * np.sqrt(h)
                lo, hi = -Z90 * s, Z90 * s
            else:
                lo, hi = np.quantile(errs, 0.10), np.quantile(errs, 0.90)
            p10.append(pt * np.exp(min(lo, 0.0)))
            p50.append(pt)
            p90.append(pt * np.exp(max(hi, 0.0)))
        return _frame(steps, p10, p50, p90)


def baseline_models() -> list:
    """Jeu de référence du banc de test. Le premier (naive) est la référence de toutes les comparaisons."""
    # NaiveVolScaled(recenter=False) a été testé (19/11) pour corriger le P90 trop optimiste : aucun gain
    # (dépassements au-dessus du P90 inchangés, perte quantile pire à 9 mois) -> retiré du jeu par défaut.
    return [Naive(), NaiveGaussian(), NaiveVolScaled(), Conformal(DriftPoint()), Conformal(AR1Point()),
            Conformal(MeanReversionPoint())]
