"""Modèles qui utilisent les signaux externes, au format du banc de test.

- SignalRidge : régression ridge (sans constante -> retombe sur « pas de changement » sans signal) de la variation
  à h mois sur les signaux retenus par le criblage (src/external/screen.py, cibles < 2024 uniquement).
- ChampionWF  : Champion dont le criblage est refait à chaque origine (``screen.screen_at``).
- Hybrid      : prend la valeur centrale (P50) d'un modèle et la forme de fourchette d'un autre (`naive_vol`),
  car les fourchettes issues des résidus de régression sont trop étroites.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.external.features import make_features
from src.external.screen import screen_at
from src.testbench.models import _frame

HORIZONS = (3, 6, 9)


@dataclass
class SignalRidge:
    selected: dict = field(default_factory=dict)  # {(target, h): [features]}
    name: str = "signal_ridge"
    alpha_per_obs: float = 0.5
    window: int = 180
    min_obs: int = 48
    always: tuple = ("own_d1",)  # toujours inclus (momentum)
    cross: tuple | None = None  # séries cross_* candidates (None -> features.CROSS, la v1)
    extended: bool = False      # features étendues (champion_wf2)

    def point_frame(self, y: pd.Series, steps, exog: pd.DataFrame) -> pd.DataFrame:
        target = y.name
        feats = make_features(exog, target, cross=self.cross, extended=self.extended).reindex(y.index)
        ly = np.log(y)
        base = float(y.iloc[-1])
        out = {}
        for st in steps:
            h = min(HORIZONS, key=lambda k: abs(k - st))
            cols = list(dict.fromkeys(list(self.always) + self.selected.get((target, h), [])))
            cols = [c for c in cols if c in feats and pd.notna(feats[c].iloc[-1])]
            if not self.selected.get((target, h)) or not cols:
                out[st] = base
                continue
            fut = (ly.shift(-st) - ly).rename("fut")
            df = feats[cols].join(fut).dropna().iloc[-self.window:]
            if len(df) < self.min_obs:
                out[st] = base
                continue
            X, Y = df[cols].to_numpy(), df["fut"].to_numpy()
            mu, sd = X.mean(0), X.std(0) + 1e-12
            Z = (X - mu) / sd
            beta = np.linalg.solve(Z.T @ Z + self.alpha_per_obs * len(Y) * np.eye(len(cols)), Z.T @ Y)
            pred = float(((feats[cols].iloc[-1].to_numpy() - mu) / sd) @ beta)
            out[st] = base * np.exp(np.clip(pred, -0.5, 0.5))
        return pd.Series(out)

    def forecast(self, y, steps, exog=None):
        p = self.point_frame(y, steps, exog)
        return _frame(steps, p.values, p.values, p.values)


@dataclass
class Champion:
    """Modèle final, avec une règle de choix DÉCIDÉE AVANT de regarder 2024-26 :
    pour chaque (série, horizon), si le criblage pré-2024 a retenu au moins ``min_signals`` signaux stables,
    P50 = régression sur signaux (version prudente, très régularisée) ; sinon P50 = prix actuel (naïf).
    Fourchette : toujours `naive_vol` (s'élargit quand le marché s'agite)."""

    selected: dict
    min_signals: int = 3
    name: str = "champion"
    cross: tuple | None = None  # séries cross_* candidates (None -> features.CROSS, la v1)
    extended: bool = False

    def __post_init__(self):
        from src.testbench.models import NaiveVolScaled
        self.bands = NaiveVolScaled()
        self.ridge = SignalRidge(selected=self.selected, alpha_per_obs=2.0, cross=self.cross, extended=self.extended)

    def uses_signals(self, target: str, step: int) -> bool:
        h = min(HORIZONS, key=lambda k: abs(k - step))
        return len(self.selected.get((target, h), [])) >= self.min_signals

    def forecast(self, y, steps, exog=None):
        b = self.bands.forecast(y, steps, exog=exog)
        base = float(y.iloc[-1])
        sig_steps = [s for s in steps if self.uses_signals(y.name, s)]
        pts = self.ridge.point_frame(y, sig_steps, exog) if sig_steps else pd.Series(dtype=float)
        p50 = np.array([pts.get(s, base) for s in steps], dtype=float)
        return _frame(steps, p50 * b["p10"] / b["p50"], p50, p50 * b["p90"] / b["p50"])


@dataclass
class ChampionWF:
    """Champion avec criblage WALK-FORWARD : à chaque prévision, ``screen_at`` choisit les signaux sur les seules
    cibles réalisées à l'origine (jamais la coupure 2024 inspectée), puis délègue à ``Champion``."""

    min_signals: int = 3
    max_keep: int = 6
    name: str = "champion_wf"
    extended: bool = False  # champion_wf2: + licensed market-vs-index gap and regime features
    cross: tuple | None = None  # candidate cross_* series (None -> features.CROSS)

    def forecast(self, y, steps, exog=None):
        sel = screen_at(exog, y.name, max_keep=self.max_keep, extended=self.extended, cross=self.cross)
        return Champion(selected=sel, min_signals=self.min_signals, name=self.name,
                        cross=self.cross, extended=self.extended).forecast(y, steps, exog=exog)


@dataclass
class Hybrid:
    """P50 de `center`, fourchette (en relatif) de `bands`."""

    center: object
    bands: object
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.center.name}+{self.bands.name}"

    def forecast(self, y, steps, exog=None):
        c = self.center.forecast(y, steps, exog=exog)
        b = self.bands.forecast(y, steps, exog=exog)
        return _frame(steps, c["p50"] * b["p10"] / b["p50"], c["p50"], c["p50"] * b["p90"] / b["p50"])
