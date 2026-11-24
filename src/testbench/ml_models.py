"""Modèle ML de référence : gradient boosting quantile (LightGBM), au format du banc de test.

Pour chaque série et chaque horizon, trois modèles LightGBM (quantiles 10 %, 50 %, 90 %) apprennent la variation en
log à h mois à partir de la dynamique propre de la série, calculée uniquement sur l'historique connu à l'origine :
variations sur 1, 3, 6 et 12 mois, écart à la moyenne 12 mois, volatilité 6 mois.

Volontairement petit (peu d'arbres, feuilles peu profondes, beaucoup d'observations par feuille) : avec ~100 à 400
mois d'historique, un modèle plus riche sur-apprend (constat du rapport 2 de l'équipe : ×1,33 à 9 mois).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import NaiveVolScaled, _frame

QUANTILES = (0.10, 0.50, 0.90)


def own_features(y: pd.Series) -> pd.DataFrame:
    ly = np.log(y.astype(float))
    d1 = ly.diff()
    return pd.DataFrame({"d1": d1, "d3": ly.diff(3), "d6": ly.diff(6), "d12": ly.diff(12),
                         "dev12": ly - ly.rolling(12).mean(), "vol6": d1.rolling(6).std()}, index=y.index)


@dataclass
class LGBMQuantile:
    name: str = "lgbm_q"
    window: int = 240         # mois d'historique au plus pour l'apprentissage
    min_obs: int = 60         # sinon : naive_vol
    n_estimators: int = 150
    learning_rate: float = 0.05
    num_leaves: int = 7
    min_child_samples: int = 20

    def _fit_predict(self, X, Y, xnow, alpha):
        import lightgbm as lgb
        m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=self.n_estimators,
                              learning_rate=self.learning_rate, num_leaves=self.num_leaves,
                              min_child_samples=self.min_child_samples, subsample=1.0, colsample_bytree=1.0,
                              random_state=0, n_jobs=1, verbose=-1)
        m.fit(X, Y)
        return float(m.predict(xnow)[0])

    def forecast(self, y, steps, exog=None):
        fallback = NaiveVolScaled().forecast(y, steps)
        feats = own_features(y)
        ly = np.log(y.astype(float))
        xnow = feats.iloc[[-1]]
        if xnow.isna().any(axis=None):
            return fallback
        last = float(y.iloc[-1])
        p10, p50, p90 = [], [], []
        for h in steps:
            df = feats.join((ly.shift(-h) - ly).rename("fut")).dropna().iloc[-self.window:]
            if len(df) < self.min_obs:
                r = fallback.loc[h]
                p10.append(r.p10); p50.append(r.p50); p90.append(r.p90)
                continue
            X, Y = df[feats.columns], df["fut"].to_numpy()
            q = sorted(np.clip([self._fit_predict(X, Y, xnow, a) for a in QUANTILES], -0.5, 0.5))  # pas de croisement
            p10.append(last * np.exp(q[0])); p50.append(last * np.exp(q[1])); p90.append(last * np.exp(q[2]))
        return _frame(steps, p10, p50, p90)
