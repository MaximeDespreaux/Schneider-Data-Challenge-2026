"""Proposition pour la partie 3 : régression sur « drivers avancés », branchée sur le banc de test.

Idée métier (cf. rapports de l'équipe) : les indices Eurostat suivent les prix de marché avec retard (acier ← minerai,
énergie ; plastiques ← pétrole ; alu ← LME alu + gaz ; cuivre EU ← LME cuivre). On prévoit la variation en log à h mois
d'une série à partir de :
  - sa propre dynamique récente (variation 1 et 3 mois, écart à sa moyenne 12 mois),
  - les variations récentes (1 et 3 mois) de ses drivers, TELLES QUE CONNUES À L'ORIGINE (exog est tronqué),
  - pour les indices qui ont un « prix de marché » jumeau : l'écart indice/marché vs sa moyenne 24 mois
    (l'indice « rattrape » le marché).
Ridge SANS constante sur variables standardisées → en l'absence de signal, la prévision retombe sur « pas de
changement » (le naïf). Fourchettes : quantiles des résidus d'entraînement à h mois (légèrement optimistes,
à recalibrer — voir `naive_vol`).

Uniquement des données gratuites du panel. L'acier MEPS/CRU (L01-L03), l'epoxy (EP01) et l'électronique (EL01) sont
désormais dans le panel et servent de drivers / cibles comme indiqué dans DRIVERS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .models import _frame

DRIVERS: dict[str, list[str]] = {
    "S07": ["D01", "S05"],
    "S08": ["D02", "ED02", "E13"],
    "S09": ["L01", "S04", "S03", "ED02", "E13", "D02"],
    "S10": ["L03", "S04", "S05", "ED02", "E13"],
    "S11": ["ED01", "ED02"],
    "S12": ["S04", "E18"],
    "S13": ["E01", "E06"],
    "S01": ["S05", "ED01"], "D01": ["S05", "ED01"],
    "S02": ["ED02", "E09"], "D02": ["ED02", "E09"],
    "S03": ["ED01"], "S04": ["ED01"], "S05": ["D01"],
    "RM6_NE_IQI_433520": ["D01", "L01", "S05", "ED02", "E13"],
    "RMAirSeT_IIQ_24kV_IEC": ["L01", "L02", "D01", "ED02", "E13"],
    "L01": ["S04", "ED02", "E13"],
    "L03": ["S04", "S05", "ED02", "E13"],
    "L02": ["S03", "ED02", "E13"],
    "EP01": ["ED01", "ED02"],
    "EL01": ["D01", "E13"],
}
MARKET_TWIN = {"S07": "D01", "S08": "D02", "S09": "L01", "S10": "L03"}


def _ridge_no_intercept(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)


@dataclass
class DriverRidge:
    name: str = "driver_ridge"
    alpha: float = 5.0
    window: int = 72  # mois d'historique pour l'apprentissage
    drivers: dict = field(default_factory=lambda: dict(DRIVERS))

    def _features(self, y: pd.Series, exog: pd.DataFrame | None, target: str) -> pd.DataFrame:
        ly = np.log(y)
        f = pd.DataFrame({"own_d1": ly.diff(1), "own_d3": ly.diff(3), "own_dev12": ly - ly.rolling(12).mean()})
        if exog is not None:
            for dname in self.drivers.get(target, []):
                if dname not in exog:
                    continue
                lx = np.log(exog[dname].where(exog[dname] > 0)).reindex(y.index.union(exog.index)).ffill()
                f[f"{dname}_d1"] = lx.diff(1).reindex(y.index)
                f[f"{dname}_d3"] = lx.diff(3).reindex(y.index)
            twin = MARKET_TWIN.get(target)
            if twin and twin in exog:
                gap = ly - np.log(exog[twin]).reindex(y.index).ffill()
                f["gap_vs_market"] = gap - gap.rolling(24).mean()
        return f

    def forecast(self, y, steps, exog=None):
        target = y.name
        feats = self._features(y, exog, target)
        ly = np.log(y)
        last_x = feats.iloc[[-1]]
        p10, p50, p90 = [], [], []
        for h in steps:
            tgt = (ly.shift(-h) - ly).rename("tgt")  # uniquement à l'intérieur de l'historique connu
            df = feats.join(tgt).iloc[-(self.window + h):].dropna()
            ok = last_x.notna().all().to_numpy()
            cols = feats.columns[ok]
            if len(df) < 24 or len(cols) == 0:
                d = (ly.shift(-h) - ly).dropna()
                q = np.quantile(d - d.median(), [0.1, 0.9]) if len(d) > 8 else (0.0, 0.0)
                p10.append(y.iloc[-1] * np.exp(q[0])); p50.append(y.iloc[-1]); p90.append(y.iloc[-1] * np.exp(q[1]))
                continue
            X, Y = df[cols].to_numpy(), df["tgt"].to_numpy()
            mu, sd = X.mean(0), X.std(0) + 1e-12
            beta = _ridge_no_intercept((X - mu) / sd, Y, self.alpha * len(Y) / 50)
            xnow = (last_x[cols].to_numpy()[0] - mu) / sd
            pred = float(xnow @ beta)
            res = Y - ((X - mu) / sd) @ beta
            lo, hi = np.quantile(res, [0.10, 0.90])
            base = float(y.iloc[-1])
            p50.append(base * np.exp(pred))
            p10.append(base * np.exp(pred + min(lo, 0)))
            p90.append(base * np.exp(pred + max(hi, 0)))
        return _frame(steps, p10, p50, p90)
