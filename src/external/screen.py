"""Criblage des signaux : lesquels annoncent vraiment la variation à h mois ?

Règles anti-hasard (et anti-fuite) :
- uniquement des CIBLES antérieures à 2024 (2024-26 est réservé à la vérification finale) ;
- corrélation calculée sur deux sous-périodes (2003-2012 et 2013-2023) : on ne garde un signal que si son signe
  est le même dans les deux ET si sa corrélation dépasse un seuil tenant compte du chevauchement des fenêtres
  (≈ 2 / racine du nombre de fenêtres indépendantes) sur l'ensemble.

    uv run python -m src.external.screen
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import data as d
from src.external.features import make_features
from src.external.signals import build_panel
from src.testbench.run import DEFAULT_TARGETS, bom_deriver

SPLIT = pd.Timestamp("2024-01-01")
PERIODS = [("2003-01-01", "2012-12-01"), ("2013-01-01", "2023-12-01")]


def screen(panel: pd.DataFrame | None = None, horizons=(3, 6, 9), max_keep: int = 6) -> pd.DataFrame:
    panel = build_panel() if panel is None else panel
    # indices BOM (poids fixés au mois d'ancrage : uniquement pour le CRIBLAGE descriptif pré-2024 ;
    # le backtest, lui, les reconstruit à chaque origine)
    der = bom_deriver(panel)
    panel = panel.join(der(panel, panel, pd.Timestamp("2026-09-01")))
    rows = []
    for tgt in DEFAULT_TARGETS:
        feats = make_features(panel, tgt)
        ly = np.log(panel[tgt])
        for h in horizons:
            fut = ly.shift(-h) - ly
            ok = fut.index + pd.DateOffset(months=h) < SPLIT
            for c in feats.columns:
                if c.startswith("own_") and c != "own_dev60":
                    pass
                x = feats[c]
                df = pd.concat([x, fut], axis=1).loc[ok].dropna()
                if len(df) < 60:
                    continue
                corr = df.corr().iloc[0, 1]
                n_ind = len(df) / h
                sub = []
                for a, b in PERIODS:
                    s = df.loc[a:b]
                    sub.append(s.corr().iloc[0, 1] if len(s) > 24 else np.nan)
                stable = np.isfinite(sub).all() and np.sign(sub[0]) == np.sign(sub[1]) == np.sign(corr)
                rows.append({"target": tgt, "horizon": h, "feature": c, "corr": corr, "corr_2003_12": sub[0],
                             "corr_2013_23": sub[1], "n": len(df), "threshold": 2 / np.sqrt(n_ind),
                             "keep": bool(stable and abs(corr) > 2 / np.sqrt(n_ind))})
    res = pd.DataFrame(rows)
    res["rank"] = res.groupby(["target", "horizon"])["corr"].transform(lambda s: s.abs().rank(ascending=False))
    return res


def selected(res: pd.DataFrame, max_keep: int = 6) -> dict:
    """{(target, h): [features]} : signaux retenus, les plus forts d'abord, au plus max_keep."""
    out = {}
    for (t, h), g in res[res.keep].groupby(["target", "horizon"]):
        out[(t, h)] = g.reindex(g["corr"].abs().sort_values(ascending=False).index)["feature"].head(max_keep).tolist()
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    res = screen()
    out = d.OUT / "external"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "signal_screen.csv", index=False)
    kept = res[res.keep]
    print(f"{len(res)} tests, {len(kept)} signaux retenus")
    print(kept.groupby("feature").size().sort_values(ascending=False).head(30).to_string())
    print(kept.sort_values("corr", key=abs, ascending=False).head(40)[
        ["target", "horizon", "feature", "corr", "corr_2003_12", "corr_2013_23", "threshold"]].round(2).to_string(index=False))
