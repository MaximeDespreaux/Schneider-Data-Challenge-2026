"""Criblage des signaux : lesquels annoncent vraiment la variation à h mois ?

- ``screen``    : criblage GLOBAL, pour l'exploration seulement (il utilise la coupure 2024 déjà inspectée ;
  ne jamais s'en servir pour une prévision notée) ;
- ``screen_at`` : criblage walk-forward, relancé à chaque origine sur les seules cibles réalisées (``ChampionWF``).

Règles anti-hasard (et anti-fuite) de ``screen`` :
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


def _test_signal(x: pd.Series, fut: pd.Series, h: int, periods=None, min_rows: int = 60) -> dict | None:
    """Corrélation du signal ``x`` (à t) avec la variation ``fut`` (t -> t+h), partagée par ``screen`` et ``screen_at``.

    Stabilité : même signe dans les deux sous-périodes ``periods`` [(début, fin), (début, fin)] (None = première
    et seconde moitié des lignes) et sur l'ensemble ; gardé si stable et |corr| > 2 / racine(n / h).
    None si moins de ``min_rows`` paires."""
    df = pd.concat([x, fut], axis=1).dropna()
    if len(df) < min_rows:
        return None
    corr = df.corr().iloc[0, 1]
    m = len(df) // 2
    parts = [df.iloc[:m], df.iloc[m:]] if periods is None else [df.loc[a:b] for a, b in periods]
    sub = [p.corr().iloc[0, 1] if len(p) > 24 else np.nan for p in parts]
    stable = np.isfinite(sub).all() and np.sign(sub[0]) == np.sign(sub[1]) == np.sign(corr)
    thr = 2 / np.sqrt(len(df) / h)
    return {"corr": corr, "sub": sub, "n": len(df), "threshold": thr, "keep": bool(stable and abs(corr) > thr)}


def screen_at(info: pd.DataFrame, target: str, horizons=(3, 6, 9), max_keep: int = 6,
              min_rows: int = 60, extended: bool = False) -> dict[tuple[str, int], list[str]]:
    """Criblage walk-forward : {(target, h): signaux retenus, plus fort |corr| d'abord, au plus ``max_keep``}.

    N'utilise que les paires (signal à τ, variation τ -> τ+h) avec τ+h <= dernier mois valide de ``info[target]`` :
    à une origine, seules les cibles déjà réalisées comptent. Sous-périodes de stabilité = première et seconde
    moitié de ces paires. Même règle que ``screen`` (``_test_signal``)."""
    feats = make_features(info, target, extended=extended)
    ly = np.log(info[target])
    last = info[target].last_valid_index()
    out = {}
    for h in horizons:
        fut = ly.shift(-h) - ly
        ok = fut.index <= (last.to_period("M") - h).to_timestamp()
        kept = {}
        for c in feats.columns:
            r = _test_signal(feats[c].loc[ok], fut.loc[ok], h, min_rows=min_rows)
            if r is not None and r["keep"]:
                kept[c] = abs(r["corr"])
        out[(target, h)] = sorted(kept, key=kept.get, reverse=True)[:max_keep]
    return out


def screen(panel: pd.DataFrame | None = None, horizons=(3, 6, 9), max_keep: int = 6, cross=None) -> pd.DataFrame:
    """Criblage GLOBAL sur les cibles antérieures à 2024. EXPLORATION SEULEMENT : utilise la coupure 2024 déjà
    inspectée (et des poids BOM ancrés en 2026-09) ; les prévisions notées passent par ``screen_at``."""
    panel = build_panel() if panel is None else panel
    # indices BOM (poids fixés au mois d'ancrage : uniquement pour le CRIBLAGE descriptif pré-2024 ;
    # le backtest, lui, les reconstruit à chaque origine)
    der = bom_deriver(panel)
    panel = panel.join(der(panel, panel, pd.Timestamp("2026-09-01")))
    rows = []
    for tgt in DEFAULT_TARGETS:
        feats = make_features(panel, tgt, cross=cross)
        ly = np.log(panel[tgt])
        for h in horizons:
            fut = ly.shift(-h) - ly
            ok = fut.index + pd.DateOffset(months=h) < SPLIT
            for c in feats.columns:
                if c.startswith("own_") and c != "own_dev60":
                    pass
                r = _test_signal(feats[c].loc[ok], fut.loc[ok], h, PERIODS)
                if r is None:
                    continue
                rows.append({"target": tgt, "horizon": h, "feature": c, "corr": r["corr"], "corr_2003_12": r["sub"][0],
                             "corr_2013_23": r["sub"][1], "n": r["n"], "threshold": r["threshold"], "keep": r["keep"]})
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
