"""Moteur d'origines glissantes (« rejouer l'histoire »)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .vintage import information_set, month_add, months_between

HORIZONS = (3, 6, 9)


def default_origins(panel: pd.DataFrame, start: str = "2020-01", end: str | None = None) -> pd.DatetimeIndex:
    """Toutes les origines mensuelles de ``start`` au dernier mois du panel (les cibles futures seront
    simplement absentes du scoring)."""
    end = panel.index.max() if end is None else pd.Timestamp(end)
    return pd.date_range(start, end, freq="MS")


def run_backtest(panel: pd.DataFrame, targets: list[str], models: list, origins=None,
                 horizons=HORIZONS, mode: str = "submission", min_history: int = 36,
                 verbose: bool = True, derive=None, lags: dict[str, int] | None = None) -> pd.DataFrame:
    """Rejoue chaque modèle sur chaque (origine, série, horizon).

    ``derive`` (optionnel) : fonction ``derive(values_df, bases_df, origin) -> DataFrame`` qui fabrique des
    séries dérivées (ex. indices BOM) À CHAQUE ORIGINE, à partir de ce qui est connu à cette origine
    (cf. ``targets.make_bom_deriver``). L'historique dérivé est calculé sur l'info tronquée ; la vérité sur le
    panel complet mais avec les MÊMES poids/bases (fixés à l'origine).

    ``lags`` : masques par colonne passés à ``information_set`` (bord irrégulier en submission, délais de
    publication en realtime ; cf. ``availability.py``). Un modèle qui a un attribut ``targets`` n'est lancé que sur
    ces séries.

    Retour (format long, une ligne par prévision) :
        model, series, origin, horizon, target_month, last_obs, step, p10, p50, p90, y_true
    ``y_true`` = valeur du panel complet au mois cible (NaN si pas encore connue). Attention : c'est la
    valeur *révisée* d'aujourd'hui, pas la première publication (les indices Eurostat/BLS sont révisés).
    """
    origins = default_origins(panel) if origins is None else pd.DatetimeIndex(origins)
    rows = []
    t0 = time.time()
    runtime = {m.name: 0.0 for m in models}
    for i, origin in enumerate(origins):
        info = information_set(panel, origin, mode=mode, lags=lags)
        truth = panel
        if derive is not None:
            hist = derive(info, info, origin)
            truth = panel.join(derive(panel, info, origin))
            info = info.join(hist)
        for s in targets:
            y = info[s].dropna()
            if len(y) < min_history:
                continue
            last = y.index[-1]
            tgt = {h: month_add(origin, h) for h in horizons}
            steps = {h: months_between(last, tgt[h]) for h in horizons}
            uniq = sorted(set(steps.values()))
            for m in models:
                if getattr(m, "targets", None) is not None and s not in m.targets:
                    continue
                tm = time.perf_counter()
                fc = m.forecast(y, uniq, exog=info)
                runtime[m.name] += time.perf_counter() - tm
                for h in horizons:
                    st = steps[h]
                    r = fc.loc[st]
                    yt = truth[s].get(tgt[h], np.nan)
                    rows.append((m.name, s, origin, h, tgt[h], last, st, r.p10, r.p50, r.p90, yt))
        if verbose and (i % 12 == 0 or i == len(origins) - 1):
            print(f"  origine {origin:%Y-%m} ({i + 1}/{len(origins)}) – {time.time() - t0:.0f}s", flush=True)
    out = pd.DataFrame(rows, columns=["model", "series", "origin", "horizon", "target_month", "last_obs", "step",
                                      "p10", "p50", "p90", "y_true"])
    _check(out)
    out.attrs["runtime_s"] = runtime  # temps de calcul par modèle (pour l'estimation énergie / CO2)
    return out


def compute_footprint(runtime_s: dict[str, float], watts: float = 30.0, kg_co2_per_kwh: float = 0.25) -> "pd.DataFrame":
    """Estimation énergie / CO2 du backtest (critère 5 « compute & CO2 »).
    Hypothèses : portable ~30 W en charge CPU, mix électrique UE ~0,25 kgCO2e/kWh (ordre de grandeur)."""
    df = pd.DataFrame({"runtime_s": runtime_s}).rename_axis("model")
    df["kWh"] = df["runtime_s"] * watts / 3.6e6
    df["gCO2e"] = df["kWh"] * kg_co2_per_kwh * 1000
    df.loc["TOTAL"] = df.sum()
    return df.round(6)


def _check(df: pd.DataFrame) -> None:
    """Garde-fous : aucune information postérieure à l'origine, quantiles ordonnés."""
    assert (df["last_obs"] <= df["origin"]).all(), "fuite : une observation postérieure à l'origine a été utilisée"
    assert (df["p10"] <= df["p50"] + 1e-9).all() and (df["p50"] <= df["p90"] + 1e-9).all(), "quantiles non ordonnés"
    assert np.isfinite(df[["p10", "p50", "p90"]].to_numpy()).all(), "prévisions non finies"
