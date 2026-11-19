"""Scores du banc de test. Tout est calculé en log (sans unité, comparable entre séries) :
une erreur de 0,05 en log ≈ 5 %.

- MAE %      : erreur absolue moyenne du P50, en % du prix
- rel_MAE    : MAE du modèle / MAE du naïf sur exactement les mêmes prévisions (< 1 = mieux que le naïf)
- QL         : perte quantile (pinball) moyenne sur P10/P50/P90 — récompense des bandes justes ET étroites
- rel_QL     : QL / QL du naïf
- coverage   : part des cas où la réalité tombe entre P10 et P90 (cible 80 %)
- below/above: part sous P10 / au-dessus de P90 (cible 10 % chacun) — révèle les bandes asymétriques
- width %    : largeur moyenne de la bande P10–P90
- DM p-value : test de Diebold-Mariano (erreurs absolues vs naïf, variance HAC, correction HLN)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}
SHOCK = (pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-01"))


def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def score_rows(fc: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les erreurs ligne par ligne (ne garde que les cibles connues)."""
    df = fc.dropna(subset=["y_true"]).copy()
    ly = np.log(df["y_true"])
    df["err"] = ly - np.log(df["p50"])
    df["ae"] = df["err"].abs()
    df["ql"] = sum(pinball(ly, np.log(df[c]), t) for c, t in QUANTILES.items()) / len(QUANTILES)
    df["covered"] = (df["y_true"] >= df["p10"]) & (df["y_true"] <= df["p90"])
    df["below"] = df["y_true"] < df["p10"]
    df["above"] = df["y_true"] > df["p90"]
    df["width"] = np.log(df["p90"] / df["p10"])
    in_shock = lambda t: (t >= SHOCK[0]) & (t <= SHOCK[1])  # noqa: E731
    df["regime"] = np.where(in_shock(df["origin"]) | in_shock(df["target_month"]), "choc 2021-22", "calme")
    return df


def dm_test(e_model: np.ndarray, e_ref: np.ndarray, h: int) -> tuple[float, float]:
    """Diebold-Mariano sur la différence d'erreurs absolues, variance Newey-West (h-1 retards) et
    correction de Harvey-Leybourne-Newbold. Retourne (stat, p-value bilatérale). stat < 0 = modèle meilleur."""
    d = np.abs(e_model) - np.abs(e_ref)
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return np.nan, np.nan
    dbar = d.mean()
    dc = d - dbar
    gamma = [np.dot(dc[k:], dc[: n - k]) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)  # HLN
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def summarize(scored: pd.DataFrame, by=("model", "series", "horizon"), ref: str = "naive") -> pd.DataFrame:
    """Table de scores par groupe, avec comparaison au naïf sur les mêmes lignes."""
    by = list(by)
    agg = scored.groupby(by).agg(n=("ae", "size"), MAE_pct=("ae", "mean"), RMSE_pct=("err", lambda e: np.sqrt((e ** 2).mean())),
                                 bias_pct=("err", "mean"), QL=("ql", "mean"), coverage=("covered", "mean"),
                                 below=("below", "mean"), above=("above", "mean"), width_pct=("width", "mean"))
    for c in ["MAE_pct", "RMSE_pct", "bias_pct", "width_pct"]:
        agg[c] *= 100
    if ref in scored["model"].unique() and "model" in by:
        keys = [k for k in by if k != "model"]
        refd = agg.xs(ref, level="model")[["MAE_pct", "QL"]].rename(columns={"MAE_pct": "ref_MAE", "QL": "ref_QL"})
        agg = agg.join(refd, on=keys) if keys else agg.assign(ref_MAE=refd["MAE_pct"].iloc[0], ref_QL=refd["QL"].iloc[0])
        agg["rel_MAE"] = agg["MAE_pct"] / agg["ref_MAE"]
        agg["rel_QL"] = agg["QL"] / agg["ref_QL"]
        agg = agg.drop(columns=["ref_MAE", "ref_QL"])
        if {"series", "horizon"} <= set(by):
            agg["DM_stat"], agg["DM_p"] = np.nan, np.nan
            key_cols = ["series", "origin", "horizon"]
            refrows = scored[scored.model == ref].set_index(key_cols)["err"]
            for idx in agg.index:
                g = dict(zip(by, idx if isinstance(idx, tuple) else (idx,)))
                if g["model"] == ref:
                    continue
                sub = scored[(scored.model == g["model"]) & (scored.series == g["series"]) & (scored.horizon == g["horizon"])]
                sub = sub.set_index(key_cols)["err"]
                common = sub.index.intersection(refrows.index)
                agg.loc[idx, ["DM_stat", "DM_p"]] = dm_test(sub[common].to_numpy(), refrows[common].to_numpy(), g["horizon"])
    return agg.round(4)


def holdout_selection(scored: pd.DataFrame, split: str = "2024-01-01", criterion: str = "ql",
                      ref: str = "naive") -> pd.DataFrame:
    """Choisit le meilleur modèle par série × horizon SUR LA PÉRIODE AVANT ``split`` seulement, puis mesure
    ce choix sur la période après (jamais vue lors du choix). Évite de surestimer le gain en choisissant le
    modèle sur les mêmes données que celles qui servent à le noter.

    Critère : ``ql`` (perte quantile, juge P10/P50/P90 ensemble) ou ``ae`` (erreur du P50)."""
    s = pd.Timestamp(split)
    # une prévision n'est « avant » que si sa CIBLE est connue avant le split (pas de chevauchement)
    train = scored[scored["target_month"] < s]
    test = scored[scored["origin"] >= s]
    rows = []
    for (ser, h), g in train.groupby(["series", "horizon"]):
        best = g.groupby("model")[criterion].mean().idxmin()
        t = test[(test.series == ser) & (test.horizon == h)]
        if t.empty:
            continue
        tb, tr = t[t.model == best], t[t.model == ref]
        rows.append({"series": ser, "horizon": h, "chosen_model": best, "n_test": len(tb),
                     "test_MAE_pct": 100 * tb["ae"].mean(), "test_rel_MAE": tb["ae"].mean() / tr["ae"].mean(),
                     "test_rel_QL": tb["ql"].mean() / tr["ql"].mean(), "test_coverage": tb["covered"].mean()})
    return pd.DataFrame(rows).set_index(["series", "horizon"]).round(3)


def overview(table: pd.DataFrame) -> pd.DataFrame:
    """Vue d'ensemble par modèle × horizon à partir d'une table summarize(by=model, series, horizon)."""
    t = table.reset_index()
    g = t.groupby(["model", "horizon"])
    out = pd.DataFrame({
        "rel_MAE_median": g["rel_MAE"].median(),
        "series_beating_naive": g["rel_MAE"].apply(lambda s: int((s < 1).sum())),
        "series_sig_better_5pct": g.apply(lambda d: int(((d["rel_MAE"] < 1) & (d.get("DM_p", 1) < 0.05)).sum()), include_groups=False),
        "n_series": g["rel_MAE"].size(),
        "rel_QL_median": g["rel_QL"].median(),
        "coverage_mean": g["coverage"].mean(),
        "below_mean": g["below"].mean(),
        "above_mean": g["above"].mean(),
        "width_pct_mean": g["width_pct"].mean(),
    })
    return out.round(3)
