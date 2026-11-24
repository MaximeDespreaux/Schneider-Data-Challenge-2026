"""One command for the AI track.

    uv sync --group ai
    uv run --group ai python -m src.ai.run            # phase 1: pretrained models (backtest + live forecast)

Needs the NDA data in data/ (licensed CRU series are forecast LOCALLY; nothing is sent anywhere except the one-off
download of the public model weights from Hugging Face). Outputs go to data/outputs/ai/ (git-ignored).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src import backtest as bt
from src import features as fe

from . import backtest as ab
from . import baselines as nb
from . import config as C

WATTS, KG_CO2_PER_KWH = 30.0, 0.25  # same order-of-magnitude assumptions as the team test bench
CLASSICAL = ["naive", "ensemble", "dynreg"]
COMBOS = ["chronos2+classical", "chronos2_p50+nvband", "chronos2~naive_vol"]
BENCHES = ("naive", "naive_vol")  # naive_vol = the team's volatility-scaled naive (tougher range benchmark)
REGION = {"D01": "WW", "D02": "WW", "S03": "WW", "S04": "WW"}  # LME / IMF world prices; the rest are EU prices
UNIT = {"S09": "index", "S10": "index", "S11": "index"}


def _ai_names() -> list[str]:
    return [f"{k}+cov" if v["covariates"] else k for k, v in C.MODELS.items()]


def classical_base(T: pd.DataFrame, X: pd.DataFrame, rebuild: bool = False) -> pd.DataFrame:
    """Classical models (naive, dynreg, ensemble...) refit here with the published-by-the-origin rule.

    Not read from data/processed/backtest.parquet: that file (notebook 02) was built before the pooled LightGBM was
    restricted to published outcomes. Cached; delete the file or pass ``rebuild`` after changing src/models.py."""
    path = C.OUT / "classical_backtest.parquet"
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    res = bt.add_ensemble(bt.rolling_origin(T, X, horizons=C.HORIZONS, start=C.BACKTEST_START))
    res = bt.add_conformal(res, "ensemble")
    C.OUT.mkdir(parents=True, exist_ok=True)
    res.to_parquet(path)
    return res


def backtest_stage(T: pd.DataFrame, X: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    base = classical_base(T, X)
    base = base[base.model.isin(CLASSICAL)]
    for m in ("naive", "dynreg"):  # same online split-conformal bands as the classical ensemble
        base = bt.add_conformal(base, m)
    ai, runtime = ab.run_foundation(T, X)
    res = pd.concat([base, nb.run_naive_vol(T), ai], ignore_index=True)
    res = ab.add_mean(res, ["chronos-2+cov", "ensemble"], "chronos2+classical")
    # CQR dropped: it over-widened every AI band (coverage 0.95+, relQL > 1) in the first run.
    # Combinations with the team's naive_vol fan (added after naive_vol was found to have the best bands).
    res = ab.add_hybrid(res, "chronos-2+cov", "naive_vol", "chronos2_p50+nvband")
    res = ab.add_qavg(res, "chronos-2+cov", "naive_vol", "chronos2~naive_vol")
    candidates = CLASSICAL + ["naive_vol"] + _ai_names() + COMBOS
    res, choice = ab.select_preregistered(res, candidates)
    return res, runtime, choice


def report_stage(res: pd.DataFrame) -> dict[str, pd.DataFrame]:
    models = ["naive_vol", "ensemble", "dynreg", *_ai_names(), *COMBOS, "ai_selected"]
    split = pd.Timestamp(C.SELECTION_SPLIT)
    late = res.origin >= split  # embargo: the choice uses outcomes up to the split, so only later origins are reported
    out = {}
    for b in BENCHES:
        sfx = "" if b == "naive" else f"_vs_{b}"
        out[f"all{sfx}"] = ab.compare(res, [m for m in models if m != b], bench=b)
        out[f"from_2024{sfx}"] = ab.compare(res, [m for m in models if m != b], late, bench=b)
        clean = []
        for k, v in C.MODELS.items():
            name = f"{k}+cov" if v["covariates"] else k
            c = ab.compare(res, [name], res.target_month > pd.Timestamp(v["clean_after"]), bench=b)
            if not c.empty:
                clean.append(c.assign(clean_after=v["clean_after"]))
        # combinations embed Chronos-2 and the selection may pick any model: clean only after the latest release
        last = max(v["clean_after"] for v in C.MODELS.values())
        c = ab.compare(res, [m for m in (*COMBOS, "ai_selected", "naive_vol") if m != b],
                       res.target_month > pd.Timestamp(last), bench=b)
        if not c.empty:
            clean.append(c.assign(clean_after=last))
        out[f"clean_window{sfx}"] = pd.concat(clean) if clean else pd.DataFrame()
    fam = res.assign(family=res.series.map(fe.FAMILY))
    out["by_family_from_2024"] = pd.concat(
        {f: ab.compare(g, models, g.origin >= split) for f, g in fam.groupby("family")}, names=["family"])
    dm = []
    for m in _ai_names() + ["ai_selected"]:
        d = bt.dm_test(res[late], m)
        if not d.empty:
            dm.append(d.assign(model=m))
    out["dm_vs_naive"] = pd.concat(dm).reset_index() if dm else pd.DataFrame()
    return out


def known_at(res: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    """Backtest rows whose outcome is dated at or before ``cut`` (...including the residuals behind the live bands)."""
    return res[res.target_month <= cut]


def live_stage(T, X, res: pd.DataFrame, choice: pd.DataFrame) -> pd.DataFrame:
    """Forecast at the official origin (``T``/``X`` already cut at it) with the pre-registered choice per (series, h).

    Team convention (champion/v2 testbench): target_month = origin + h for every series; a series whose last
    observation is older than the origin (publication lag) is forecast ``step = h + gap`` months ahead.
    AI and naive_vol bands are the models' own; classical bands come from their realised backtest residuals at h.
    """
    origin = T.index.max()
    gap = {s: (origin.to_period("M") - T[s].last_valid_index().to_period("M")).n for s in T}
    steps = sorted({h + g for h in C.HORIZONS for g in gap.values()})
    ai_live, _ = ab.run_foundation(T, X, live=True, horizons=steps)
    nv_live = nb.run_naive_vol(T, live=True, horizons=steps)
    fans = bt.forecast_now(T, X, res, horizons=steps)  # classical models (nb02 recipe) refit at the live origin
    rows = []
    for r in choice.itertuples():
        s, h, m = r.series, r.h, r.chosen
        st = h + gap[s]
        past = res[(res.model == m) & (res.series == s) & (res.h == h)].dropna(subset=["q10", "q90"])
        y0, last_obs = T[s].dropna().iloc[-1], T[s].last_valid_index()
        pick = lambda df, name: df[(df.model == name) & (df.series == s) & (df.h == st)].iloc[0]  # noqa: E731
        if m in _ai_names() or m == "naive_vol":
            a = pick(nv_live if m == "naive_vol" else ai_live, m)
            lo, mid, hi = a.q10, a.q50, a.q90
        elif m in ("chronos2_p50+nvband", "chronos2~naive_vol"):
            a, v = pick(ai_live, "chronos-2+cov"), pick(nv_live, "naive_vol")
            if m == "chronos2_p50+nvband":
                lo, mid, hi = a.q50 + v.q10, a.q50, a.q50 + v.q90
            else:
                lo, mid, hi = (a.q10 + v.q10) / 2, a.q50 / 2, (a.q90 + v.q90) / 2
        else:  # classical: live log-change from the private nb02 run (one column per model), band from residuals
            f = fans[(fans.series == s) & (fans.h == st)].iloc[0]
            if m == "chronos2+classical":
                a = pick(ai_live, "chronos-2+cov")
                mid = (f["ensemble"] + a.q50) / 2
            else:
                mid = float(f[m])  # naive = 0, dynreg / ensemble = their own live forecast
            resid = (past.actual - past.pred).dropna()
            lo, hi = mid + np.quantile(resid, 0.1), mid + np.quantile(resid, 0.9)
        rows.append(dict(series=s, origin=origin, last_obs=last_obs, step=st, h=h, model=m, y_origin=y0,
                         p10=y0 * np.exp(lo), p50=y0 * np.exp(mid), p90=y0 * np.exp(hi)))
    live = pd.DataFrame(rows)
    live["target_month"] = pd.to_datetime([o + pd.DateOffset(months=int(h)) for o, h in zip(live.origin, live.h)])
    live[["p10", "p50", "p90"]] = np.sort(live[["p10", "p50", "p90"]].to_numpy(), axis=1)
    return live


def submission_csv(live: pd.DataFrame, team_id: str = "team_xhec") -> pd.DataFrame:
    """Schneider's CSV format (deck slide 17). ``ai_tools`` lists the pretrained model behind each row."""
    def ai_tools(m: str) -> str:
        used = [k for k in C.MODELS if k in m] + (["chronos-2"] if m.startswith("chronos2") else [])
        return "; ".join(dict.fromkeys(used)) or "none"
    return pd.DataFrame({
        "team_id": team_id, "target_type": "material", "target_id": live.series,
        "region": live.series.map(REGION).fillna("EU"), "origin_month": live.origin.dt.strftime("%Y-%m"),
        "horizon_m": live.h.astype(int), "target_month": live.target_month.dt.strftime("%Y-%m"),
        "unit": live.series.map(UNIT).fillna("EUR/t"), "p10": live.p10, "p50": live.p50, "p90": live.p90,
        "model_name": "ai_track_" + live.model, "ai_tools": live.model.map(ai_tools),
    })


def footprint(runtime: dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame({"runtime_s": runtime}).rename_axis("model")
    df["kWh"] = df.runtime_s * WATTS / 3.6e6
    df["gCO2e"] = df.kWh * KG_CO2_PER_KWH * 1000
    return df


def main() -> None:
    C.OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    T, X = fe.build_panel()
    res, runtime, choice = backtest_stage(T, X)
    res.to_parquet(C.OUT / "ai_backtest.parquet")
    choice.to_csv(C.OUT / "ai_selection.csv", index=False)
    tables = report_stage(res)
    for name, df in tables.items():
        df.to_csv(C.OUT / f"ai_scores_{name}.csv")
    cut = pd.Timestamp(C.LIVE_ORIGIN)  # everything after the official origin is dropped for the live forecast...
    live = live_stage(T.loc[:cut], X.loc[:cut], known_at(res, cut), choice)
    live.to_parquet(C.OUT / "ai_live_forecast.parquet")
    sub = submission_csv(live)
    sub.to_csv(C.OUT / "team_xhec_forecast_ai.csv", index=False)
    C.licence_table().to_csv(C.OUT / "ai_licences.csv", index=False)
    fp = footprint(runtime)
    fp.to_csv(C.OUT / "ai_compute_footprint.csv")
    pd.set_option("display.width", 200)
    notes = {"clean_window": "target months after each model's release: the contamination-free check",
             "from_2024": "origins from 2024 on; pretrained models may have seen part of these months",
             "all": "all origins, in-sample for the selection; pretrained models may have seen most of these months"}
    for name in ("clean_window", "clean_window_vs_naive_vol", "from_2024", "from_2024_vs_naive_vol", "all"):
        bench = "naive_vol" if "naive_vol" in name else "naive"
        print(f"\n=== {name} (rel* < 1 beats {bench}; coverage target 0.80)\n    {notes[name.split('_vs_')[0]]}\n",
              tables[name].round(3))
    print("\n=== pre-registered choice (count by model)\n", choice.chosen.value_counts())
    print("\n=== compute\n", fp.round(4), f"\n\ntotal wall time {time.perf_counter() - t0:.0f}s -> {C.OUT}")


if __name__ == "__main__":
    main()
