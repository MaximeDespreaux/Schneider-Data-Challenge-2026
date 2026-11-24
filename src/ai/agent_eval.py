"""Phase 2 evaluation: does the classical regression improve with the LLM-proposed features?

Same model (``models.dynreg_predict``: ridge, no intercept, shrunk to the random walk), same origins, same
drivers, with and without the accepted (formula, target) pairs from ``agent_features``. The formulas were
screened on targets before 2024, so only target months from 2024 on are reported (the earlier rows are in-sample).

    uv run --group ai python -m src.ai.agent_eval
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import backtest as bt
from src import features as fe
from src import models as mo

from . import backtest as ab
from . import config as C
from . import feature_dsl as dsl
from .agent_features import OUT


def add_agent_columns(feats: dict, acc: pd.DataFrame, T, X) -> tuple[dict, dict[str, list[str]]]:
    """Append accepted formulas as ``x_AGENTk_f`` columns on their target's rows; return extra drivers per target."""
    cols = {(f"AGENT{k}", r.target): dsl.compile_feature(r.formula, r.target, T, X) for k, r in enumerate(acc.itertuples())}
    return add_columns(feats, cols)


def add_columns(feats: dict, cols: dict[tuple[str, str], pd.Series]) -> tuple[dict, dict[str, list[str]]]:
    """Generic version: ``{(driver_name, target): monthly series}`` -> ``x_<driver_name>_f`` on that target's rows."""
    extra: dict[str, list[str]] = {}
    for (name, target), f in cols.items():
        extra.setdefault(target, []).append(name)
        for h in feats:
            col = f"x_{name}_f"
            idx = feats[h].index
            if col not in feats[h]:  # one driver name can serve several targets: fill each target's rows only
                feats[h][col] = np.nan
            mask = idx.get_level_values("series") == target
            feats[h].loc[mask, col] = f.reindex(idx[mask].get_level_values("month")).to_numpy()
    return feats, extra


def run(start: str = C.BACKTEST_START, acc: pd.DataFrame | None = None,
        cols: dict | None = None, label: str = "dynreg+agent") -> tuple[pd.DataFrame, pd.DataFrame]:
    """dynreg with vs without extra drivers: the agent's accepted formulas (default) or any ``cols`` mapping."""
    T, X = fe.build_panel()
    feats = {h: fe.make_features(T, X, h) for h in C.HORIZONS}
    if cols is None:
        acc = pd.read_csv(OUT / "accepted.csv") if acc is None else acc
        feats, extra = add_agent_columns(feats, acc, T, X)
    else:
        feats, extra = add_columns(feats, cols)

    def one(s):
        rows = []
        for h in C.HORIZONS:
            for t in bt._origins(T[s], h, start):
                base = mo.dynreg_predict(feats[h], s, fe.DRIVERS[s], t, h)
                rows.append(dict(series=s, origin=t, h=h, model="dynreg", pred=base))
                pred = mo.dynreg_predict(feats[h], s, fe.DRIVERS[s] + extra[s], t, h) if s in extra else base
                rows.append(dict(series=s, origin=t, h=h, model=label, pred=pred))
                rows.append(dict(series=s, origin=t, h=h, model="naive", pred=0.0))
        return rows

    rows = [r for part in Parallel(n_jobs=-1)(delayed(one)(s) for s in T.columns) for r in part]
    res = bt.attach_truth(pd.DataFrame(rows), T)
    for m in ("naive", "dynreg", label):
        res = bt.add_conformal(res, m)
    late = res.target_month >= pd.Timestamp(C.SELECTION_SPLIT)
    touched = res.series.isin(extra)
    out = pd.concat({
        "all targets, 2024 on": ab.compare(res, ["dynreg", label], late),
        "targets with the extra features, 2024 on": ab.compare(res, ["dynreg", label], late & touched),
    }, names=["window"])
    per = pd.concat({s: ab.compare(g, ["dynreg", label], g.target_month >= pd.Timestamp(C.SELECTION_SPLIT))
                     for s, g in res[touched].groupby("series")}, names=["series"])
    return out, per


def main() -> None:
    out, per = run()
    out.to_csv(OUT / "eval_summary.csv")
    per.to_csv(OUT / "eval_per_series.csv")
    pd.set_option("display.width", 200)
    print("=== dynreg with vs without the agent's features (rel* vs naive, < 1 = better)\n", out.round(3))
    print("\n=== per target that received features\n", per.round(3))


if __name__ == "__main__":
    main()
