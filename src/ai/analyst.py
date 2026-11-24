"""Phase 4: the LLM as a judgmental forecaster (P10/P50/P90), on public series only.

* Series: the 7 public targets (LME copper / aluminium in EUR, nickel, iron ore, EU steel / German steel / plastics
  PPIs). Licensed targets are never in the prompt.
* Prompt: 24 months of each series rebased to 100 at the forecast month, with relative month labels and no dates,
  plus 12 months of EUR/USD, Brent and TTF gas (also rebased, lagged by their publication delay). No date and no
  level make it harder for the model to recall what happened next.
* Timing: one prompt covers series with different publication lags, so a forecast month is a DECISION month and each
  series stops at ``decision - its publication lag``. t0 (and the row's origin) is therefore specific to each series.
* One call per forecast month, all 7 series at once; JSON schema output (percent changes at h = 3/6/9).
* Contamination: the model may have seen 2024-25 prices in pre-training. Only target months after its release
  (``config.LLMS[model]['released']``) are a fair test; earlier months are reported separately and flagged.
* Inclusion rule (pre-registered): the analyst enters the CSV only if, on the clean window, its range quality beats
  naive_vol. Otherwise it is reported as a negative result.

    uv run --group ai python -m src.ai.analyst backtest    # one LLM call per month from 2024-01
    uv run --group ai python -m src.ai.analyst live        # forecast at the official origin
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src import backtest as bt
from src import features as fe

from . import backtest as ab
from . import config as C
from . import guard, llm

SERIES = ["D01", "D02", "S03", "S04", "S09", "S10", "S11"]
CONTEXT = {"S06": "EUR/USD exchange rate", "ED01": "Brent crude in EUR", "ED02": "TTF natural gas in EUR"}
OUT = C.OUT / "analyst"
SYSTEM = ("You are an experienced commodity and producer-price analyst. You forecast monthly prices 3, 6 and 9 months "
          "ahead with calibrated uncertainty: the true value should fall below your P10 one time in ten and above "
          "your P90 one time in ten. Prices are close to a random walk, so do not predict large moves without a "
          "reason, and widen ranges when recent volatility is high.")


def _schema() -> dict:
    hq = {"type": "object", "properties": {q: {"type": "number"} for q in ("p10", "p50", "p90")},
          "required": ["p10", "p50", "p90"], "additionalProperties": False}
    one = {"type": "object", "properties": {"h3": hq, "h6": hq, "h9": hq, "rationale": {"type": "string"}},
           "required": ["h3", "h6", "h9", "rationale"], "additionalProperties": False}
    return {"type": "object", "properties": {s: one for s in SERIES}, "required": SERIES, "additionalProperties": False}


def _fmt(v: float) -> str:
    """One-decimal index value. A rebased public value can equal a private number on the deny-list by chance; it is
    then moved by 0.1 point (0.1%), so the gate itself never has to be relaxed."""
    for c in (v, v + 0.1, v - 0.1, v + 0.2):
        t = f"{c:.1f}"
        if not guard.problems(f"x {t} x"):
            return t
    raise guard.NDAViolation("no safe spelling for a value")


def _block(y: pd.Series, n: int) -> str:
    y = y.dropna().iloc[-n:]
    idx = 100 * y / y.iloc[-1]
    k = len(idx) - 1
    return "; ".join(f"t-{k - i}={_fmt(v)}" if k - i else f"t0={_fmt(v)}" for i, v in enumerate(idx))


def _cut(series: str, decision: pd.Timestamp, lagged: bool) -> pd.Timestamp:
    """Last month of ``series`` published at ``decision`` (month end). ``lagged=False``: data already cut (live)."""
    return decision - pd.DateOffset(months=fe.pub_lag(series) if lagged else 0)


def prompt(origin: pd.Timestamp, T: pd.DataFrame, X: pd.DataFrame, lagged: bool = True) -> str:
    cat = pd.read_parquet(C.ROOT / "data" / "processed" / "series_catalog.parquet")
    guard.public_frame(T[SERIES])
    guard.public_frame(X[list(CONTEXT)])
    lines = ["Each series is rebased to 100 at t0, its latest published month (t-1 is one month earlier, and so on).", ""]
    for s in SERIES:
        lines += [f"{s} - {cat.at[s, 'description']}:", _block(T[s].loc[:_cut(s, origin, lagged)], 24), ""]
    lines += ["Context (rebased to 100 at its latest published month):"]
    for c, name in CONTEXT.items():
        lines += [f"{name}:", _block(X[c].loc[:_cut(c, origin, lagged)], 12)]
    lines += ["", "For every series, give P10, P50 and P90 of the percent change from t0 to t+3, t+6 and t+9 "
              "(e.g. -4.5 means 4.5% lower than at t0), and a one-sentence rationale."]
    return "\n".join(lines)


def forecast(origin: pd.Timestamp, T, X, lagged: bool = True) -> pd.DataFrame:
    out = llm.ask(prompt(origin, T, X, lagged), system=SYSTEM, schema=_schema(), tag=f"analyst/{origin:%Y-%m}",
                  max_tokens=12000)
    rows = []
    for s in SERIES:
        t0 = T[s].loc[:_cut(s, origin, lagged)].last_valid_index()
        for h in C.HORIZONS:
            q = out[s][f"h{h}"]
            p = np.sort([q["p10"], q["p50"], q["p90"]])
            lq = np.log1p(np.clip(p, -95, 500) / 100)
            rows.append(dict(series=s, origin=t0, h=h, model="llm_analyst", pred=lq[1], q10=lq[0], q50=lq[1],
                             q90=lq[2], rationale=out[s]["rationale"]))
    return pd.DataFrame(rows)


def backtest(start: str = "2024-01-01") -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    T, X = fe.build_panel()
    last = T[SERIES].dropna(how="all").index.max() - pd.DateOffset(months=min(C.HORIZONS))
    frames = []
    for o in pd.date_range(start, last, freq="MS"):
        try:
            frames.append(forecast(o, T, X))
            print(f"  analyst {o:%Y-%m} done", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  analyst {o:%Y-%m} skipped: {type(e).__name__}: {str(e)[:150]}", flush=True)
    res = bt.attach_truth(pd.concat(frames, ignore_index=True), T[SERIES])
    res.to_parquet(OUT / "analyst_backtest.parquet")
    return res


def evaluate(res: pd.DataFrame | None = None) -> pd.DataFrame:
    res = pd.read_parquet(OUT / "analyst_backtest.parquet") if res is None else res
    base = pd.read_parquet(C.OUT / "ai_backtest.parquet")
    base = base[base.model.isin(["naive", "naive_vol", "chronos-2+cov", "chronos2~naive_vol"]) & base.series.isin(SERIES)]
    allr = pd.concat([base, res.drop(columns="rationale")], ignore_index=True)
    allr = allr.merge(res[["series", "origin", "h"]].drop_duplicates(), on=["series", "origin", "h"])
    released = C.llm_released(llm.model_in_use())
    if released is None:  # unverified release date: everything counts as possibly contaminated
        released = pd.Timestamp.max
    models = ["llm_analyst", "chronos-2+cov", "chronos2~naive_vol"]
    out = {}
    for b in ("naive", "naive_vol"):
        for name, w in [("before release (contaminated)", allr.target_month <= released),
                        ("after release (clean)", allr.target_month > released)]:
            c = ab.compare(allr, [m for m in models + (["naive_vol"] if b == "naive" else []) if m != b], w, bench=b)
            if not c.empty:
                out[(f"vs {b}", name)] = c
    table = pd.concat(out, names=["benchmark", "window"])
    table.to_csv(OUT / "analyst_scores.csv")
    return table


def live() -> pd.DataFrame:
    T, X = fe.build_panel()
    cut = pd.Timestamp(C.LIVE_ORIGIN)
    Tl, Xl = T.loc[:cut], X.loc[:cut]
    last = {s: Tl[s].last_valid_index() for s in SERIES}
    if any(m != cut for m in last.values()):  # the prompt asks for t0 + h, which must be the official origin + h
        raise RuntimeError(f"live analyst forecast needs every series up to {cut:%Y-%m}, got {last}")
    OUT.mkdir(parents=True, exist_ok=True)
    f = forecast(cut, Tl, Xl, lagged=False)  # team convention: everything dated <= the official origin is known
    f["last_obs"] = [Tl[s].last_valid_index() for s in f.series]
    f["y_origin"] = [Tl[s].dropna().iloc[-1] for s in f.series]
    for q in ("q10", "q50", "q90"):
        f[q.replace("q", "p")] = f.y_origin * np.exp(f[q])
    f.to_parquet(OUT / "analyst_live.parquet")
    return f


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    pd.set_option("display.width", 200)
    if cmd == "backtest":
        backtest()
        print(evaluate().round(3))
    elif cmd == "evaluate":
        print(evaluate().round(3))
    elif cmd == "live":
        print(live()[["series", "h", "p10", "p50", "p90", "rationale"]].to_string())
