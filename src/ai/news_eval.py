"""Phase 3 evaluation: does the LLM news signal beat (a) nothing and (b) GDELT's own tone?

For each material signal and each of its targets:
1. Screen (pre-2024 targets only, same rules as the agent's features) for: LLM score, its 3-month mean, GDELT tone,
   tone 3-month change.
2. dynreg with vs without the signal (origins from 2024 on), for the LLM signal and for the tone baseline separately.
Headlines of month m are known at the end of month m, the same time as the month-m price: no extra shift.
The LLM may have seen how 2024-25 unfolded; ``agent_eval.run`` adds the rows after its release as a separate window.
"""

from __future__ import annotations

import pandas as pd

from src import features as fe

from . import agent_eval as ev
from . import config as C
from . import feature_dsl as dsl
from . import llm
from .news import DIR, MATERIAL_TARGETS

OUT = C.OUT / "news"


def signals() -> dict[str, pd.DataFrame]:
    sc = pd.read_parquet(DIR / "llm_scores.parquet")[list(MATERIAL_TARGETS)].astype(float)
    tone = pd.read_parquet(DIR / "tone.parquet")[list(MATERIAL_TARGETS)]
    return {"llm": sc, "llm_ma3": sc.rolling(3, min_periods=2).mean(), "tone": tone, "tone_d3": tone - tone.shift(3)}


def screen_table() -> pd.DataFrame:
    T, _ = fe.build_panel()
    rows = []
    for kind, df in signals().items():
        for mat, targets in MATERIAL_TARGETS.items():
            f = df[mat].reindex(T.index)
            for s in targets:
                for h in C.HORIZONS:
                    r = dsl.screen(f.rename(f"{kind}:{mat}"), s, h, T)
                    rows.append({"signal": kind, "material": mat, "target": s, "h": h, "n": r.n, "rho": r.rho,
                                 "stable": r.stable_regimes, "passed": r.passed, "reason": r.reason})
    return pd.DataFrame(rows)


def dynreg_tables() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    T, _ = fe.build_panel()
    out = {}
    for kind, label in [("llm_ma3", "dynreg+news_llm"), ("tone_d3", "dynreg+news_tone")]:
        df = signals()[kind]
        cols = {(f"NEWS_{kind}_{mat}", s): df[mat].reindex(T.index)
                for mat, targets in MATERIAL_TARGETS.items() for s in targets}
        out[label] = ev.run(cols=cols, label=label)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    sc = screen_table()
    sc.to_csv(OUT / "news_screen.csv", index=False)
    print("=== screen (targets < 2024): share of (target, h) checks passed, mean rho\n",
          sc.groupby("signal").agg(passed=("passed", "mean"), mean_rho=("rho", "mean"), n_checks=("passed", "size")).round(2))
    for label, (summary, per) in dynreg_tables().items():
        summary.to_csv(OUT / f"{label}_summary.csv")
        per.to_csv(OUT / f"{label}_per_series.csv")
        print(f"\n=== {label} vs dynreg, origins 2024 on (rel* vs naive)\n", summary.round(3))
    rel = C.llm_released(llm.model_in_use())
    print(f"\nnote: the LLM ({llm.model_in_use()}) was released {rel:%Y-%m-%d}; earlier targets may be contaminated."
          if rel is not None else f"\nnote: no verified release date for {llm.model_in_use()}: no clean window.")
