"""Phase 2: an LLM proposes features in the formula language; local screening decides.

Loop, per target family (2–3 rounds):
    public metadata + language spec + feedback so far  ──►  Claude  ──►  JSON list of formulas
    formulas ──► parse / compile locally ──► screen on every target of the family, h = 3/6/9, targets < 2024
    feedback to the LLM: pass / fail, a coarse signal bucket and the reason. Never values, dates or data.

What leaves the machine (all through ``llm.ask`` → ``guard.check``): the language spec, public series ids with
their public descriptions, units and publication lags, a plain-words description of the target family, and the
feedback buckets. The targets' names, data and the existing feature values stay local.

Acceptance (pre-registered): a formula is kept for a target if it passes the screen at >= 2 of the 3 horizons.

    uv run --group ai python -m src.ai.agent_features            # needs ANTHROPIC_API_KEY with credits
"""

from __future__ import annotations

import json

import pandas as pd

from src import features as fe

from . import config as C
from . import feature_dsl as dsl
from . import guard, llm

ROUNDS, PER_ROUND = 3, 8
OUT = C.OUT / "agent_features"
# Plain-words descriptions of what TARGET is in each family (no client, no licensed source names).
FAMILY_DESC = {
    "steel_market": "European flat carbon steel market prices (cold-rolled and hot-dip galvanised coil, "
                    "Germany / France), monthly averages in EUR/t, published with no delay.",
    "stainless": "European stainless steel 304 market price and its alloy surcharge (Germany), monthly, EUR/t. "
                 "The surcharge follows nickel and chrome prices by formula with about a one-month lag.",
    "lme": "Base-metal world prices in EUR/t (copper and aluminium converted with EUR/USD) and nickel in USD/t, "
           "monthly averages, published with a one-month delay.",
    "ore": "Iron ore fines 62% Fe CFR China, USD per dry metric tonne, monthly, published with a one-month delay.",
    "ppi": "EU producer price indices (Eurostat, 2021 = 100): basic iron and steel (EU27 and Germany) and plastics "
           "in primary forms (EU27), published with a two-month delay.",
}
EXISTING_DESC = (
    "The model already has, for every target: its own last three monthly log-changes, 3/6/12-month momentum, "
    "6/12-month volatility, deviation from its 12 and 24-month moving average; and 1- and 3-month log-changes of "
    "Brent (EUR), TTF gas (EUR), German power, EU energy PPI, EUR/USD, nickel, iron ore, copper and aluminium (EUR) "
    "and the EU steel PPI. Proposals that only restate these are rejected as duplicates."
)
SCHEMA = {
    "type": "object",
    "properties": {"features": {"type": "array", "items": {
        "type": "object",
        "properties": {"formula": {"type": "string"}, "rationale": {"type": "string"}},
        "required": ["formula", "rationale"], "additionalProperties": False}}},
    "required": ["features"], "additionalProperties": False,
}
SYSTEM = ("You are a commodity economist designing leading indicators for monthly price forecasts 3, 6 and 9 months "
          "ahead. You only write formulas in the small language you are given; you never see the data. Prefer "
          "features with a clear economic mechanism (cost pass-through, input-output links, mean reversion of "
          "spreads, energy intensity, currency effects) over data-mined combinations.")


def metadata() -> str:
    cat = pd.read_parquet(C.ROOT / "data" / "processed" / "series_catalog.parquet")
    ids = sorted(guard.public_series())
    rows = [f"- {i}: {cat.at[i, 'description']} [{cat.at[i, 'unit']}, published {int(cat.at[i, 'pub_lag_months'])} "
            f"month(s) late]" for i in ids]
    return "\n".join(rows)


LANGUAGE = """Formula language (nothing else is accepted):
  leaves : a series id from the list, or TARGET (the price being forecast), or an integer 1..36
  dlog(x, k=1)   log change over k months        lag(x, k)       value k months earlier
  ratio(a, b)    a / b                           spread(a, b)    a - b
  dev_ma(x, k)   deviation from its trailing k-month mean (in logs for prices, plain for spreads)
  zscore(x, k)   (x - trailing k-month mean) / trailing k-month std
  vol(x, k)      std of monthly log changes over the trailing k months
No arithmetic operators (+ - * /): use spread() and ratio(). Max nesting depth 4. Every series is automatically lagged by its publication delay. Examples:
  zscore(spread(dlog(E09, 3), dlog(TARGET, 3)), 24)
  dev_ma(ratio(TARGET, S06), 24)"""


def prompt(family: str, history: list[dict]) -> str:
    tried = "\n".join(f"- {h['formula']}: {h['summary']}" for h in history) or "- (none yet)"
    return (f"TARGET family: {FAMILY_DESC[family]}\n\n{EXISTING_DESC}\n\nPublic series you may use:\n{metadata()}\n\n"
            f"{LANGUAGE}\n\nAlready tried (result of the out-of-sample screen; 'signal' is a coarse bucket of rank "
            f"correlation with the future return):\n{tried}\n\nPropose {PER_ROUND} NEW formulas that are not in the "
            f"tried list, each with a one-sentence economic rationale.")


def evaluate(formula: str, family: str, T: pd.DataFrame, X: pd.DataFrame, existing: dict) -> list[dsl.Screen]:
    out = []
    for s in [t for t in T.columns if fe.FAMILY[t] == family]:
        f = dsl.compile_feature(formula, s, T, X)
        for h in C.HORIZONS:
            e = existing[h].xs(s, level="series").drop(columns=["family", "y_level", "y"]).dropna(axis=1, how="all")
            out.append(dsl.screen(f, s, h, T, existing=e))
    return out


def summarise(screens: list[dsl.Screen]) -> str:
    """Feedback for the LLM: passes per target-horizon, best bucket, most common failure reason. No numbers."""
    fb = [s.feedback() for s in screens]
    n_pass = sum(f["passed"] for f in fb)
    order = ["none", "weak", "moderate", "strong"]
    best = max((f["signal"] for f in fb), key=order.index)
    fails = pd.Series([f["reason"] for f in fb if not f["passed"]])
    why = f"; most common failure: {fails.mode().iat[0]}" if len(fails) else ""
    return f"passed {n_pass} of {len(fb)} target-horizon checks, best signal {best}{why}"


def run_family(family: str, T, X, existing, rounds: int = ROUNDS, ask=llm.ask) -> list[dict]:
    history, rows = [], []
    for r in range(1, rounds + 1):
        resp = ask(prompt(family, history), system=SYSTEM, schema=SCHEMA, tag=f"agent_features/{family}/r{r}",
                   effort="high")
        for item in resp["features"][:PER_ROUND]:
            f = item["formula"].strip()
            if any(h["formula"] == f for h in history):
                continue
            try:
                dsl.parse(f)
                screens = evaluate(f, family, T, X, existing)
                summary = summarise(screens)
            except dsl.FormulaError as e:
                screens, summary = [], f"rejected by the parser: {e}"
            history.append({"formula": f, "summary": summary})
            for s in screens:
                rows.append({**s.__dict__, "family": family, "round": r, "rationale": item["rationale"]})
            if not screens:
                rows.append({"formula": f, "family": family, "round": r, "rationale": item["rationale"],
                             "passed": False, "reason": summary})
    return rows


def accepted(res: pd.DataFrame, min_horizons: int = 2) -> pd.DataFrame:
    """Pre-registered rule: keep (formula, target) if it passes at >= ``min_horizons`` of the 3 horizons."""
    ok = res.dropna(subset=["target"]).groupby(["formula", "target"]).passed.sum()
    return ok[ok >= min_horizons].rename("n_horizons_passed").reset_index()


def main(families: list[str] | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    T, X = fe.build_panel()
    existing = {h: pd.read_parquet(C.ROOT / "data" / "processed" / f"features_h{h}.parquet") for h in C.HORIZONS}
    rows = []
    for fam in families or list(FAMILY_DESC):
        rows += run_family(fam, T, X, existing)
        print(f"{fam}: {sum(1 for r in rows if r['family'] == fam)} screen rows | spent {llm.spent_usd():.3f} USD")
    res = pd.DataFrame(rows)
    res.to_parquet(OUT / "screen_results.parquet")
    acc = accepted(res)
    acc.to_csv(OUT / "accepted.csv", index=False)
    print("\naccepted (formula, target) pairs:\n", acc.to_string(index=False) if len(acc) else "none")
    print(json.dumps({"spent_usd": round(llm.spent_usd(), 4), "log": str(llm.LOG)}))


if __name__ == "__main__":
    main()
