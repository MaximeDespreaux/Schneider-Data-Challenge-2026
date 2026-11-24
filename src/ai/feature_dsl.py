"""A small formula language for features proposed by an LLM, its safe compiler, and the local screening.

The LLM never writes code and never sees data. It writes formulas such as

    zscore(spread(dlog(E09, 3), dlog(TARGET, 3)), 24)

which are parsed with ``ast`` (no ``eval``) and compiled here, locally, on the full panel.

Language
--------
Leaves:  a public series id (``S09``, ``E09``, ``ED02``...), or ``TARGET`` (the series being forecast: its name
         and data never leave the machine), or an integer window / lag (1..36).
Ops (all strictly backward-looking: shifts are positive, windows trail):
    dlog(x, k=1)      log(x_t) - log(x_{t-k})
    lag(x, k)         x_{t-k}
    ratio(a, b)       a / b
    spread(a, b)      a - b
    dev_ma(x, k)      log(x_t) - log(mean(x_{t-k+1..t})); plain x_t - mean if x has values <= 0 (spreads)
    zscore(x, k)      (x_t - mean_k) / std_k over the trailing k months
    vol(x, k)         std of monthly dlog over the trailing k months

Publication lags: every series leaf is shifted by ``features._shift_for(leaf, target)`` before any op, so a
feature only uses values that were published by the forecast date.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import features as fe

from . import guard

MAX_K, MAX_DEPTH, MAX_NODES = 36, 4, 15
ARITY = {"dlog": (1, 2), "lag": (2, 2), "ratio": (2, 2), "spread": (2, 2), "dev_ma": (2, 2), "zscore": (2, 2),
         "vol": (2, 2)}
SCREEN_CUTOFF = "2024-01-01"  # screening only uses targets realised before this month (pre-registered)
REGIMES = {"pre-2021": ("2012-01-01", "2020-12-01"), "2021-22 shock": ("2021-01-01", "2022-12-01"),
           "2023": ("2023-01-01", "2023-12-01")}


class FormulaError(ValueError):
    """The formula is not in the language (unknown op, bad arity, window out of range, non-public series...)."""


# --------------------------------------------------------------------------------------------- parse
def parse(formula: str) -> ast.AST:
    try:
        tree = ast.parse(formula.strip(), mode="eval").body
    except SyntaxError as e:
        raise FormulaError(f"syntax: {e.msg}") from None
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES * 2:  # ast.walk counts Load contexts too
        raise FormulaError("formula too long")
    _validate(tree, depth=0)
    return tree


def _validate(n: ast.AST, depth: int) -> None:
    if depth > MAX_DEPTH:
        raise FormulaError(f"nesting deeper than {MAX_DEPTH}")
    if isinstance(n, ast.Name):
        if n.id != "TARGET" and n.id not in guard.public_series():
            raise FormulaError(f"unknown or non-public series '{n.id}'")
        return
    if isinstance(n, ast.Constant):
        if not (isinstance(n.value, int) and not isinstance(n.value, bool) and 1 <= n.value <= MAX_K):
            raise FormulaError(f"constants must be integers in 1..{MAX_K}, got {n.value!r}")
        return
    if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and not n.keywords):
        raise FormulaError(f"only calls like op(a, b) are allowed, got {type(n).__name__}")
    op = n.func.id
    if op not in ARITY:
        raise FormulaError(f"unknown op '{op}' (allowed: {', '.join(ARITY)})")
    lo, hi = ARITY[op]
    if not lo <= len(n.args) <= hi:
        raise FormulaError(f"{op} takes {lo}-{hi} arguments")
    if op in ("dlog", "lag", "dev_ma", "zscore", "vol") and len(n.args) == 2 and not isinstance(n.args[1], ast.Constant):
        raise FormulaError(f"the 2nd argument of {op} must be an integer window")
    if op in ("ratio", "spread") and any(isinstance(a, ast.Constant) for a in n.args):
        raise FormulaError(f"{op} takes two series expressions")
    if isinstance(n.args[0], ast.Constant):
        raise FormulaError(f"the 1st argument of {op} must be a series expression")
    for a in n.args:
        _validate(a, depth + 1)


def leaves(formula: str) -> set[str]:
    """Series names used by a formula (op names excluded)."""
    return {n.id for n in ast.walk(parse(formula)) if isinstance(n, ast.Name)} - set(ARITY)


# --------------------------------------------------------------------------------------------- compile
def compile_feature(formula: str, target: str, targets: pd.DataFrame, exog: pd.DataFrame) -> pd.Series:
    """Monthly feature for ``target``; value at t uses only data published by t."""
    tree = parse(formula)
    panel = pd.concat([targets, exog.drop(columns=targets.columns, errors="ignore")], axis=1)

    def leaf(name: str) -> pd.Series:
        if name == "TARGET":
            return targets[target]
        if name not in panel:
            raise FormulaError(f"series '{name}' not in the local panel")
        return panel[name].shift(fe._shift_for(name, target))

    def ev(n: ast.AST):
        if isinstance(n, ast.Name):
            return leaf(n.id).astype(float)
        if isinstance(n, ast.Constant):
            return int(n.value)
        op, args = n.func.id, [ev(a) for a in n.args]
        x = args[0]
        k = args[1] if len(args) > 1 else 1
        logx = lambda s: np.log(s.where(s > 0))  # noqa: E731
        if op == "dlog":
            return logx(x) - logx(x.shift(k))
        if op == "lag":
            return x.shift(k)
        if op == "ratio":
            return x / args[1].replace(0, np.nan)
        if op == "spread":
            return x - args[1]
        if op == "dev_ma":
            m = x.rolling(k, min_periods=max(2, k // 2)).mean()
            return logx(x) - logx(m) if (x.dropna() > 0).all() else x - m
        if op == "zscore":
            r = x.rolling(k, min_periods=max(3, k // 2))
            return (x - r.mean()) / r.std().replace(0, np.nan)
        if op == "vol":
            return logx(x).diff().rolling(k, min_periods=max(3, k // 2)).std()
        raise FormulaError(op)

    out = ev(tree)
    if not isinstance(out, pd.Series):
        raise FormulaError("formula must produce a series")
    return out.replace([np.inf, -np.inf], np.nan).rename(formula)


# --------------------------------------------------------------------------------------------- screen
@dataclass
class Screen:
    formula: str
    target: str
    h: int
    n: int
    rho: float
    stable_regimes: int
    max_corr_existing: float
    passed: bool
    reason: str

    def feedback(self) -> dict:
        """What the LLM is allowed to see: pass/fail, a coarse |rho| bucket and the reason. No values, no dates."""
        b = abs(self.rho) if np.isfinite(self.rho) else 0
        bucket = "none" if b < 0.10 else "weak" if b < 0.20 else "moderate" if b < 0.35 else "strong"
        return {"formula": self.formula, "horizon": self.h, "passed": self.passed, "signal": bucket,
                "reason": self.reason}


def screen(feature: pd.Series, target: str, h: int, targets: pd.DataFrame, existing: pd.DataFrame | None = None,
           cutoff: str = SCREEN_CUTOFF, min_rho: float = 0.10, min_n: int = 60, max_dup: float = 0.9) -> Screen:
    """Notebook 03's filters on targets realised before ``cutoff``: n >= 60, |Spearman rho| >= 0.10, sign-stable
    in >= 2 of the 3 regimes, and not a near-duplicate (|corr| <= 0.9) of a feature already selected."""
    ly = np.log(targets[target])
    y = (ly.shift(-h) - ly).rename("y")
    last_origin = pd.Timestamp(cutoff) - pd.DateOffset(months=h + 1)  # target month t+h < cutoff
    xy = pd.concat([feature.rename("x"), y], axis=1).loc[:last_origin].dropna()
    name = feature.name or "feature"
    if len(xy) < min_n or xy.x.std() == 0:
        return Screen(name, target, h, len(xy), np.nan, 0, np.nan, False, f"too few usable months ({len(xy)} < {min_n})")
    rho = xy.x.corr(xy.y, method="spearman")
    stable = 0
    for a, b in REGIMES.values():
        sub = xy.loc[a:min(pd.Timestamp(b), last_origin)]
        if len(sub) >= 12 and sub.x.std() > 0:
            stable += int(np.sign(sub.x.corr(sub.y, method="spearman")) == np.sign(rho))
    dup = 0.0
    if existing is not None and not existing.empty:
        e = existing.loc[:last_origin].reindex(xy.index)
        c = e.corrwith(xy.x, method="spearman").abs()
        dup = float(c.max()) if c.notna().any() else 0.0
    if abs(rho) < min_rho:
        ok, why = False, "no predictive signal before 2024"
    elif stable < 2:
        ok, why = False, "sign not stable across regimes"
    elif dup > max_dup:
        ok, why = False, "duplicates a feature already in the model"
    else:
        ok, why = True, "passed"
    return Screen(name, target, h, len(xy), float(rho), stable, dup, ok, why)
