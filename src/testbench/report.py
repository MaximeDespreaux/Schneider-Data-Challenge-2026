"""Plain-language report of a backtest run (dev or graded), applying the pre-registered pass rule.

Pass rule (docs/backtest/preregistration.md): wf_blend beats naive in a group x horizon only if the one-sided pooled
DM test, Holm-adjusted, gives p < 0.05 in BOTH modes; otherwise "not detectable" with the minimum detectable effect.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SUBMITTED = "wf_blend"
ALPHA = 0.05


def verdicts(tests: dict[str, pd.DataFrame], mde: pd.DataFrame | None = None, model: str = SUBMITTED) -> pd.DataFrame:
    keys = ["group", "horizon"]
    parts = []
    for mode, t in tests.items():
        t = t[t.model == model][keys + ["rel_QL", "rel_QL_lo", "rel_QL_hi", "p_holm", "coverage"]]
        parts.append(t.set_index(keys).add_suffix(f"_{mode}"))
    v = pd.concat(parts, axis=1).reset_index()
    modes = list(tests)
    ok = pd.concat([v[f"p_holm_{m}"] < ALPHA for m in modes], axis=1).all(axis=1)
    if mde is not None:
        v = v.merge(mde[keys + ["mde_rel_ql"]], on=keys, how="left")
    def word(row, good):
        if good:
            return "beats naive"
        m = row.get("mde_rel_ql")
        return "not detectable" + (f" (MDE {m:.1%} of naive loss)" if pd.notna(m) else "")
    v["verdict"] = [word(r, g) for r, g in zip(v.to_dict("records"), ok)]
    return v


def _block(df: pd.DataFrame) -> str:
    return "```\n" + df.to_string(index=False) + "\n```"


def write_report(run_dir: Path, canary_json: Path, mde: pd.DataFrame | None = None) -> Path:
    run_dir = Path(run_dir)
    tests = {m: pd.read_csv(run_dir / m / "pooled_tests.csv") for m in ("submission", "realtime")
             if (run_dir / m / "pooled_tests.csv").exists()}
    can = json.loads(Path(canary_json).read_text())
    v = verdicts(tests, mde)
    lines = ["# Backtest report", "",
             f"Run folder: `{run_dir.name}`. Leak canaries for commit {can['head'][:7]}: "
             f"{'PASS' if can['PASS'] else 'FAIL'} (dirty tree: {can['dirty']}).", "",
             "## Submitted model (wf_blend) vs naive, pre-registered rule", "",
             "rel_QL < 1 means a lower pinball loss than naive. CI = 95% block bootstrap. p = one-sided pooled DM, "
             "Holm-adjusted. 'beats naive' requires p < 0.05 in both modes.", "",
             _block(v.round(3)), ""]
    for mode, t in tests.items():
        piv = t.pivot_table(index=["group", "horizon"], columns="model", values="rel_QL").round(3)
        lines += [f"## All models, rel_QL vs naive ({mode} mode)", "", "```\n" + piv.to_string() + "\n```", ""]
    path = run_dir / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
