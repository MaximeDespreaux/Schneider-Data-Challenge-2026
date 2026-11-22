"""Which provided series are used, and how? One row per series of the two challenge catalogs.

    uv run python -m src.checks.provided_series_table

Status (first match wins):
- "kept"                       : at least one of its cross_* features was kept by the screening (v1 or v2);
- "empty - licensed, not provided" : no data at all in the panel (licensed placeholder);
- "used via derived series (...)"  : input of a derived series that is a cross candidate or a target
  (E01->ED01, E05->ED02, E06->ED04, E07->ED03, S01->D01, S02->D02; S06 / E23 = FX conversion);
- "not tested - near-duplicate of <id>" : E02/E03/E04 (Brent E01), E16 (E14), E17/E19 (E13), E24 (E23);
- "tested - insufficient history (no data before 2013)" : candidate in CROSS v1 or v2, first month >= 2013-01
  (or no data at all) -> the 2003-2012 sub-period of the screening cannot validate it;
- "tested - not stable"        : candidate in CROSS v1 or v2, enough history, no feature kept;
- "target only"                : forecast target, not a cross candidate;
- "not tested"                 : neither a target nor a cross candidate.
Inputs: the challenge catalogs, the long panel (after extend_history), data/outputs/external/signal_screen*.csv.
Output (confidential, gitignored): data/outputs/checks/provided_series_status.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import data as d
from src.external.features import CROSS, CROSS_V2

HISTORY_CUTOFF = pd.Timestamp("2013-01-01")
# input series -> derived series that is a cross candidate or a target
DERIVED_FROM = {"E01": "ED01", "E05": "ED02", "E06": "ED04", "E07": "ED03", "S01": "D01", "S02": "D02",
                "S06": "FX conversion: D01/D02/ED01-ED04", "E23": "FX conversion: ED01-ED04"}
# not tested on purpose: near-duplicate of a series already used
NEAR_DUPLICATE = {"E02": "E01", "E03": "E01", "E04": "E01", "E16": "E14", "E17": "E13", "E19": "E13", "E24": "E23"}


def kept_pairs(screen: pd.DataFrame | None, sid: str) -> float:
    """Number of (target, horizon) pairs where any cross_<sid>_* feature was kept."""
    if screen is None:
        return np.nan
    k = screen[screen["keep"].astype(bool) & screen["feature"].str.startswith(f"cross_{sid}_")]
    return float(len(k[["target", "horizon"]].drop_duplicates()))


def main() -> None:
    from src.external.signals import build_panel
    from src.testbench.run import default_targets

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_colwidth", 45)
    rm, en = d.load_catalogs()
    cat = pd.concat([rm[["series_name"]], en[["series_name"]]])
    panel = build_panel(long=True, external=False)
    targets = set(default_targets(panel))
    ext = d.OUT / "external"
    screens = {}
    for v, f in {"v1": "signal_screen.csv", "v2": "signal_screen_v2.csv"}.items():
        p = ext / f
        screens[v] = pd.read_csv(p) if p.exists() else None
        if screens[v] is None:
            print(f"WARNING: {p} missing (run src.external.run_champion{' --candidates v2' if v == 'v2' else ''})")

    rows = []
    for sid, r in cat.iterrows():
        first = panel[sid].first_valid_index() if sid in panel and panel[sid].notna().any() else None
        in1, in2 = sid in CROSS, sid in CROSS_V2
        n1, n2 = kept_pairs(screens["v1"], sid), kept_pairs(screens["v2"], sid)
        derived = DERIVED_FROM.get(sid)
        derived_used = derived is not None and (derived.startswith("FX") or derived in CROSS_V2 or derived in targets)
        if (n1 or 0) > 0 or (n2 or 0) > 0:
            status = "kept"
        elif first is None:
            status = "empty - licensed, not provided"
        elif derived_used:
            status = f"used via derived series ({derived})"
        elif sid in NEAR_DUPLICATE:
            status = f"not tested - near-duplicate of {NEAR_DUPLICATE[sid]}"
        elif in1 or in2:
            status = ("tested - insufficient history (no data before 2013)"
                      if first is None or first >= HISTORY_CUTOFF else "tested - not stable")
        elif sid in targets:
            status = "target only"
        else:
            status = "not tested"
        rows.append({"series_id": sid, "name": r["series_name"], "forecast_target": "yes" if sid in targets else "no",
                     "in_cross_v1": "yes" if in1 else "no", "in_cross_v2": "yes" if in2 else "no",
                     "first_month": first.strftime("%Y-%m") if first is not None else "none",
                     "kept_pairs_v1": n1, "kept_pairs_v2": n2, "status": status})
    tab = pd.DataFrame(rows)
    out = d.OUT / "checks"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "provided_series_status.csv", index=False)
    print(tab.to_string(index=False))
    print("\n" + tab["status"].value_counts().to_string())
    print(f"-> {out / 'provided_series_status.csv'}")


if __name__ == "__main__":
    main()
