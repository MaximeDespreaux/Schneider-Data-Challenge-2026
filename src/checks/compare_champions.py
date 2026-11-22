"""Champion v1 vs champion v2 on the 2024-26 holdout, with a decision rule fixed BEFORE looking at the results.

    uv run python -m src.checks.compare_champions

PRE-REGISTERED DECISION RULE:
    Adopt v2 only if, on 2024-01 to 2026-09 origins, its median rel_QL across (series, horizon) is lower than v1's
    AND its median coverage is not lower than v1's minus 0.02. Otherwise keep v1.

driver_ridge is shown for information only; it is not part of the rule.
Inputs: data/outputs/external/champion_holdout.parquet (v1) and champion_holdout_v2.parquet (v2).
Output (confidential, gitignored): data/outputs/checks/champion_v1_vs_v2.csv.
"""

from __future__ import annotations

import pandas as pd

from src import data as d
from src.testbench import metrics

RULE = ("Adopt v2 only if, on 2024-01 to 2026-09 origins, its median rel_QL across (series, horizon) is lower than "
        "v1's AND its median coverage is not lower than v1's minus 0.02. Otherwise keep v1.")
START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-09-01")
MODELS = ["naive", "naive_vol", "champion", "champion_v2", "driver_ridge"]
COVERAGE_TOL = 0.02


def main() -> None:
    pd.set_option("display.width", 200)
    print("PRE-REGISTERED DECISION RULE:")
    print("  " + RULE + "\n")
    ext = d.OUT / "external"
    paths = {"v1": ext / "champion_holdout.parquet", "v2": ext / "champion_holdout_v2.parquet"}
    cmds = {"v1": "uv run python -m src.external.run_champion",
            "v2": "uv run python -m src.external.run_champion --candidates v2"}
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        for k in missing:
            print(f"Missing {paths[k]} -> create it with: {cmds[k]}")
        raise SystemExit(1)

    v1 = pd.read_parquet(paths["v1"])
    v2 = pd.read_parquet(paths["v2"])
    fc = pd.concat([v1, v2[v2["model"] == "champion_v2"]], ignore_index=True)
    fc = fc[(pd.to_datetime(fc["origin"]) >= START) & (pd.to_datetime(fc["origin"]) <= END)]
    scored = metrics.score_rows(fc)
    tab = metrics.summarize(scored).reset_index()  # one row per model × series × horizon, rel_* vs naive

    models = [m for m in MODELS if m in set(tab["model"])]
    per_h = (tab[tab["model"].isin(models)]
             .groupby(["horizon", "model"])[["rel_MAE", "rel_QL", "coverage"]].median()
             .rename(columns=lambda c: f"median_{c}"))
    print("Per horizon, median across series (rel_* vs naive):")
    print(per_h.unstack("model").round(3).to_string())

    # rule: pooled over (series, horizon) pairs that both champions forecast
    c1 = tab[tab["model"] == "champion"].set_index(["series", "horizon"])
    c2 = tab[tab["model"] == "champion_v2"].set_index(["series", "horizon"])
    common = c1.index.intersection(c2.index)
    m1_ql, m2_ql = c1.loc[common, "rel_QL"].median(), c2.loc[common, "rel_QL"].median()
    m1_cov, m2_cov = c1.loc[common, "coverage"].median(), c2.loc[common, "coverage"].median()
    better_ql = m2_ql < m1_ql
    cov_ok = m2_cov >= m1_cov - COVERAGE_TOL
    outcome = "ADOPT V2" if (better_ql and cov_ok) else "KEEP V1"
    print(f"\nOver {len(common)} (series, horizon) pairs:")
    print(f"  median rel_QL   v1 = {m1_ql:.4f}   v2 = {m2_ql:.4f}   -> v2 lower: {better_ql}")
    print(f"  median coverage v1 = {m1_cov:.4f}   v2 = {m2_cov:.4f}   -> v2 >= v1 - {COVERAGE_TOL}: {cov_ok}")
    print(f"\nRULE OUTCOME: {outcome}")

    out = d.OUT / "checks"
    out.mkdir(parents=True, exist_ok=True)
    res = per_h.reset_index()
    overall = pd.DataFrame([
        {"horizon": "all", "model": "champion", "median_rel_QL": m1_ql, "median_coverage": m1_cov},
        {"horizon": "all", "model": "champion_v2", "median_rel_QL": m2_ql, "median_coverage": m2_cov}])
    res = pd.concat([res, overall], ignore_index=True)
    res["n_pairs_rule"] = len(common)
    res["rule_outcome"] = outcome
    res.to_csv(out / "champion_v1_vs_v2.csv", index=False)
    print(f"-> {out / 'champion_v1_vs_v2.csv'}")


if __name__ == "__main__":
    main()
