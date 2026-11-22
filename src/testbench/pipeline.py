"""The hardened backtest, end to end (spec 5.8 / plan Task 12).

    uv run python -m src.testbench.pipeline --dev                 # DEV window, both modes (default)
    uv run python -m src.testbench.pipeline --canaries            # leak canaries on real data -> report per commit
    uv run python -m src.testbench.pipeline --graded              # ONE graded run (tagged clean commit + canaries)
    uv run python -m src.testbench.pipeline --submit              # CSV at origin 2026-09 from the graded run (wf_blend)
    options: --mode submission|realtime|both, --no-licensed

Base pass: every candidate at every origin (2010-01 on, 1992-extended panel, causal BOM indices). Submission mode
masks each input by its live ragged edge; realtime mode by the publication lags of config/availability.yaml.
Meta pass (Strategy A): wf_best / wf_blend (submitted) / wf_median from realised rows only.
Outputs (confidential, gitignored) in data/outputs/backtest/<dev | graded/<tag>>/<mode>/.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src import data as d

from . import metrics, protocol, stats
from .availability import ragged_edge_lags, realtime_lags
from .backtest import compute_footprint, run_backtest
from .meta import MetaConfig, meta_pass, pick_stability

OUT = d.OUT / "backtest"
BOM_IDS = ["RM6_NE_IQI_433520", "RMAirSeT_IIQ_24kV_IEC"]
INTERNAL = ["L01", "L02", "L03", "EP01"]          # forecast and scored, never in the CSV
CSV_TARGETS = d.SCHNEIDER_RM_TARGETS + BOM_IDS
SUBMITTED = "wf_blend"


def targets(panel: pd.DataFrame) -> list[str]:
    return [s for s in d.SCHNEIDER_RM_TARGETS + INTERNAL if s in panel and panel[s].notna().any()] + BOM_IDS


def lags_for(mode: str, panel: pd.DataFrame, tg: list[str], live_origin: str) -> dict[str, int]:
    if mode == "realtime":
        return realtime_lags()
    return ragged_edge_lags(panel, live_origin, exempt=set(tg))


def stat_tables(scored: pd.DataFrame, mode: str, lags: dict[str, int]) -> pd.DataFrame:
    """Per model x group x horizon vs naive: rel QL, pooled DM (one-sided, model better), Holm within model,
    block-bootstrap CI of rel QL, coverage and Kupiec p-values (non-overlapping origins)."""
    rows = []
    models = [m for m in scored.model.unique() if m != "naive"]
    for m in models:
        for g, members in stats.GROUPS.items():
            sub = scored[scored.series.isin(members)]
            if sub.empty:
                continue
            for h in sorted(sub.horizon.unique()):
                extra = 0 if mode == "submission" else max(lags.get(s, 0) for s in members)
                stat, p, n = stats.pooled_dm(sub, m, "naive", members, h, extra_lag=extra)
                a = sub[(sub.model == m) & (sub.horizon == h)]
                b = sub[(sub.model == "naive") & (sub.horizon == h)]
                if a.empty or b.empty:
                    continue
                lo, pt, hi = stats.block_bootstrap_ci(sub, m, "naive", h, "rel_ql", n_boot=500)
                no = stats.non_overlapping(a, h + extra)
                kp = stats.kupiec_pof(~no["covered"].to_numpy(), 0.20)[1] if len(no) > 5 else np.nan
                rows.append({"model": m, "group": g, "horizon": h, "n_origins": n, "rel_QL": a.ql.mean() / b.ql.mean(),
                             "rel_QL_lo": lo, "rel_QL_hi": hi, "DM_stat": stat, "p_better": p,
                             "coverage": a.covered.mean(), "kupiec_p": kp})
    t = pd.DataFrame(rows)
    if not t.empty:
        t["p_holm"] = t.groupby("model")["p_better"].transform(
            lambda s: stats.holm(s.fillna(1.0)).reindex(s.index))
    return t


def run_mode(panel, der, tg, models, mode, origins, out, live_origin, scorer) -> pd.DataFrame:
    t0 = time.time()
    lags = lags_for(mode, panel, tg, live_origin)
    fc = run_backtest(panel, tg, models, origins=origins, mode=mode, derive=der, lags=lags, verbose=True)
    meta_cfg = MetaConfig(shortlist=SHORTLIST)
    meta, diag = meta_pass(fc, meta_cfg, mode, lags=lags if mode == "realtime" else None)
    allfc = pd.concat([fc, meta], ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    allfc.to_parquet(out / "forecasts.parquet")
    diag.to_parquet(out / "meta_diag.parquet")
    scored = scorer(metrics.score_rows(allfc))
    metrics.summarize(scored).to_csv(out / "scores.csv")
    metrics.overview(metrics.summarize(scored)).to_csv(out / "overview.csv")
    stat_tables(scored, mode, lags).to_csv(out / "pooled_tests.csv", index=False)
    pick_stability(diag).to_csv(out / "pick_stability.csv", index=False)
    compute_footprint(fc.attrs["runtime_s"]).to_csv(out / "compute_footprint.csv")
    protocol.write_manifest(out, mode, origins, time.time() - t0, inputs=sorted(d.RAW.glob("*.xlsx")) +
                            sorted(d.PANEL.glob("*.csv")), configs=[d.ROOT / "config" / "assumptions.yaml",
                            d.ROOT / "config" / "availability.yaml"], use_licensed="L01" in panel)
    return scored


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dev", action="store_true")
    g.add_argument("--canaries", action="store_true")
    g.add_argument("--graded", action="store_true")
    g.add_argument("--submit", action="store_true")
    ap.add_argument("--mode", default="both", choices=["submission", "realtime", "both"])
    ap.add_argument("--no-licensed", action="store_true")
    args = ap.parse_args(argv)

    from src.external.signals import build_panel
    from .panel import load_backtest_config
    from .run import bom_deriver
    cfg = load_backtest_config()
    use_lic = cfg["use_licensed"] and not args.no_licensed
    live = cfg["live_origin"] + "-01"
    panel = build_panel(use_licensed=use_lic)
    der = bom_deriver(panel)
    tg = targets(panel)
    modes = ["submission", "realtime"] if args.mode == "both" else [args.mode]

    if args.submit:
        return submit(cfg, live)
    if args.canaries:
        return canaries(panel, der, tg, live)
    if args.graded:
        tag = protocol.assert_can_grade()
        origins = pd.date_range(protocol.DEV[0], protocol.GRADED[1], freq="MS")
        for mode in modes:
            run_mode(panel, der, tg, candidate_models(), mode, origins, OUT / "graded" / tag / mode, live,
                     lambda s: s[s.origin >= protocol.PURGE])
        print(f"graded run written to {OUT / 'graded' / tag}")
        return tag
    origins = pd.date_range(protocol.DEV[0], protocol.DEV[1], freq="MS")
    for mode in modes:
        run_mode(panel, der, tg, candidate_models(), mode, origins, OUT / "dev" / mode, live, protocol.dev_view)
    print(f"dev run written to {OUT / 'dev'}")


def canaries(panel, der, tg, live):
    from .canaries import run_canaries
    origins = [pd.Timestamp(o) for o in ("2012-06-01", "2016-03-01", "2020-06-01")]
    lags_by_mode = {m: lags_for(m, panel, tg, live) for m in ("submission", "realtime")}
    skill = pd.date_range("2012-01-01", "2021-09-01", freq="3MS")
    rep = run_canaries(panel, lambda p: p, tg, candidate_models(), der, lags_by_mode, origins, d.OUT / "canaries",
                       skill_origins=skill)
    print("CANARIES PASS" if rep["PASS"] else "CANARIES FAIL", {k: v.get("PASS", v) if isinstance(v, dict) else v
                                                                for k, v in rep["results"].items()})
    return rep


def submit(cfg: dict, live: str) -> pd.DataFrame:
    from .submission import from_forecasts, validate_submission
    tag = protocol.git_tag()
    src = OUT / "graded" / str(tag) / "submission" / "forecasts.parquet"
    if tag is None or not src.exists():
        raise SystemExit(f"run --graded on a tagged commit first (missing {src})")
    fc = pd.read_parquet(src)
    fc = fc[(fc.model == SUBMITTED) & (fc.origin == pd.Timestamp(live)) & fc.series.isin(CSV_TARGETS)]
    protocol.assert_publishable_ids(fc.series.unique())
    cat, _ = d.load_catalogs()
    o = live[:7]
    mats = [s for s in d.SCHNEIDER_RM_TARGETS if s in set(fc.series)]
    region = {s: ("EU" if s in ["S07", "S08", "S09", "S10", "S11"] else "WW") for s in mats}
    unit = {s: cat.loc[s, "unit"] for s in mats}
    name = f"{SUBMITTED} (walk-forward blend, tag {tag})"
    ai = "claude-code; public data: FRED/IMF/BLS, Eurostat, NY Fed GSCPI; licensed MEPS/CRU as inputs"
    a = from_forecasts(fc[fc.series.isin(mats)], o, cfg["team_id"], name, ai, region, unit)
    b = from_forecasts(fc[fc.series.isin(BOM_IDS)], o, cfg["team_id"], name, ai, {s: "EU" for s in BOM_IDS},
                       {s: f"index ({o}=100)" for s in BOM_IDS}, target_type="bom_index")
    sub = pd.concat([a, b], ignore_index=True)
    problems = validate_submission(sub)
    if problems:
        raise SystemExit(f"submission invalid: {problems}")
    path = OUT / "graded" / str(tag) / f"{cfg['team_id']}_forecast_{o.replace('-', '')}.csv"
    sub.to_csv(path, index=False)
    print(f"{len(sub)} rows -> {path}")
    return sub


from .candidates import SHORTLIST, candidate_models  # noqa: E402  (after the helpers, avoids an import cycle)

if __name__ == "__main__":
    main()
