"""Task 10 — run every business module, time them, write data/outputs/business/REPORT.md.

    uv run python -m src.business.run_all

A failing module is reported as FAILED (traceback printed) and the others still run.
REPORT.md contains EUR per box, %, kg and headline figures only — never licensed steel prices.
"""

from __future__ import annotations

import importlib
import subprocess
import time
import traceback
from datetime import datetime

import numpy as np

from .config import load_business_config
from .paths import out_dir

MODULES = ["forecasts", "convergence", "shocks", "lags", "eaton", "electronics", "pricing_levers", "tco", "suppliers"]


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _plain(v):
    if isinstance(v, (np.floating, float)):
        return round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def run(cfg: dict | None = None) -> dict:
    from src.testbench.backtest import compute_footprint
    from src.testbench.targets import load_assumptions

    from .config import selling_prices

    cfg = load_business_config() if cfg is None else cfg
    results, runtimes, status = {}, {}, {}
    t_all = time.perf_counter()
    for name in MODULES:
        print(f"--- {name}")
        t0 = time.perf_counter()
        try:
            mod = importlib.import_module(f"src.business.{name}")
            head = mod.run(cfg)
            results[name] = {k: _plain(v) for k, v in head.items()}
            status[name] = "OK"
            print(f"    OK ({time.perf_counter() - t0:.1f} s)")
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            results[name] = {}
            status[name] = "FAILED"
            print(f"    FAILED ({time.perf_counter() - t0:.1f} s)")
        runtimes[name] = time.perf_counter() - t0
    total = time.perf_counter() - t_all

    out = out_dir()
    fp = compute_footprint(runtimes)
    fp.to_csv(out / "compute_footprint.csv")

    a = load_assumptions()
    prices = selling_prices(a, cfg)
    fc_source = results.get("forecasts", {}).get("source", "unknown")
    lines = ["# Business modules — run report", "",
             f"- Date/time: {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"- Git branch: {_git('rev-parse', '--abbrev-ref', 'HEAD')} @ {_git('rev-parse', '--short', 'HEAD')}",
             f"- Forecast source: {fc_source}",
             f"- BOM version: {a['bom']['version']}",
             f"- AirSeT selling price assumed: {'yes (RM6 price × (1 + ' + str(cfg['pricing']['airset_price_fallback_premium']) + '))' if prices['airset_price_assumed'] else 'no'}",
             f"- Total runtime: {total:.1f} s, ≈ {fp.loc['TOTAL', 'gCO2e']:.4f} gCO2e (estimate)", "",
             "## Module status", "", "| Module | Status | Runtime (s) |", "|---|---|---|"]
    lines += [f"| {m} | {status[m]} | {runtimes[m]:.2f} |" for m in MODULES]
    lines += ["", "## Headline figures", ""]
    for m in MODULES:
        lines += [f"### {m}", ""]
        if not results[m]:
            lines += ["_no headline (module failed)_", ""]
            continue
        lines += ["| Metric | Value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in results[m].items()]
        lines.append("")
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    n_ok = sum(s == "OK" for s in status.values())
    print(f"\nReport: {out / 'REPORT.md'}")
    print(f"BUSINESS RUN: {n_ok}/{len(MODULES)} modules OK — total {total:.1f} s, "
          f"{fp.loc['TOTAL', 'gCO2e']:.4f} gCO2e")
    return {"status": status, "runtimes": runtimes, "results": results}


if __name__ == "__main__":
    run()
