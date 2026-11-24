"""Run the three clarification modules in order and time each.

    uv run python -m src.business.clarifications.run

(``src.business.run_all`` also runs them, after ``suppliers``, and puts them in REPORT.md.)
"""

from __future__ import annotations

import importlib
import time

from ..config import load_business_config

MODULES = ["lme_reference", "sensitivity", "procurement_summary"]


def run(cfg: dict | None = None) -> dict:
    cfg = load_business_config() if cfg is None else cfg
    results = {}
    for name in MODULES:
        print(f"--- clarifications.{name}")
        t0 = time.perf_counter()
        head = importlib.import_module(f"src.business.clarifications.{name}").run(cfg)
        dt = time.perf_counter() - t0
        results[name] = {"runtime_s": round(dt, 2), **head}
        print(f"    OK ({dt:.1f} s)")
    return results


if __name__ == "__main__":
    for mod, head in run().items():
        print(f"\n[{mod}]")
        for k, v in head.items():
            print(f"  {k}: {v}")
