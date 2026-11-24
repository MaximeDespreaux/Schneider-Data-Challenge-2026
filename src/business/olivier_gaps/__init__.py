"""Organiser clarifications of 23 Nov 2026 ("Olivier gaps"): three small modules that close the remaining gaps.

- ``lme_reference``       : which LME reference (cash, 3-month, indexed, fixed + trigger) best protects margin;
- ``sensitivity``         : sensitivity of the results to the unidentified BOM mass and to the proxies (brass, plastics,
                            BOM version);
- ``procurement_summary`` : one table of existing numbers for the procurement slide.

Outputs (confidential, gitignored): data/outputs/business/olivier_gaps/; figures in data/outputs/business/figures/.
Nothing is read or created at import time.
"""

from __future__ import annotations

from pathlib import Path

from .. import paths

OUT = paths.OUT / "olivier_gaps"


def out_dir() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT
