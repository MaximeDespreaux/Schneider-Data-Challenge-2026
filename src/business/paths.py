"""Output folders of the business modules (confidential, gitignored). Created on demand, never at import."""

from __future__ import annotations

from pathlib import Path

from src import data as d

OUT = d.OUT / "business"
FIG = OUT / "figures"


def out_dir() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def fig_dir() -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    return FIG
