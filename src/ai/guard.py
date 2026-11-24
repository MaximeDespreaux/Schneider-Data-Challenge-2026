"""NDA gate: the only way data or text reaches an external LLM API.

Two layers, both must pass:

1. **Series allow-list.** Only series from public providers (IMF, Eurostat, ECB, US BLS, Ember, Bank of Japan,
   JEPX, and series derived from them) may be sent. CRU, MEPS, the LME workbook, the epoxy cost build-up, the
   BOM, suppliers and competitors never leave the machine. ``public_frame`` refuses anything else.
2. **Text check.** The full prompt is matched against the same deny-list as ``public/scan_leaks.py`` (private
   numbers, supplier / plant / part names, etc.) plus a few words that would reveal the client or licensed
   sources. ``check`` raises ``NDAViolation`` on any hit, before the request is built.
"""

from __future__ import annotations

import importlib.util
import re
from functools import lru_cache

import pandas as pd

from . import config as C

PUBLIC_PROVIDERS = ("IMF", "Eurostat", "European Central Bank", "US BLS", "Ember", "Bank of Japan", "JEPX", "Derived")
# Derived series are allowed only when every input is public (checked by id below).
DERIVED_OK = {"D01", "D02", "ED01", "ED02", "ED03", "ED04", "ED05", "ED06"}
# Words that would identify the client, the products or a licensed source in a prompt.
BLOCKED_WORDS = re.compile(r"\b(Schneider|AirSeT|RM6|NE-IQI|CRU|MEPS|BOM|supplier\w*|teardown|SAP|NDA)\b", re.I)


class NDAViolation(RuntimeError):
    """Raised when a payload for an external API contains non-public data."""


@lru_cache(maxsize=1)
def public_series() -> frozenset[str]:
    cat = pd.read_parquet(C.ROOT / "data" / "processed" / "series_catalog.parquet")
    ok = cat.provider.str.startswith(PUBLIC_PROVIDERS) & ~cat.provider.str.startswith("Derived")
    return frozenset(cat.index[ok]) | (DERIVED_OK & set(cat.index))


@lru_cache(maxsize=1)
def _patterns() -> dict:
    spec = importlib.util.spec_from_file_location("scan_leaks", C.ROOT / "public" / "scan_leaks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {"find": mod.find_leaks, "pats": mod.compile_patterns()}


def public_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` unchanged if every column is a public series, else raise."""
    bad = sorted(set(df.columns) - public_series())
    if bad:
        raise NDAViolation(f"non-public series in payload: {bad}")
    return df


def problems(text: str) -> list[str]:
    p = _patterns()
    out = [f"blocked word '{m.group(0)}'" for m in BLOCKED_WORDS.finditer(text)]
    for line in text.splitlines():
        out += p["find"](line, p["pats"])
    return out


def check(text: str) -> str:
    """Return ``text`` if it may be sent to an external API, else raise ``NDAViolation``."""
    found = problems(text)
    if found:
        raise NDAViolation("payload refused by the NDA gate:\n  " + "\n  ".join(found[:20]))
    return text
