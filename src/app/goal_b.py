"""Goal B selling logic (AirSeT in markets with no SF6 ban): strength badge, premium band, matching notes.
Content lives in config/goal_b.yaml (public, sourced); model numbers are computed by the app from local outputs."""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[2] / "config" / "goal_b.yaml"
LEVELS = ("WEAK", "MEDIUM", "STRONG")


def load(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def strength(cfg: dict, country: str, industry: str) -> tuple[str, int]:
    """STRONG if score >= 4, MEDIUM 2-3, WEAK <= 1; an industry's ``cap_strength`` caps the level."""
    c, i = cfg["countries"][country], cfg["industries"][industry]
    score = int(c["disclosure_score"]) + int(c["sf6_momentum"]) + int(i["esg_sensitivity"])
    level = "STRONG" if score >= 4 else "MEDIUM" if score >= 2 else "WEAK"
    cap = i.get("cap_strength")
    if cap and LEVELS.index(level) > LEVELS.index(cap):
        level = cap
    return level, score


def premium_band(cfg: dict, country: str, industry: str) -> tuple[float, float]:
    """Industry premium band (%) shifted by the country modifier, never below 0."""
    lo, hi = cfg["industries"][industry]["premium_band_pct"]
    m = float(cfg["countries"][country].get("premium_modifier_pts", 0))
    return max(0.0, lo + m), max(0.0, hi + m)


def overrides(cfg: dict, country: str, industry: str) -> list[dict]:
    return [o for o in cfg.get("overrides", []) if o["country"] == country and o["industry"] == industry]
