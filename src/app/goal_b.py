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


def sf6_scenario(draws, capex: float, eol_pct: float | None = None, premium: float | None = None):
    """Re-run the lifetime-cost draws with a fixed end-of-life gas loss (%) and/or a fixed AirSeT premium."""
    import numpy as np
    import pandas as pd

    from src.business import tco
    x = {k: draws[k].to_numpy(dtype=float) for k in tco.INPUTS}
    if eol_pct is not None:
        x["eol_loss_pct"] = np.full(len(draws), float(eol_pct))
    if premium is not None:
        x["airset_premium_pct"] = np.full(len(draws), float(premium))
    return pd.DataFrame(tco.compute(x, capex), index=draws.index)


def client_brief(cfg: dict, country: str, industry: str, numbers: dict) -> str:
    """Markdown brief for the client: pitch, model numbers, local context, sources. No Schneider pricing."""
    c, i = cfg["countries"][country], cfg["industries"][industry]

    def link(t: str, u: str) -> str:
        return f"{t} ([source]({u}))"

    pitch = [link(o["note"], o["source"]) for o in overrides(cfg, country, industry)] + list(i["pitch"])
    why = ([c["regulation_outlook"], link(c["disclosure"]["text"], c["disclosure"]["source"])]
           + [link(p["text"], p["source"]) for p in c["local_proof"]]
           + [cfg["pfas_point"]["text"] + " " + " ".join(f"([source]({u}))" for u in cfg["pfas_point"]["sources"])])
    nums = [f"AirSeT pays for itself through lower maintenance below a {numbers['breakeven']:.0%} price premium "
            f"over RM6, with no carbon price (our model, simulated 30-40 year lifetimes).",
            f"One RM6 emits about {numbers['sf6_t']:.1f} t CO2e of SF6 over its life and holds "
            f"{numbers['held_lo']:.0f}-{numbers['held_hi']:.0f} t CO2e of gas. AirSeT holds none (our model).",
            "SF6 leaks count in the owner's own direct emissions (Scope 1)."]
    out = [f"# RM AirSeT for {i['label'].lower()} in {c['label']}", "", "## Why AirSeT"]
    out += [f"- {p}" for p in pitch] + ["", "## The numbers"] + [f"- {n}" for n in nums]
    out += ["", "## Why now, here"] + [f"- {w}" for w in why]
    out += ["", "## SF6 rules spreading elsewhere"] + [f"- {link(r['text'], r['source'])}" for r in cfg["rules_spreading"]]
    out += ["", f"_Market facts as of {cfg['as_of']}. Regulation is an upside only, with no date assumed._", ""]
    return "\n".join(out)
