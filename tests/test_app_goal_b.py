"""Goal B selling logic: strength rule (with the price-only cap), premium band, override matching, real config."""

import pytest

from src.app import goal_b as gb

CFG = {
    "countries": {"A": {"disclosure_score": 2, "sf6_momentum": 1, "premium_modifier_pts": 0},
                  "B": {"disclosure_score": 0, "sf6_momentum": 0, "premium_modifier_pts": -2}},
    "industries": {"dc": {"esg_sensitivity": 2, "premium_band_pct": [16, 20]},
                   "price": {"esg_sensitivity": 0, "cap_strength": "MEDIUM", "premium_band_pct": [0, 10]},
                   "util": {"esg_sensitivity": 1, "premium_band_pct": [10, 16]}},
    "overrides": [{"country": "A", "industry": "dc", "note": "x", "source": "u"}],
}


@pytest.mark.parametrize("country,industry,level,score", [
    ("A", "dc", "STRONG", 5), ("A", "util", "STRONG", 4), ("B", "util", "WEAK", 1), ("B", "dc", "MEDIUM", 2),
    ("A", "price", "MEDIUM", 3)])
def test_strength_rule(country, industry, level, score):
    assert gb.strength(CFG, country, industry) == (level, score)


def test_price_only_is_capped_at_medium():
    cfg = {**CFG, "countries": {**CFG["countries"], "C": {"disclosure_score": 3, "sf6_momentum": 1}}}
    assert gb.strength(cfg, "C", "price")[0] == "MEDIUM"


def test_premium_band_applies_the_country_modifier_and_floors_at_zero():
    assert gb.premium_band(CFG, "A", "dc") == (16, 20)
    assert gb.premium_band(CFG, "B", "dc") == (14, 18)
    assert gb.premium_band(CFG, "B", "price") == (0, 8)


def test_overrides_match_the_pair_only():
    assert len(gb.overrides(CFG, "A", "dc")) == 1 and gb.overrides(CFG, "B", "dc") == []


def test_real_config_is_complete_and_every_claim_has_a_source():
    cfg = gb.load()
    for k, c in cfg["countries"].items():
        assert c["disclosure"]["source"].startswith("http") and all(p["source"].startswith("http") for p in c["local_proof"])
        for i in cfg["industries"]:
            assert gb.strength(cfg, k, i)[0] in gb.LEVELS
    assert all(o["source"].startswith("http") and o["country"] in cfg["countries"] and o["industry"] in cfg["industries"]
               for o in cfg["overrides"])
    assert gb.strength(cfg, "BR", "price_only")[0] == "WEAK" and gb.strength(cfg, "SA_AE", "data_centres")[0] == "STRONG"


def test_sf6_scenario_moves_tonnes_with_end_of_life_loss_and_cost_with_premium():
    from src.business import tco
    from src.business.config import load_business_config
    draws = tco.simulate(load_business_config()["tco"], 1000.0, seed=1, n=200)
    ours, high = gb.sf6_scenario(draws, 1000.0), gb.sf6_scenario(draws, 1000.0, eol_pct=40)
    assert (high["sf6_lifetime_tco2e"] > ours["sf6_lifetime_tco2e"]).all()
    assert abs(ours["sf6_lifetime_tco2e"] - draws["sf6_lifetime_tco2e"]).max() < 1e-9  # no override = same draws
    a, b = gb.sf6_scenario(draws, 1000.0, premium=0.10), gb.sf6_scenario(draws, 1000.0, premium=0.20)
    assert abs((b["diff"] - a["diff"]) - 100.0).max() < 1e-6  # +10 points of a 1000 EUR price


def test_client_brief_has_only_config_links_and_no_internal_pricing():
    import re
    cfg = gb.load()
    md = gb.client_brief(cfg, "US", "data_centres", {"breakeven": 0.16, "sf6_t": 3.0, "held_lo": 29, "held_hi": 79})
    urls = set(re.findall(r"\]\((http[^)]+)\)", md))
    assert urls and urls <= set(re.findall(r"https?://[^\s\"']+", gb.CONFIG.read_text(encoding="utf-8")))
    assert "premium band" not in md.lower() and "schneider only" not in md.lower() and "margin" not in md.lower()
