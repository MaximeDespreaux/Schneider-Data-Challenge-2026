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
