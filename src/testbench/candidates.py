"""Pre-registered candidate set for the walk-forward base pass, and the wf_median shortlist (spec 5.5)."""

from __future__ import annotations

from .driver_models import DriverRidge
from .models import (AR1Point, Conformal, DampedTrendETS, DriftPoint, MeanReversionPoint, Naive, NaiveGaussian,
                     NaiveVolScaled)

SHORTLIST = ("naive", "naive_vol_hl3", "ar1_conf", "driver_ridge", "champion_wf2")
# Submitted model (user decision 2026-11-22, before grading): the walk-forward champion; extended signal set
# (champion_wf2) adopted by the pre-written dev rule over champion_wf (pooled dev loss 0.95874 vs 0.95884 of naive).
SUBMITTED = "champion_wf2"


def candidate_models() -> list:
    """``naive`` first: it is the reference of every comparison and the meta-pass fallback."""
    from src.external.models import ChampionWF
    return [Naive(), NaiveGaussian(),
            NaiveVolScaled(halflife=3, name="naive_vol_hl3"), NaiveVolScaled(halflife=6, name="naive_vol_hl6"),
            NaiveVolScaled(halflife=12, name="naive_vol_hl12"),
            Conformal(DriftPoint()), Conformal(AR1Point()), Conformal(MeanReversionPoint()),
            DriverRidge(), DampedTrendETS(), ChampionWF(extended=True, name="champion_wf2")]
