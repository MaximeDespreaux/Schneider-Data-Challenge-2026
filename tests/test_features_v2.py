"""Candidate lists v1 / v2 of the signal features, on a SYNTHETIC toy panel only."""

import numpy as np
import pandas as pd

from src.external.features import CROSS, CROSS_V2, make_features

CROSS_V1_FROZEN = ["S03", "S04", "S05", "D01", "D02", "ED01", "ED02", "E09", "E11", "E13"]


def _toy(cols):
    rng = np.random.default_rng(5)
    idx = pd.date_range("2000-01-01", periods=150, freq="MS")
    return pd.DataFrame({c: 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(idx)))) for c in cols}, index=idx)


def test_cross_none_equals_v1():
    assert CROSS == CROSS_V1_FROZEN
    info = _toy(["S09", "S03", "S04", "D01", "ED02", "E09", "E08", "ED03"])
    default = make_features(info, "S09")
    v1 = make_features(info, "S09", cross=CROSS_V1_FROZEN)
    pd.testing.assert_frame_equal(default, v1)
    assert not any(c.startswith(("cross_E08", "cross_ED03")) for c in default.columns)


def test_cross_v2_adds_only_existing_columns():
    info = _toy(["S09", "S03", "D01", "ED02", "E08", "ED03"])  # E14, E15, ... absent
    v1 = make_features(info, "S09")
    v2 = make_features(info, "S09", cross=CROSS_V2)
    extra = [c for c in v2.columns if c not in v1.columns]
    assert sorted(extra) == ["cross_E08_d1", "cross_E08_d3", "cross_ED03_d1", "cross_ED03_d3"]
    pd.testing.assert_frame_equal(v2[v1.columns], v1)  # v1 columns unchanged
