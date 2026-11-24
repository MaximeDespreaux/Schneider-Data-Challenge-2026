"""Shared fixtures. Tests run on synthetic data; real-data tests are skipped when data/raw is absent (CI)."""

import numpy as np
import pandas as pd
import pytest

from src.data import RAW

needs_data = pytest.mark.skipif(not (RAW / "panel").exists(), reason="no challenge data")


def toy_derive(values_df, bases_df, origin):
    """Toy derived index (same logic as the BOM indices): weights and bases fixed at the origin."""
    known = bases_df.loc[:origin]
    ba, bb = known["A"].dropna().iloc[-1], known["B"].dropna().iloc[-1]
    wa = ba / (ba + bb)
    return pd.DataFrame({"IDX": 100 * (wa * values_df["A"] / ba + (1 - wa) * values_df["B"] / bb)},
                        index=values_df.index)


@pytest.fixture
def panel():
    """Two synthetic series, 120 months from 2015-01; B is published 2 months late (last 2 rows NaN)."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2015-01-01", periods=120, freq="MS")
    a = 100 * np.exp(np.cumsum(rng.normal(0.003, 0.04, 120)))
    b = 50 * np.exp(np.cumsum(rng.normal(0.0, 0.02, 120)))
    df = pd.DataFrame({"A": a, "B": b}, index=idx)
    df.iloc[-2:, 1] = np.nan
    return df
