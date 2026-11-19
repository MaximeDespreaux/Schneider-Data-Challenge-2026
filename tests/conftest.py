"""Shared fixtures. Tests run on synthetic data; real-data tests are skipped when data/raw is absent (CI)."""

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
