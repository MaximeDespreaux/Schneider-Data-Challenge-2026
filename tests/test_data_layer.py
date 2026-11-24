import pandas as pd
import pytest

from src import data as d


def test_deprecated_bom_needs_flag():
    with pytest.raises(ValueError, match="deprecated"):
        d.load_bom("csv")


def test_epoxy_ech_typo_corrected_x10_and_resin_flagged():
    idx = pd.date_range("2026-03-01", periods=4, freq="MS")
    df = pd.DataFrame({"EPX_ECH": [2500.0, 2480.0, 257.0, 2600.0],
                       "EPX_RESIN": [4900.0, 4930.0, 3600.0, 4980.0]}, index=idx)
    out = d.correct_epoxy(df)
    assert out.at[pd.Timestamp("2026-05-01"), "EPX_ECH"] == 2570.0
    assert out.at[pd.Timestamp("2026-05-01"), "EPX_RESIN"] == 3600.0  # left as delivered
    assert out.attrs["corrections"][0]["month"] == "2026-05"
    assert df.at[pd.Timestamp("2026-05-01"), "EPX_ECH"] == 257.0  # input not mutated


def test_no_publication_lag_table_left_in_data():
    assert not hasattr(d, "PUBLICATION_LAG")
