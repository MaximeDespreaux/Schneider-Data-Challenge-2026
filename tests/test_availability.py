import pandas as pd
import pytest

from src import data as d
from src.licensed import LIC_COLUMNS
from src.testbench.availability import check_coverage, load_availability, ragged_edge_lags, realtime_lags
from src.testbench.vintage import information_set
from tests.conftest import needs_data


def test_every_known_column_has_a_row():
    from src.external.signals import EXTERNAL
    cols = (d.RM_TARGETS + ["S06"] + [f"E{i:02d}" for i in range(1, 25)] + [f"ED0{i}" for i in range(1, 7)]
            + list(EXTERNAL.values()) + ["X_GSCPI", "X_CU_TERM", "X_AL_TERM", "X_CU_STOCKS", "X_AL_STOCKS"]
            + list(LIC_COLUMNS) + list(d.EPOXY_COLUMNS)
            + ["RM6_NE_IQI_433520", "RMAirSeT_IIQ_24kV_IEC"])
    check_coverage(cols)


def test_missing_row_raises():
    with pytest.raises(KeyError, match="NEW_SERIES"):
        check_coverage(["S01", "NEW_SERIES"])


def test_table_schema_and_known_lags():
    av = load_availability()
    assert {"provider", "formation", "realtime_lag", "revision_months", "vintages", "licence"} <= set(av.columns)
    lags = realtime_lags(av)
    assert lags["S09"] == 2 and lags["S01"] == 1 and lags["X_VIX"] == 1 and lags["RM6_NE_IQI_433520"] == 2
    assert not av["vintages"].any()


def test_ragged_edge_mirrors_live_gap(panel):  # B ends 2 months before the last row, A is a target
    edge = ragged_edge_lags(panel, live_origin=panel.index[-1], exempt={"A"})
    assert edge == {"B": 2}
    info = information_set(panel, "2019-06-01", lags=edge)
    assert info["B"].last_valid_index() == pd.Timestamp("2019-04-01")
    assert info["A"].last_valid_index() == pd.Timestamp("2019-06-01")


def test_series_running_past_live_origin_is_cut_not_negative(panel):
    edge = ragged_edge_lags(panel, live_origin=panel.index[-5], exempt=set())
    assert min(edge.values()) == 0


def test_submission_mode_without_lags_is_plain_cut(panel):
    o = pd.Timestamp("2019-06-01")
    pd.testing.assert_frame_equal(information_set(panel, o), panel.loc[:o])


def test_realtime_default_lags_come_from_table():
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    p = pd.DataFrame({"S09": range(1, 25), "S01": range(1, 25)}, index=idx, dtype=float)
    info = information_set(p, idx[-1], mode="realtime")
    assert info["S09"].last_valid_index() == idx[-3] and info["S01"].last_valid_index() == idx[-2]


def test_realtime_unknown_column_raises():
    idx = pd.date_range("2020-01-01", periods=12, freq="MS")
    p = pd.DataFrame({"UNKNOWN": range(12)}, index=idx, dtype=float)
    with pytest.raises(KeyError, match="UNKNOWN"):
        information_set(p, idx[-1], mode="realtime")


@needs_data
@pytest.mark.xfail(strict=True, reason="Task 4 adds panel.build_full_panel")
def test_real_panel_fully_covered():
    from src.testbench.panel import build_full_panel
    check_coverage(build_full_panel(use_licensed=True).columns)
