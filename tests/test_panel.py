import pandas as pd
import pytest

from src.testbench.backtest import run_backtest
from src.testbench.panel import load_backtest_config
from tests.conftest import needs_data


class Spy:
    name = "spy"

    def __init__(self, targets=None):
        if targets is not None:
            self.targets = targets
        self.last = []

    def forecast(self, y, steps, exog=None):
        self.last.append(y.index[-1])
        v = float(y.iloc[-1])
        return pd.DataFrame({"p10": v, "p50": v, "p90": v}, index=pd.Index(steps, name="step"))


def test_missing_backtest_section_message():
    with pytest.raises(KeyError) as e:
        load_backtest_config({"bom": {}})
    assert "backtest" in str(e.value) and "assumptions.example.yaml" in str(e.value)


def test_backtest_config_fields():
    cfg = load_backtest_config({"backtest": {"use_licensed": True, "live_origin": "2026-09", "team_id": "t",
                                             "availability_file": "config/availability.yaml"}})
    assert cfg["use_licensed"] is True and cfg["live_origin"] == "2026-09"


def test_model_targets_filter(panel):
    fc = run_backtest(panel, ["A", "B"], [Spy(targets=("A",))], origins=["2020-01-01"], verbose=False)
    assert set(fc["series"]) == {"A"}


def test_lags_passthrough(panel):
    o = pd.Timestamp("2020-06-01")
    fc = run_backtest(panel, ["A"], [Spy()], origins=[o], lags={"A": 2}, verbose=False)
    assert (fc["last_obs"] == pd.Timestamp("2020-04-01")).all()


@needs_data
@pytest.mark.parametrize("lic", [False, True])
def test_build_full_panel(lic):
    from src.testbench.panel import build_full_panel
    p = build_full_panel(use_licensed=lic)
    assert not any(c in p.columns for c in ["L04", "L05", "L06", "L07", "L08"])
    assert ("L01" in p.columns) == lic and ("EP01" in p and "BR01" in p)
    assert "EPX_ECH" in p and p.index.min() == pd.Timestamp("1992-01-01")
    assert p.index.is_unique and p.index.is_monotonic_increasing


@needs_data
def test_brass_proxy_extended_with_its_inputs():
    from src.testbench.panel import build_full_panel
    p = build_full_panel(use_licensed=False)
    assert p["BR01"].first_valid_index() <= pd.Timestamp("2000-01-01")
