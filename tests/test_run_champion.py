"""make champion scores the walk-forward champion on inputs masked by the live ragged edge (no global screen)."""

import pandas as pd
import pytest

from src.external import run_champion as rc
from src.external.models import ChampionWF


def test_scored_champion_is_walk_forward_and_inputs_are_masked(monkeypatch, tmp_path):
    seen = {}

    def fake_backtest(panel, targets, models, **kw):
        seen.update(models={m.name: m for m in models}, lags=kw.get("lags"))
        raise StopIteration

    idx = pd.date_range("2020-01-01", "2026-09-01", freq="MS")
    panel = pd.DataFrame({"S09": 1.0, "L02": 1.0}, index=idx)
    panel.loc[idx[-1], "L02"] = float("nan")                                       # one month late at the live edge
    monkeypatch.setattr(rc.d, "OUT", tmp_path)
    monkeypatch.setattr(rc, "build_panel", lambda: panel)
    monkeypatch.setattr(rc, "bom_deriver", lambda p: None)
    monkeypatch.setattr(rc, "default_targets", lambda p: ["S09"])
    monkeypatch.setattr(rc, "screen", lambda *a, **k: pd.DataFrame({"x": []}))
    monkeypatch.setattr(rc, "run_backtest", fake_backtest)
    with pytest.raises(StopIteration):
        rc.main()
    assert isinstance(seen["models"]["champion"], ChampionWF)
    assert seen["lags"]["L02"] == 1 and "S09" not in seen["lags"]
