"""Shared-driver columns (phase 3) and analyst parsing (phase 4) with a fake LLM."""

import re

import numpy as np
import pandas as pd
import pytest

from src.ai import config as C


def test_one_driver_for_several_targets_keeps_each_targets_values():
    from src.ai.agent_eval import add_columns
    months = pd.date_range("2020-01-01", periods=6, freq="MS")
    idx = pd.MultiIndex.from_product([months, ["A", "B"]], names=["month", "series"])
    feats = {3: pd.DataFrame({"y": 0.0}, index=idx)}
    sig = pd.Series(np.arange(6.0), index=months)
    feats, extra = add_columns(feats, {("NEWS", "A"): sig, ("NEWS", "B"): sig * 10})
    f = feats[3]["x_NEWS_f"]
    assert f.xs("A", level="series").tolist() == list(np.arange(6.0))
    assert f.xs("B", level="series").tolist() == list(np.arange(6.0) * 10)
    assert extra == {"A": ["NEWS"], "B": ["NEWS"]}


@pytest.mark.skipif(not (C.ROOT / "data" / "processed" / "series_catalog.parquet").exists(), reason="needs local data")
def test_analyst_prompt_has_no_dates_and_parses(monkeypatch):
    from src import features as fe
    from src.ai import analyst, llm
    T, X = fe.build_panel()
    o = pd.Timestamp("2025-03-01")
    p = analyst.prompt(o, T, X)
    assert not re.search(r"20[12]\d", p)  # no year anywhere
    fake = {s: {**{f"h{h}": {"p10": 5.0, "p50": -1.0, "p90": -8.0} for h in (3, 6, 9)}, "rationale": "x"}
            for s in analyst.SERIES}
    monkeypatch.setattr(llm, "ask", lambda *a, **k: fake)
    f = analyst.forecast(o, T, X)
    assert len(f) == 7 * 3 and (f.q10 <= f.q50).all() and (f.q50 <= f.q90).all()  # unsorted input gets sorted
    assert np.allclose(f.q50, np.log(0.99))
