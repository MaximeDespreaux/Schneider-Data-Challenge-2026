import numpy as np
import pandas as pd

from src.testbench.candidates import SHORTLIST, candidate_models


def test_candidate_set():
    models = candidate_models()
    names = [m.name for m in models]
    assert names[0] == "naive" and len(set(names)) == len(names)
    assert set(SHORTLIST) <= set(names)


def test_every_candidate_respects_the_contract():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2005-01-01", periods=200, freq="MS")
    exog = pd.DataFrame({"S09": 100 * np.exp(np.cumsum(rng.normal(0, .02, 200))),
                         "S04": 50 * np.exp(np.cumsum(rng.normal(0, .05, 200)))}, index=idx)
    y = exog["S09"].rename("S09")
    for m in candidate_models():
        out = m.forecast(y, [3, 6, 9], exog=exog)
        assert list(out.columns) == ["p10", "p50", "p90"] and list(out.index) == [3, 6, 9], m.name
        assert np.isfinite(out.to_numpy()).all() and (out.p10 <= out.p50 + 1e-9).all() and (out.p50 <= out.p90 + 1e-9).all(), m.name
