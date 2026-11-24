"""Extended champion: licensed market-vs-index gap and regime features (synthetic)."""

import numpy as np
import pandas as pd

from src.external.features import make_features
from src.external.models import ChampionWF


def _info(n=240, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-01", periods=n, freq="MS")
    coil = 500 * np.exp(np.cumsum(rng.normal(0, .04, n)))
    idxv = 100 * np.exp(np.r_[0, np.diff(np.log(coil))].cumsum() * 0.0)  # placeholder replaced below
    s09 = pd.Series(coil, index=idx).shift(1).bfill() / 5                 # index follows coil with a 1-month lag
    return pd.DataFrame({"S09": s09.to_numpy() * np.exp(rng.normal(0, .005, n)), "L01": coil}, index=idx)


def test_extended_features_present_only_when_asked():
    info = _info()
    base = make_features(info, "S09")
    ext = make_features(info, "S09", extended=True)
    assert {"lic_gap", "lic_gap_dev24", "mom_calm", "dev_hot"} <= set(ext.columns)
    assert not ({"lic_gap", "mom_calm"} & set(base.columns))


def test_extended_features_are_causal():
    info = _info()
    full = make_features(info, "S09", extended=True)
    for t in info.index[[100, 180, 239]]:
        cut = make_features(info.loc[:t], "S09", extended=True)
        pd.testing.assert_series_equal(cut.loc[t], full.loc[t], check_names=False)


def test_champion_wf2_name_and_contract():
    info = _info()
    m = ChampionWF(extended=True, name="champion_wf2")
    out = m.forecast(info["S09"].rename("S09"), [3, 6, 9], exog=info)
    assert m.name == "champion_wf2" and list(out.index) == [3, 6, 9]
    assert np.isfinite(out.to_numpy()).all() and (out.p10 <= out.p50).all() and (out.p50 <= out.p90).all()
