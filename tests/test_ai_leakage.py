"""Look-ahead checks of the AI track and of the benchmark it builds on (synthetic data, no NDA data, no LLM call)."""

import re

import numpy as np
import pandas as pd
import pytest

from src import backtest as bt
from src import features as fe
from src import models as mo
from src.ai import backtest as ab
from src.ai import config as C
from src.ai import run as ai_run


# ------------------------------------------------------------------------------------------ publication lags
def test_published_by_matches_the_release_calendar():
    months = pd.Series(pd.to_datetime(["2021-04-01", "2021-05-01", "2021-06-01", "2021-07-01"]))
    o = pd.Timestamp("2021-06-01")
    same = fe.published_by(months, pd.Series(["S09"] * 4), o, "S09")
    assert same.tolist() == [True, True, True, False]                     # own series: month <= origin
    ppi_for_steel = fe.published_by(months, pd.Series(["S09"] * 4), o, "CRU_CR_DE")
    assert ppi_for_steel.tolist() == [True, False, False, False]          # PPI comes out 2 months after the CRU price
    steel_for_ppi = fe.published_by(months, pd.Series(["CRU_CR_DE"] * 4), o, "S09")
    assert steel_for_ppi.tolist() == [True, True, True, False]            # out by then, but never after the origin


def test_bank_of_japan_series_are_not_known_at_month_end():
    assert fe.pub_lag("E04") >= 1 and fe.pub_lag("E19") >= 1
    assert not fe.has_declared_lag("NOT_A_SERIES")


def _rows(series, months, resid, h=3):
    o = pd.DatetimeIndex(months)
    return pd.DataFrame(dict(series=series, origin=o, h=h, model="m", pred=0.0, q10=-0.1, q90=0.1, actual=resid,
                             target_month=o + pd.DateOffset(months=h)))


@pytest.mark.parametrize("band", ["conformal", "cqr"])
def test_pooled_bands_ignore_unpublished_outcomes(band):
    """A CRU forecast with too few own residuals pools the PPI ones; those not yet published must not matter."""
    rng = np.random.default_rng(0)
    o = pd.Timestamp("2021-06-01")
    ppi = _rows("S09", pd.date_range("2019-01-01", "2021-03-01", freq="MS"), rng.normal(0, 0.05, 27))
    cru = _rows("CRU_CR_DE", [o], [0.0])
    unpublished = ppi.target_month > o - pd.DateOffset(months=2)
    assert (unpublished & (ppi.target_month <= o)).any()  # realised, but not published: the case that used to leak
    ppi2 = ppi.copy()
    ppi2.loc[unpublished, "actual"] += 5.0
    out = []
    for p in (ppi, ppi2):
        res = pd.concat([p, cru], ignore_index=True)
        r = bt.add_conformal(res, "m") if band == "conformal" else ab.add_cqr(res, "m").query("model == 'm+cqr'")
        out.append(r[r.series == "CRU_CR_DE"][["q10", "q90"]].to_numpy())
    assert np.isfinite(out[0]).all()
    np.testing.assert_allclose(out[0], out[1])


def test_pooled_lightgbm_ignores_unpublished_outcomes():
    rng = np.random.default_rng(1)
    months = pd.date_range("2012-01-01", "2021-06-01", freq="MS")
    o, h = months[-1], 3
    idx = pd.MultiIndex.from_product([months, ["CRU_CR_DE", "S09"]], names=["month", "series"])
    feat = pd.DataFrame({"family": ["steel_market", "ppi"] * len(months), "y_level": 100.0,
                         "y": rng.normal(0, 0.05, len(idx)), "own_dlog_l0": rng.normal(size=len(idx))}, index=idx)
    feat2 = feat.copy()
    late = feat2.index.get_level_values("month") > o - pd.DateOffset(months=h + 2)
    feat2.loc[late & (feat2.index.get_level_values("series") == "S09"), "y"] += 5.0
    a, b = mo.lgbm_predict(feat, o, h), mo.lgbm_predict(feat2, o, h)
    assert set(a.index) == {"CRU_CR_DE", "S09"}
    np.testing.assert_allclose(a.loc["CRU_CR_DE"], b.loc["CRU_CR_DE"])
    assert not np.allclose(a.loc["S09"].astype(float), b.loc["S09"].astype(float))  # control: S09 may use its own rows


# ------------------------------------------------------------------------------------------ selection and live
def _two_models(n=72):
    rng = np.random.default_rng(2)
    rows = []
    for m, w in [("naive", 0.05), ("model", 0.08)]:
        for t in pd.date_range("2019-01-01", periods=n, freq="MS"):
            rows.append(dict(series="X", origin=t, h=9, model=m, pred=0.0, q10=-w, q90=w, actual=rng.normal(0, 0.05),
                             target_month=t + pd.DateOffset(months=9)))
    return pd.DataFrame(rows)


def test_selected_rows_start_after_every_outcome_used_to_choose():
    split = pd.Timestamp("2022-01-01")
    res = _two_models()
    out, choice = ab.select_preregistered(res, ["naive", "model"], split=str(split.date()))
    sel = out[out.model == "ai_selected"]
    assert len(sel) and (sel.origin >= split).all()
    used = res[res.target_month < split].target_month.max()
    assert sel.origin.min() > used
    res2 = res.copy()  # outcomes after the split cannot change the choice
    res2.loc[res2.target_month >= split, "actual"] += 1.0
    pd.testing.assert_frame_equal(choice, ab.select_preregistered(res2, ["naive", "model"], split=str(split.date()))[1])


def test_live_bands_only_see_outcomes_up_to_the_origin():
    res = _two_models()
    cut = pd.Timestamp("2023-06-01")
    kept = ai_run.known_at(res, cut)
    assert kept.target_month.max() <= cut and len(kept) < len(res)


def test_main_passes_only_known_rows_to_the_live_stage(monkeypatch, tmp_path):
    idx = pd.date_range("2019-01-01", "2026-10-01", freq="MS")
    T = pd.DataFrame({"X": 1.0}, index=idx)
    res = pd.concat([_two_models(), _rows("X", pd.date_range("2025-01-01", "2026-07-01", freq="MS"), 0.0)])
    choice = pd.DataFrame({"series": ["X"], "h": [9], "chosen": ["naive"]})
    seen = {}

    def live(T_, X_, res_, choice_):
        seen.update(T=T_.index.max(), X=X_.index.max(), res=res_.target_month.max())
        raise StopIteration

    monkeypatch.setattr(C, "OUT", tmp_path)
    monkeypatch.setattr(fe, "build_panel", lambda: (T, T))
    monkeypatch.setattr(ai_run, "backtest_stage", lambda *a: (res, {}, choice))
    monkeypatch.setattr(ai_run, "report_stage", lambda r: {})
    monkeypatch.setattr(ai_run, "live_stage", live)
    with pytest.raises(StopIteration):
        ai_run.main()
    assert max(seen.values()) <= pd.Timestamp(C.LIVE_ORIGIN) < res.target_month.max()


# ------------------------------------------------------------------------------------------ formula language
@pytest.mark.parametrize("formula", ["dev_ma(E09, 12)", "dev_ma(ratio(TARGET, E09), 12)", "dev_ma(spread(E09, S06), 12)",
                                     "zscore(E09, 12)", "vol(E09, 6)"])
def test_formulas_ignore_a_future_sign_change(monkeypatch, formula):
    from src.ai import feature_dsl as dsl
    from src.ai import guard
    monkeypatch.setattr(guard, "public_series", lambda: frozenset({"E09", "S06"}))
    rng = np.random.default_rng(4)
    idx = pd.date_range("2012-01-01", "2026-09-01", freq="MS")
    walk = lambda base: base * np.exp(np.cumsum(rng.normal(0, 0.03, len(idx))))  # noqa: E731
    T = pd.DataFrame({"CRU_CR_DE": walk(100)}, index=idx)
    X = pd.DataFrame({"E09": walk(1000), "S06": walk(1)}, index=idx)
    o = pd.Timestamp("2021-06-01")
    X2 = X.copy()
    X2.loc[X2.index > o, "E09"] *= -1
    a = dsl.compile_feature(formula, "CRU_CR_DE", T, X).loc[:o]
    assert a.notna().sum() > 60
    pd.testing.assert_series_equal(a, dsl.compile_feature(formula, "CRU_CR_DE", T, X2).loc[:o])


# ------------------------------------------------------------------------------------------ LLM inputs
def test_analyst_prompt_only_uses_published_months(monkeypatch):
    from src.ai import analyst, guard
    rng = np.random.default_rng(5)
    idx = pd.date_range("2020-01-01", "2026-09-01", freq="MS")
    walk = lambda: 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(idx))))  # noqa: E731
    T = pd.DataFrame({s: walk() for s in analyst.SERIES}, index=idx)
    X = pd.DataFrame({s: walk() for s in analyst.CONTEXT}, index=idx)
    cat = pd.DataFrame({"description": "x"}, index=analyst.SERIES)
    monkeypatch.setattr(analyst.pd, "read_parquet", lambda *a, **k: cat)
    monkeypatch.setattr(guard, "public_frame", lambda d: d)
    monkeypatch.setattr(guard, "problems", lambda t: [])
    d = pd.Timestamp("2025-06-01")
    T2, X2 = T.copy(), X.copy()
    for df in (T2, X2):
        for c in df:
            df.loc[df.index > d - pd.DateOffset(months=fe.pub_lag(c)), c] *= 3.0
    assert analyst.prompt(d, T, X) == analyst.prompt(d, T2, X2)
    fake = {s: {**{f"h{h}": {"p10": -5.0, "p50": 0.0, "p90": 5.0} for h in C.HORIZONS}, "rationale": ""}
            for s in analyst.SERIES}
    monkeypatch.setattr(analyst.llm, "ask", lambda *a, **k: fake)
    f = analyst.forecast(d, T, X).drop_duplicates("series").set_index("series").origin
    assert f["D01"] == pd.Timestamp("2025-05-01") and f["S09"] == pd.Timestamp("2025-04-01")


def test_news_prompt_has_no_year_and_skips_the_running_month():
    from src.ai import news
    titles = ["Steel outlook for 2022", "Copper in Q3 '21 falls", "FY23 nickel output", "Price up 12 percent",
              "2021-22 crisis", "Plant makes 20000 t"]
    p = news._prompt({"steel": titles})
    assert not re.search(r"(?<!\d)(19|20)\d{2}(?!\d)|'21|FY23", p)
    assert "Price up 12 percent" in p and "20000 t" in p
    assert news._months().max() < pd.Timestamp.today().normalize().replace(day=1)


def test_answer_from_another_model_is_refused():
    from src.ai import llm
    llm._same_model("claude-opus-5-5", "claude-opus-5-5")
    with pytest.raises(RuntimeError, match="answered by"):
        llm._same_model("another-model", "claude-opus-5-5")


def test_unverified_release_date_gives_no_clean_window():
    assert C.llm_released("claude-opus-5-5") is None and C.llm_released("unknown") is None
    assert C.llm_released("qwen3:14b") == pd.Timestamp("2025-04-29")


# ------------------------------------------------------------------------------------------ nothing after the origin
@pytest.mark.parametrize("target", ["D01", "S09"])
def test_pooled_lightgbm_of_a_lagged_series_ignores_everything_after_its_origin(target):
    """A lagged series is forecast when published, i.e. after its origin month: steel outcomes dated in between must
    still not be used (the live forecast is cut at the origin)."""
    rng = np.random.default_rng(6)
    months = pd.date_range("2012-01-01", "2021-06-01", freq="MS")
    o, h = months[-1], 3
    idx = pd.MultiIndex.from_product([months, ["CRU_CR_DE", target]], names=["month", "series"])
    feat = pd.DataFrame({"family": ["steel_market", "other"] * len(months), "y_level": 100.0,
                         "y": rng.normal(0, 0.05, len(idx)), "own_dlog_l0": rng.normal(size=len(idx))}, index=idx)
    feat2 = feat.copy()
    feat2.loc[feat2.index.get_level_values("month") > o - pd.DateOffset(months=h), "y"] += 5.0
    np.testing.assert_allclose(mo.lgbm_predict(feat, o, h), mo.lgbm_predict(feat2, o, h))


@pytest.mark.parametrize("band", ["conformal", "cqr"])
def test_pooled_bands_of_a_lagged_series_ignore_everything_after_its_origin(band):
    rng = np.random.default_rng(7)
    o = pd.Timestamp("2021-06-01")
    cru = _rows("CRU_CR_DE", pd.date_range("2019-01-01", "2021-05-01", freq="MS"), rng.normal(0, 0.05, 29))
    ppi = _rows("S09", [o], [0.0])
    cru2 = cru.copy()
    after = cru2.target_month > o
    assert after.any()
    cru2.loc[after, "actual"] += 5.0
    out = []
    for c in (cru, cru2):
        res = pd.concat([c, ppi], ignore_index=True)
        r = bt.add_conformal(res, "m") if band == "conformal" else ab.add_cqr(res, "m").query("model == 'm+cqr'")
        out.append(r[r.series == "S09"][["q10", "q90"]].to_numpy())
    assert np.isfinite(out[0]).all()
    np.testing.assert_allclose(out[0], out[1])


# ------------------------------------------------------------------------------------------ submission file
def test_ai_csv_never_carries_a_licensed_series():
    from src.external.run_champion import TEAM_ID
    from src.testbench.submission import validate_submission
    live = pd.DataFrame([dict(series=s, origin=pd.Timestamp("2026-09-01"), h=h, model="naive",
                              target_month=pd.Timestamp("2026-09-01") + pd.DateOffset(months=h), p10=1.0, p50=2.0, p90=3.0)
                         for s in fe.TARGETS for h in C.HORIZONS])
    sub = ai_run.submission_csv(ai_run.publishable(live))
    assert set(sub.target_id) == {"D01", "D02", "S03", "S04", "S09", "S10", "S11"}
    assert set(sub.team_id) == {TEAM_ID} and validate_submission(sub) == []
    with pytest.raises(ValueError, match="not publishable"):
        ai_run.submission_csv(live)
