"""Figures of the AI track (synthetic data): files written, licensed series kept apart, nothing after the origin."""

import numpy as np
import pandas as pd
import pytest

from src.ai import config as C
from src.ai import plots as P

SERIES = ["D01", "S09", "CRU_CR_DE", "SS304_ALLOY_DE"]
MODELS = ["naive_vol", "chronos-2+cov", "ai_selected"]


@pytest.fixture
def outputs(tmp_path):
    origin = pd.Timestamp(C.LIVE_ORIGIN)
    live = pd.DataFrame([dict(series=s, origin=origin, last_obs=origin, step=h, h=h, model="naive_vol", y_origin=100.0,
                              p10=90.0, p50=101.0, p90=115.0, target_month=origin + pd.DateOffset(months=h))
                         for s in SERIES for h in C.HORIZONS])
    live.to_parquet(tmp_path / "ai_live_forecast.parquet")
    months = pd.date_range("2024-01-01", periods=12, freq="MS")
    bt = pd.DataFrame([dict(series=s, origin=o, h=6, model="ai_selected", pred=0.01, q10=-0.1, q50=0.01, q90=0.1,
                            target_month=o + pd.DateOffset(months=6), y_origin=100.0, y_true=102.0,
                            actual=np.log(1.02)) for s in SERIES for o in months])
    bt.to_parquet(tmp_path / "ai_backtest.parquet")
    score = pd.DataFrame(dict(model=MODELS, n=10, relMAE=[1.0, 0.9, 0.95], relQL=[0.8, 1.0, 0.84], coverage=0.8,
                              width_pct=30.0))
    for f in P.INPUTS[2:6]:
        df = score[score.model != "naive_vol"] if "naive_vol" in f else score
        df.to_csv(tmp_path / f, index=False)
    pd.concat([score.assign(family=f) for f in ("lme", "ppi")]).to_csv(
        tmp_path / "ai_scores_by_family_from_2024.csv", index=False)
    return tmp_path


@pytest.fixture
def panel_T():
    idx = pd.date_range("2020-01-01", "2026-12-01", freq="MS")  # goes past the official origin
    return pd.DataFrame({s: np.linspace(90, 110, len(idx)) for s in SERIES}, index=idx)


def test_every_figure_is_written(outputs, panel_T):
    paths = P.build(P.load(outputs), panel_T, outputs / "figures")
    assert {p.stem for p in paths} == {
        "01_scores", "02_scores_by_family",
        *(f"{n}_{g}" for n in ("03_backtest", "04_forecast", "05_forecast_change")
          for g in ("csv_series", "LICENSED_internal"))}
    assert all(p.stat().st_size > 5_000 for p in paths)


def test_licensed_series_are_never_drawn_with_the_others(outputs, panel_T, monkeypatch):
    drawn = {}

    def save(fig, out, stem):
        drawn[stem] = " ".join(t.get_text() for ax in fig.axes for t in [ax.title, *ax.get_yticklabels(),
                                                                         ax._left_title])
        P.plt.close(fig)
        return out / stem
    monkeypatch.setattr(P, "_save", save)
    P.build(P.load(outputs), panel_T, outputs / "figures")
    for stem, text in drawn.items():
        if stem.endswith("csv_series"):
            assert "D01" in text and "CRU" not in text and "SS304" not in text
        if stem.endswith("LICENSED_internal"):
            assert "CRU_CR_DE" in text and "D01" not in text
    assert P.groups(["D01", "MEPS_X", "L02"])["csv_series"] == ["D01"]


def test_history_stops_at_the_official_origin(panel_T):
    assert P.history(panel_T, "D01").index.max() == pd.Timestamp(C.LIVE_ORIGIN)


def test_range_quality_vs_naive_vol_is_a_ratio_on_the_same_rows(outputs):
    fam = P.load(outputs)["ai_scores_by_family_from_2024"]
    _, rng = P.family_tables(fam)
    assert rng.loc["chronos-2+cov", "lme"] == pytest.approx(1.0 / 0.8)
    fam.loc[fam.model == "ai_selected", "n"] = 9
    with pytest.raises(ValueError, match="same rows"):
        P.family_tables(fam)


def test_missing_or_stale_inputs_stop_with_a_clear_message(outputs):
    live = pd.read_parquet(outputs / "ai_live_forecast.parquet")
    live.assign(origin=pd.Timestamp("2026-08-01")).to_parquet(outputs / "ai_live_forecast.parquet")
    with pytest.raises(SystemExit, match="make ai"):
        P.load(outputs)
    (outputs / "ai_backtest.parquet").unlink()
    with pytest.raises(SystemExit, match="ai_backtest.parquet.*make ai"):
        P.load(outputs)
