"""Tests of src/business/olivier_gaps/, on SYNTHETIC data only (they run with an empty data/ folder, as in CI)."""

import numpy as np
import openpyxl
import pandas as pd
import pytest

from src.business.olivier_gaps import lme_reference, procurement_summary, sensitivity


# ------------------------------------------------------------------------------------- 1. LME loader
def test_lme_loader_drops_provisional_month(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Monthly Prices"
    ws.append(["LME copper and aluminium (synthetic test file)"])
    ws.append([])
    ws.append(["notes"])
    ws.append([])
    ws.append(["Month", "Period status", "Cu cash EUR/t", "Cu 3m EUR/t", "Al cash EUR/t", "Al 3m EUR/t", "extra"])
    for i, status in enumerate(["Completed month"] * 3 + ["Month-to-date"]):
        ws.append([pd.Timestamp(2026, 7 + i, 1).to_pydatetime(), status, 100 + i, 101 + i, 50 + i, 51 + i, 0])
    path = tmp_path / "lme.xlsx"
    wb.save(path)

    df = lme_reference.load_lme_monthly(path)
    assert list(df.columns) == ["copper_cash", "copper_3m", "aluminium_cash", "aluminium_3m"]
    assert len(df) == 3 and df.index.max() == pd.Timestamp("2026-09-01")
    assert df.loc["2026-08-01", "copper_3m"] == 102


def test_lme_loader_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        lme_reference.load_lme_monthly(tmp_path / "nope.xlsx")


# ------------------------------------------------------------------------------- 2. mechanism E
def test_trigger_price():
    base = 1000.0
    assert lme_reference.trigger_price(base, 0.08, 0.05) == pytest.approx(base * 1.03)
    assert lme_reference.trigger_price(base, 0.04, 0.05) == pytest.approx(base)
    assert lme_reference.trigger_price(base, -0.08, 0.05) == pytest.approx(base * 0.97)


# --------------------------------------------------------------------------------- 3. no leakage
def _lme_series(n=60, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="MS")
    cash = pd.Series(8000 * np.exp(np.cumsum(rng.normal(0, 0.05, n))), index=idx)
    fwd = cash * (1 + rng.normal(0.003, 0.005, n))
    return cash, fwd


@pytest.mark.parametrize("h", [3, 6, 9])
def test_no_leakage(h):
    cash, fwd = _lme_series()
    t = cash.index[20]
    p0 = lme_reference.mechanism_prices(cash, fwd, h, 0.05, 3)

    # changing anything AFTER t leaves the signature prices / bases of A, B and E unchanged
    c1, f1 = cash.copy(), fwd.copy()
    c1[c1.index > t] *= 1.5
    f1[f1.index > t] *= 1.5
    p1 = lme_reference.mechanism_prices(c1, f1, h, 0.05, 3)
    for col in ["A_fixed_cash", "B_fixed_3m", "base_cash", "base_3m"]:
        assert p1.at[t, col] == p0.at[t, col]

    # changing months >= t+h (delivery and later) changes only the realised cost, never a price
    deliv = (t.to_period("M") + h).to_timestamp()
    c2, f2 = cash.copy(), fwd.copy()
    c2[c2.index >= deliv] *= 1.5
    f2[f2.index >= deliv] *= 1.5
    p2 = lme_reference.mechanism_prices(c2, f2, h, 0.05, 3)
    for col in lme_reference.MECHANISMS:
        assert p2.at[t, col] == pytest.approx(p0.at[t, col])
    assert p2.at[t, "cost"] == pytest.approx(1.5 * p0.at[t, "cost"])

    # C uses month t+h-1 (the latest one allowed)
    c3 = cash.copy()
    c3[(t.to_period("M") + h - 1).to_timestamp()] *= 1.5
    assert lme_reference.mechanism_prices(c3, fwd, h, 0.05, 3).at[t, "C_index_cash_m1"] != p0.at[t,
                                                                                                "C_index_cash_m1"]


def test_backtest_offer_window():
    cash, fwd = _lme_series()
    lme = pd.DataFrame({"copper_cash": cash, "copper_3m": fwd})
    off = lme_reference.backtest(lme, ["copper"], [6], "2018-06-01", 0.05, 3)
    assert off["offer_month"].min() == pd.Timestamp("2018-06-01")
    assert off["delivery_month"].max() <= lme.index.max()
    fixed = off[off["mechanism"].isin(["A_fixed_cash", "B_fixed_3m"])]
    assert (fixed["client_change"].abs() < 1e-12).all()


# ------------------------------------------------------------------------------- 4. sensitivity
CTX = {"eur": {"copper": 9.0, "carbon_steel": 0.8, "plastics": 2.5, "epoxy": 3.0, "brass": 0.63 * 9 + 0.37 * 3},
       "change": {"copper": 0.10, "carbon_steel": 0.05, "plastics": 0.02, "epoxy": 0.08},
       "brass_share_base": 0.63, "zinc_eur": 3.0, "zinc_change": 0.04}
KG = pd.Series({"copper": 10.0, "carbon_steel": 100.0, "brass": 3.0})


def test_unidentified_priced_as_expensive_material_raises_value():
    base = sensitivity.evaluate(sensitivity.build_lines(KG, CTX, 0.63), 5000)
    cheap = sensitivity.evaluate(sensitivity.build_lines(KG, CTX, 0.63, extra=("carbon_steel", 20)), 5000)
    dear = sensitivity.evaluate(sensitivity.build_lines(KG, CTX, 0.63, extra=("epoxy", 20)), 5000)
    assert base["bom_value_eur"] < cheap["bom_value_eur"] < dear["bom_value_eur"]


def test_brass_copper_share_raises_copper_exposure():
    lo = sensitivity.evaluate(sensitivity.build_lines(KG, CTX, 0.63), 5000)
    hi = sensitivity.evaluate(sensitivity.build_lines(KG, CTX, 0.70), 5000)
    assert hi["copper_share"] > lo["copper_share"]
    assert hi["bom_value_eur"] > lo["bom_value_eur"]  # copper dearer than zinc


# ------------------------------------------------------------------------- 5. procurement summary
def test_procurement_summary_from_fake_inputs(tmp_path, monkeypatch):
    bus, og = tmp_path / "business", tmp_path / "business" / "olivier_gaps"
    og.mkdir(parents=True)
    pd.DataFrame([{"product": p, "item": "copper", "kind": "material", "shock_pct": s, "delta_eur": 5.0 * s}
                  for p in ["RM6", "AirSeT"] for s in [10, 30]]).to_csv(bus / "shock_table.csv", index=False)
    pd.DataFrame({"product": ["RM6", "AirSeT"], "copper_share_2026_09": [0.5, 0.3]}).to_csv(
        bus / "convergence_summary.csv", index=False)
    pd.DataFrame([{"name": n, "sample": "full", "lag_50": 1, "lag_80": 3}
                  for n in procurement_summary.LAG_PAIRS]).to_csv(bus / "lag_table.csv", index=False)
    pd.DataFrame({"name": list("ABCD"), "priority_rank": [2, 1, 4, 3],
                  "confidence": ["low", "medium", "low", "high"]}).to_csv(bus / "supplier_estimates.csv",
                                                                         index=False)
    pd.DataFrame([{"metal": "copper", "period": "all", "horizon": h, "mechanism": m, "p90_err_pct": 10.0 + h}
                  for h in [3, 6, 9] for m in lme_reference.MECHANISMS]).to_csv(og / "lme_reference_summary.csv",
                                                                                index=False)
    pd.DataFrame({"metal": ["copper"], "mean_spread_pct": [0.3]}).to_csv(og / "lme_spread_stats.csv", index=False)
    monkeypatch.setattr(procurement_summary, "BUS_OUT", bus)
    monkeypatch.setattr(procurement_summary, "OG_OUT", og)

    head = procurement_summary.run({"products": ["RM6", "AirSeT"]})
    tab = pd.read_csv(og / "procurement_levers_summary.csv")
    assert list(tab.columns) == ["lever", "metric", "product_or_pair", "value", "unit", "source_file"]
    assert set(tab["lever"]) == {"copper exposure to hedge", "hedge reference", "back-to-back indexation",
                                 "negotiation timing", "supplier engagement order"}
    sup = tab[tab["lever"] == "supplier engagement order"]
    assert list(sup["product_or_pair"]) == ["B", "A", "D"]
    assert len(tab[tab["lever"] == "negotiation timing"]) == 8
    assert len(head) == len(tab)
