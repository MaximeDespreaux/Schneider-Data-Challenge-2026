"""Gap 1 — which LME reference best protects margin while remaining commercially usable?

    uv run python -m src.business.clarifications.lme_reference

Data: monthly AVERAGE LME cash and 3-month prices in EUR/t (challenge file, sheet ``Monthly Prices``). Only
``Period status == "Completed month"`` rows are kept: the last row (Nov 2026, "Month-to-date") is provisional.

ASSUMPTION: Schneider buys the metal at the delivery month t+h at that month's average LME cash price,
``cost = cash(t+h)``. Each mechanism is the price charged for the metal content (EUR/t), set with information
available at the time the price is set (no future leakage):

- ``A_fixed_cash``            : fixed at signature at cash(t);
- ``B_fixed_3m``              : fixed at signature at 3m(t), the market price for delivery in 3 months, known at t.
                                For h = 6 and 9 the 3-month price is still the latest forward available in the data;
- ``C_index_cash_m1``         : indexed to cash(t+h-1) (one-month publication lag);
- ``D_index_cash_avg3``       : indexed to the average of cash(t+h-3 ... t+h-1);
- ``E_fixed_3m_plus_trigger`` : base 3m(t); at delivery x = avg3 / cash(t) - 1 (avg3 as in D); if |x| <= trigger the
                                price stays at the base, else the part beyond the trigger is passed on both ways:
                                price = base × (1 + x - sign(x) × trigger).

Margin effect per offer, in % of the metal cost: err = (cost - price) / cost (positive = Schneider loses margin).
Client-price variability: client_change = price / base_at_signature - 1 (base = cash(t) for A, C, D; 3m(t) for B, E).
Fixed prices have 0 variability by construction: this is what makes them commercially attractive.

Offer months t run from ``start`` to (last completed month - h): realised costs use completed months only.
Spread statistics describe what is known at the origin (months <= 2026-09).
Outputs in data/outputs/business/clarifications/ (gitignored) and figure lme_reference.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import data as d  # noqa: E402

from ..config import load_business_config, origin_ts, selling_prices  # noqa: E402
from ..paths import fig_dir  # noqa: E402
from . import out_dir  # noqa: E402

LME_FILE = "LME_Copper_Aluminium_EU_Monthly_2016_2026.xlsx"
COLS = {"Cu cash EUR/t": "copper_cash", "Cu 3m EUR/t": "copper_3m",
        "Al cash EUR/t": "aluminium_cash", "Al 3m EUR/t": "aluminium_3m"}
MECHANISMS = ["A_fixed_cash", "B_fixed_3m", "C_index_cash_m1", "D_index_cash_avg3", "E_fixed_3m_plus_trigger"]
USABLE = ["A_fixed_cash", "B_fixed_3m", "E_fixed_3m_plus_trigger"]  # fixed, or fixed with a trigger clause


def load_lme_monthly(path: Path | str | None = None) -> pd.DataFrame:
    """Completed months only, indexed by month start: copper_cash, copper_3m, aluminium_cash, aluminium_3m (EUR/t)."""
    path = d.RAW / LME_FILE if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LME file not found: {path}. Put the challenge file {LME_FILE} in {d.RAW}/ "
                                "(run `make data-check`).")
    df = pd.read_excel(path, sheet_name="Monthly Prices", header=4)
    df = df[df["Period status"].astype(str).str.strip() == "Completed month"].copy()
    df["month"] = pd.to_datetime(df["Month"]).dt.to_period("M").dt.to_timestamp()
    out = df.set_index("month")[list(COLS)].rename(columns=COLS).astype(float).sort_index()
    out.index.name = "month"
    return out


def trigger_price(base, x, trigger: float):
    """Mechanism E: base if |x| <= trigger, else base × (1 + x - sign(x) × trigger) (trigger as a fraction)."""
    base, x = np.asarray(base, dtype=float), np.asarray(x, dtype=float)
    return np.where(np.abs(x) <= trigger, base, base * (1 + x - np.sign(x) * trigger))


def mechanism_prices(cash: pd.Series, fwd: pd.Series, h: int, trigger: float, n_avg: int = 3) -> pd.DataFrame:
    """One row per offer month t (index): cost = cash(t+h), the price of each mechanism and the signature bases.

    ``cash`` and ``fwd`` (3-month) must be on a complete monthly index. Uses months <= t at signature and months
    <= t+h-1 for the indexed prices; cash(t+h) is only the realised cost."""
    avg = cash.rolling(n_avg).mean().shift(-(h - 1))  # mean of cash(t+h-n_avg ... t+h-1)
    out = pd.DataFrame({"cost": cash.shift(-h), "base_cash": cash, "base_3m": fwd}, index=cash.index)
    out["A_fixed_cash"] = cash
    out["B_fixed_3m"] = fwd
    out["C_index_cash_m1"] = cash.shift(-(h - 1))
    out["D_index_cash_avg3"] = avg
    out["E_fixed_3m_plus_trigger"] = trigger_price(fwd, avg / cash - 1, trigger)
    return out


def backtest(lme: pd.DataFrame, metals: list, horizons: list, start, trigger: float, n_avg: int) -> pd.DataFrame:
    """Long table: one row per metal × horizon × offer month × mechanism."""
    idx = pd.date_range(lme.index.min(), lme.index.max(), freq="MS")
    lme = lme.reindex(idx)
    last = lme.dropna(how="all").index.max()
    frames = []
    for metal in metals:
        cash, fwd = lme[f"{metal}_cash"], lme[f"{metal}_3m"]
        for h in horizons:
            p = mechanism_prices(cash, fwd, h, trigger, n_avg)
            p = p[(p.index >= pd.Timestamp(start)) & (p.index <= (last.to_period("M") - h).to_timestamp())]
            for m in MECHANISMS:
                base = p["base_3m"] if m in ("B_fixed_3m", "E_fixed_3m_plus_trigger") else p["base_cash"]
                f = pd.DataFrame({"metal": metal, "horizon": h, "offer_month": p.index,
                                  "delivery_month": p.index + pd.DateOffset(months=h), "mechanism": m,
                                  "cost": p["cost"].to_numpy(), "price": p[m].to_numpy(),
                                  "base_at_signature": base.to_numpy()})
                frames.append(f.dropna(subset=["cost", "price", "base_at_signature"]))
    off = pd.concat(frames, ignore_index=True)
    off["err"] = (off["cost"] - off["price"]) / off["cost"]
    off["client_change"] = off["price"] / off["base_at_signature"] - 1
    return off


def _stats(g: pd.DataFrame) -> dict:
    e, c = g["err"], g["client_change"]
    return {"n": int(len(g)), "mean_err_pct": 100 * e.mean(), "mae_pct": 100 * e.abs().mean(),
            "p90_err_pct": 100 * e.quantile(0.9), "max_err_pct": 100 * e.max(),
            "share_err_gt_5pct": float((e > 0.05).mean()), "client_change_std_pct": 100 * c.std(ddof=0),
            "client_change_p90_abs_pct": 100 * c.abs().quantile(0.9)}


def summarize(off: pd.DataFrame, periods: dict) -> pd.DataFrame:
    """Per metal × horizon × mechanism, for all offers ("all") and per sub-period of offer months."""
    rows = []
    sets = {"all": off}
    for name, (a, b) in periods.items():
        sets[name] = off[(off["offer_month"] >= pd.Timestamp(a)) & (off["offer_month"] <= pd.Timestamp(b))]
    for period, sub in sets.items():
        for (metal, h, m), g in sub.groupby(["metal", "horizon", "mechanism"], sort=False):
            rows.append({"metal": metal, "horizon": h, "mechanism": m, "period": period,
                         "first_offer": g["offer_month"].min().strftime("%Y-%m"),
                         "last_offer": g["offer_month"].max().strftime("%Y-%m"), **_stats(g)})
    return pd.DataFrame(rows)


def spread_stats(lme: pd.DataFrame, metals: list, until) -> pd.DataFrame:
    """(3m - cash) / cash per metal, months <= ``until`` (known at the origin)."""
    rows = []
    known = lme.loc[:pd.Timestamp(until)]
    for metal in metals:
        s = ((known[f"{metal}_3m"] - known[f"{metal}_cash"]) / known[f"{metal}_cash"]).dropna()
        rows.append({"metal": metal, "n_months": int(len(s)), "first_month": s.index.min().strftime("%Y-%m"),
                     "last_month": s.index.max().strftime("%Y-%m"), "mean_spread_pct": 100 * s.mean(),
                     "p10_spread_pct": 100 * s.quantile(0.1), "p90_spread_pct": 100 * s.quantile(0.9),
                     "share_contango": float((s > 0).mean()), "share_backwardation": float((s < 0).mean())})
    return pd.DataFrame(rows)


def eur_per_box(summary: pd.DataFrame, w: pd.DataFrame, prices: dict, material_share: float, products: list,
                metals: list) -> pd.DataFrame:
    """P90 err (all offers) × the metal's value per box, for both cost bases; margin pts = EUR / price × 100."""
    s = summary[summary["period"] == "all"]
    rows = []
    for prod in products:
        g = w[w["product"] == prod].set_index("material")
        total = float(g["value_eur"].sum())
        price = float(prices[prod])
        for metal in metals:
            if metal not in g.index:
                continue
            value = {"bom_value": float(g.at[metal, "value_eur"]),
                     "material_share": material_share * price * float(g.at[metal, "value_eur"]) / total}
            for r in s[s["metal"] == metal].itertuples():
                for basis, v in value.items():
                    eur = r.p90_err_pct / 100 * v
                    rows.append({"product": prod, "metal": metal, "horizon": r.horizon, "mechanism": r.mechanism,
                                 "cost_basis": basis, "metal_value_eur_per_box": v, "p90_err_pct": r.p90_err_pct,
                                 "p90_eur_per_box": eur, "p90_margin_pts": eur / price * 100})
    return pd.DataFrame(rows)


def headline(summary: pd.DataFrame, spreads: pd.DataFrame, metals: list, horizons: list) -> dict:
    s = summary[summary["period"] == "all"].set_index(["metal", "horizon", "mechanism"])["p90_err_pct"].sort_index()
    head = {}
    for metal in metals:
        head[f"{metal}_mean_3m_cash_spread_pct"] = round(float(
            spreads.set_index("metal").at[metal, "mean_spread_pct"]), 2)
        for h in horizons:
            x = s.loc[(metal, h)]
            u = x[x.index.isin(USABLE)]
            head[f"{metal}_h{h}_best_usable"] = f"{u.idxmin()} (P90 err {u.min():.1f}%)"
            head[f"{metal}_h{h}_best_overall"] = f"{x.idxmin()} (P90 err {x.min():.1f}%)"
            if h in (6, 9):
                head[f"{metal}_h{h}_P90_err_E_vs_B_pct"] = (f"{x['E_fixed_3m_plus_trigger']:.1f} vs "
                                                            f"{x['B_fixed_3m']:.1f}")
    return head


def _figure(summary: pd.DataFrame, horizons: list) -> None:
    s = summary[(summary["period"] == "all") & (summary["metal"] == "copper")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    width = 0.8 / len(MECHANISMS)
    xs = np.arange(len(horizons))
    for k, m in enumerate(MECHANISMS):
        g = s[s["mechanism"] == m].set_index("horizon").reindex(horizons)
        off = (k - (len(MECHANISMS) - 1) / 2) * width
        axes[0].bar(xs + off, g["p90_err_pct"], width, label=m)
        axes[1].bar(xs + off, g["client_change_std_pct"], width, label=m)
    axes[0].set_title("Copper: P90 margin loss (% of metal cost)")
    axes[0].set_ylabel("P90 of (cost - price) / cost (%)")
    axes[1].set_title("Copper: client-price variability")
    axes[1].set_ylabel("Std of price change vs signature (%)")
    for ax in axes:
        ax.set_xticks(xs, [f"{h} months" for h in horizons])
        ax.set_xlabel("Offer-to-delivery horizon")
        ax.axhline(0, color="grey", lw=0.8)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("LME reference for offers: margin protection vs commercial usability (offers 2017 onwards)")
    fig.tight_layout()
    fig.savefig(fig_dir() / "lme_reference.png", dpi=150)
    plt.close(fig)


def run(cfg: dict | None = None) -> dict:
    from src.testbench.targets import bom_weights, load_assumptions

    cfg = load_business_config() if cfg is None else cfg
    lr = cfg["clarifications"]["lme_reference"]
    a = load_assumptions()
    trig_cfg = float(a["quote_rules"]["indexation_trigger_pct"])
    assert float(lr["trigger_pct"]) == trig_cfg, (
        f"clarifications.lme_reference.trigger_pct ({lr['trigger_pct']}) != quote_rules.indexation_trigger_pct "
        f"({trig_cfg}) in config/assumptions.yaml")
    metals, horizons = lr["metals"], [int(h) for h in lr["horizons"]]

    lme = load_lme_monthly()
    off = backtest(lme, metals, horizons, lr["start"] + "-01", trig_cfg / 100, int(lr["lagged_avg_months"]))
    summary = summarize(off, lr["periods"])
    spreads = spread_stats(lme, metals, origin_ts(cfg))
    w = bom_weights(a)
    box = eur_per_box(summary, w, selling_prices(a, cfg), float(a["pricing"]["material_share_of_price"]["value"]),
                      cfg["products"], metals)

    out = out_dir()
    off.to_parquet(out / "lme_reference_offers.parquet")
    summary.to_csv(out / "lme_reference_summary.csv", index=False)
    box.to_csv(out / "lme_reference_eur_per_box.csv", index=False)
    spreads.to_csv(out / "lme_spread_stats.csv", index=False)
    _figure(summary, horizons)
    return headline(summary, spreads, metals, horizons)


if __name__ == "__main__":
    for k, v in run().items():
        print(f"{k}: {v}")
