"""Task 6 — margin at risk, base quote rules, competitor and customer levers, margin floor.

    uv run python -m src.business.pricing_levers

All rule values (margins, floors, triggers, bands, pass-through, tiers, deal sizes) are read from
config/assumptions.yaml. This module adjusts the price for a GIVEN deal; the pricing-options backtest
(src/proposals/part4_pricing_options.py, not imported here) picks which rule protects margin on past data.
"""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import BOM_IDS, PRODUCT_KEY, last_valid, load_business_config, origin_ts, selling_prices  # noqa: E402
from .paths import fig_dir, out_dir  # noqa: E402

SCENARIOS = ["p10", "p50", "p90"]
BASES = ["bom_value", "material_share"]
STRATEGIES = ["protect_margin", "gain_share"]
TIERS_GRID = ["strategic", "key", "standard"]
DEALS_GRID = [150_000, 600_000, 2_000_000]
POS_NOTE = "fixed price; two-way clause lets the client request a review beyond the trigger"


# ---------------------------------------------------------------------------------------------- config parsing
def _amount(tok: str) -> float:
    """'250k' -> 250000, '1M' -> 1000000."""
    m = re.fullmatch(r"\s*([\d.]+)\s*([kKmM]?)\s*", tok)
    if not m:
        raise ValueError(f"cannot read amount {tok!r}")
    mult = {"": 1, "k": 1e3, "m": 1e6}[m.group(2).lower()]
    return float(m.group(1)) * mult


def deal_bounds(deal_size_eur: dict) -> dict:
    """{'small': (0, 250e3), 'medium': (250e3, 1e6), 'large': (1e6, inf)} from the config range strings."""
    out = {}
    for name, spec in deal_size_eur.items():
        r = str(spec["range"]).strip()
        if r.startswith("<"):
            out[name] = (0.0, _amount(r[1:]))
        elif r.startswith(">"):
            out[name] = (_amount(r[1:]), float("inf"))
        else:
            lo, hi = re.split(r"\s*[-–]\s*", r)
            out[name] = (_amount(lo), _amount(hi))
    return out


def deal_bucket(value_eur: float, deal_size_eur: dict) -> str:
    for name, (lo, hi) in deal_bounds(deal_size_eur).items():
        if lo <= value_eur < hi:
            return name
    raise ValueError(f"no deal-size bucket for {value_eur}")


def giveup_pts(tier: str, deal_value_eur: float, customer_lever: dict) -> float:
    bucket = deal_bucket(deal_value_eur, customer_lever["deal_size_eur"])
    return float(customer_lever["customer_tiers"][tier]["max_margin_giveup_pts"]
                 + customer_lever["deal_size_eur"][bucket]["extra_giveup_pts"])


def band_thresholds(bands: dict) -> tuple[float, float]:
    """(hi, lo) in percentage points from '>= +2' / '<= -2'."""
    hi = float(re.search(r"[-+]?\d+(\.\d+)?", bands["competitors_hit_harder"]).group())
    lo = float(re.search(r"[-+]?\d+(\.\d+)?", bands["we_hit_harder"]).group())
    return hi, lo


def band_for(gap_pp: float, bands: dict) -> str:
    hi, lo = band_thresholds(bands)
    if gap_pp >= hi:
        return "competitors_hit_harder"
    if gap_pp <= lo:
        return "we_hit_harder"
    return "similar"


def pass_through_for(band: str, strategy: str, comp: dict) -> float:
    if band == "competitors_hit_harder" and strategy == "gain_share":
        return float(comp["gain_share_pass_through"])
    return float(comp["pass_through"][band])


# ---------------------------------------------------------------------------------------------- 6a / 6b
def margin_at_risk(ctx: dict, horizons: list) -> pd.DataFrame:
    rows = []
    for prod in ctx["price"]:
        price, margin, floor = ctx["price"][prod], ctx["margin"][prod], ctx["floor"][prod]
        for h in horizons:
            for q in SCENARIOS:
                chg = ctx["index"][(prod, h)][q] / 100 - 1
                r = {"product": prod, "horizon": h, "scenario": q, "cost_change": chg, "price_eur": price,
                     "margin": margin, "floor": floor}
                for b in BASES:
                    delta = ctx["cost_base"][(prod, b)] * chg
                    r[f"cost_base_{b}_eur"] = ctx["cost_base"][(prod, b)]
                    r[f"delta_eur_{b}"] = delta
                    r[f"margin_pts_lost_{b}"] = delta / price * 100
                    r[f"margin_after_{b}"] = margin - delta / price
                    r[f"below_floor_{b}"] = bool(margin - delta / price < floor)
                rows.append(r)
    return pd.DataFrame(rows)


def quote_rules_table(ctx: dict, horizons: list, quote_rules: dict, basis: str) -> pd.DataFrame:
    v = quote_rules["validity"]
    thr = float(v["band_threshold_pct"]) / 100
    trig = float(quote_rules["indexation_trigger_pct"])
    rows = []
    for prod in ctx["price"]:
        cb, price = ctx["cost_base"][(prod, basis)], ctx["price"][prod]
        for h in horizons:
            ix = ctx["index"][(prod, h)]
            d50, d90 = cb * (ix["p50"] / 100 - 1), cb * (ix["p90"] / 100 - 1)
            bw = (ix["p90"] - ix["p10"]) / ix["p50"]
            rows.append({"product": prod, "horizon": h, "cost_basis": basis, "price_adjustment_pct": d50 / price,
                         "p50_delta_eur": d50, "p90_delta_eur": d90, "buffer_eur": d90 - d50, "band_width_pct": bw,
                         "validity_days": int(v["narrow_band_days"] if bw < thr else v["wide_band_days"]),
                         "indexation_recommended": bool(bw >= thr or abs(ix["p90"] - 100) >= trig),
                         "trigger_pct": trig})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- 6c
def competitor_value_shares(td: pd.DataFrame, eur_per_kg: dict, cfg: dict) -> pd.DataFrame:
    groups = list(cfg["competitor_lever"]["priced_groups"])
    if cfg["competitor_lever"]["exclude_brands_if_broken"]:
        td = td[~td["broken"]]
    rows = []
    for brand, g in td.groupby("brand"):
        mass = g.groupby("group")["mass_kg"].sum()
        total = float(g["total_kg"].iloc[0])
        val = {grp: float(mass.get(grp, 0.0)) * eur_per_kg[grp] for grp in groups}
        tot_val = sum(val.values())
        r = {"brand": brand, "total_kg": total,
             "excluded_mass_share": 1 - sum(float(mass.get(grp, 0.0)) for grp in groups) / total}
        for grp in groups:
            r[f"mass_kg_{grp}"] = float(mass.get(grp, 0.0))
            r[f"value_share_{grp}"] = val[grp] / tot_val
        rows.append(r)
    return pd.DataFrame(rows)


def group_changes(fc: pd.DataFrame, cfg: dict, horizon: int, q: str = "p50") -> dict:
    from .forecasts import pct_change

    mix = cfg["competitor_lever"]["brass_mix"]
    out = {}
    for grp, ser in cfg["competitor_lever"]["priced_groups"].items():
        if grp == "brass":
            cu, zn = ser
            out[grp] = mix["copper"] * pct_change(fc, cu, horizon, q) + mix["zinc"] * pct_change(fc, zn, horizon, q)
        else:
            out[grp] = pct_change(fc, ser, horizon, q)
    return out


def realised_group_changes(panel: pd.DataFrame, cfg: dict, end: pd.Timestamp, months: int = 12) -> dict:
    """Realised price change of each priced group over the ``months`` to ``end`` (last values <= each date)."""
    start = (end.to_period("M") - months).to_timestamp()
    mix = cfg["competitor_lever"]["brass_mix"]

    def chg(s):
        return last_valid(panel[s], end) / last_valid(panel[s], start) - 1

    out = {}
    for grp, ser in cfg["competitor_lever"]["priced_groups"].items():
        if grp == "brass":
            cu, zn = ser
            out[grp] = mix["copper"] * chg(cu) + mix["zinc"] * chg(zn)
        else:
            out[grp] = chg(ser)
    return out


def _brand_changes(shares: pd.DataFrame, groups: list, chg: dict) -> dict:
    return {r.brand: sum(getattr(r, f"value_share_{g}") * chg[g] for g in groups) for r in shares.itertuples()}


def competitor_lever(shares: pd.DataFrame, fc: pd.DataFrame, cfg: dict, comp_cfg: dict, horizons: list,
                     realised: dict | None = None) -> pd.DataFrame:
    """Teardown-weighted cost change per brand and its gap vs the Schneider teardown, for p10 / p50 / p90.

    Band and pass-through come from the gap of ``competitor_lever.band_scenario`` (default p90). ``realised`` =
    realised group price changes over the last 12 months (backward-looking check, same weights)."""
    groups = list(cfg["competitor_lever"]["priced_groups"])
    band_q = cfg["competitor_lever"].get("band_scenario", "p90")
    competitors = [b for b in comp_cfg["competitors"] if b in set(shares["brand"])]
    real = _brand_changes(shares, groups, realised) if realised is not None else None
    rows = []
    for h in horizons:
        by_q = {q: _brand_changes(shares, groups, group_changes(fc, cfg, h, q)) for q in SCENARIOS}
        off = fc.loc[(fc["series"] == BOM_IDS["RM6"]) & (fc["horizon"] == h), "p50"]
        official = float(off.iloc[0] / 100 - 1) if len(off) else np.nan
        for b in competitors + ["competitor_average"]:
            r = {"horizon": h, "brand": b, "band_scenario": band_q}
            for q in SCENARIOS:
                bc = by_q[q]
                c = np.mean([bc[x] for x in competitors]) if b == "competitor_average" else bc[b]
                r[f"cost_change_{q}"] = c
                r[f"schneider_teardown_change_{q}"] = bc["Schneider"]
                r[f"gap_pp_{q}"] = (c - bc["Schneider"]) * 100
            band = band_for(r[f"gap_pp_{band_q}"], comp_cfg["bands_pct_points"])
            r.update({"band": band, "pass_through_protect_margin": pass_through_for(band, "protect_margin", comp_cfg),
                      "pass_through_gain_share": pass_through_for(band, "gain_share", comp_cfg)})
            if real is not None:
                rc = np.mean([real[x] for x in competitors]) if b == "competitor_average" else real[b]
                r["cost_change_realised_12m"] = rc
                r["gap_realised_12m_pp"] = (rc - real["Schneider"]) * 100
            r["schneider_official_bom_change_p50_info"] = official
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- context
def build_context(cfg: dict | None = None) -> dict:
    """Everything ``recommend`` needs, from the real config, forecasts and BOM (computed once per run)."""
    from src import data as d
    from src.testbench.run import build_panel
    from src.testbench.targets import bom_weights, load_assumptions

    from .forecasts import get_origin_forecasts

    cfg = load_business_config() if cfg is None else cfg
    a = load_assumptions()
    prices = selling_prices(a, cfg)
    fc = get_origin_forecasts(cfg)
    w = bom_weights(a)
    p = a["pricing"]
    share = float(p["material_share_of_price"]["value"])
    products = cfg["products"]
    ctx = {"price": {pr: prices[pr] for pr in products},
           "margin": {pr: float(p[f"profit_margin_{PRODUCT_KEY[pr]}"]["value"]) for pr in products},
           "floor": {pr: float(p["margin_floor"][PRODUCT_KEY[pr]]["value"]) for pr in products},
           "cost_base": {}, "index": {}, "competitor": {}, "quote": {},
           "customer": a["quote_rules"]["customer_lever"], "default_basis": cfg["pricing"]["cost_basis_default"],
           "default_strategy": a["quote_rules"]["competitor_lever"]["strategy_when_competitors_hit_harder"],
           "airset_price_assumed": prices["airset_price_assumed"]}
    for pr in products:
        ctx["cost_base"][(pr, "bom_value")] = float(w.loc[w["product"] == pr, "value_eur"].sum())
        ctx["cost_base"][(pr, "material_share")] = share * prices[pr]
        for h in cfg["horizons"]:
            row = fc[(fc["series"] == BOM_IDS[pr]) & (fc["horizon"] == h)].iloc[0]
            ctx["index"][(pr, h)] = {q: float(row[q]) for q in SCENARIOS}

    # competitor lever (RM6): EUR/kg at the last value <= origin
    panel = build_panel()
    base = origin_ts(cfg)
    cl = cfg["competitor_lever"]
    lv = {s: last_valid(panel[s], base) for s in ["D01", "D02", "L01", "EP01", "S05", "S06"]}
    eur_per_kg = {}
    for grp, ser in cl["priced_groups"].items():
        if grp == "brass":
            eur_per_kg[grp] = (cl["brass_mix"]["copper"] * lv["D01"] + cl["brass_mix"]["zinc"] * lv["S05"] / lv["S06"]) / 1000
        elif grp == "plastics":
            eur_per_kg[grp] = float(cl["plastics_eur_per_kg"])
        else:
            eur_per_kg[grp] = lv[ser] / 1000
    shares = competitor_value_shares(d.load_competitor_teardowns(), eur_per_kg, cfg)
    realised = realised_group_changes(panel, cfg, base)
    comp = competitor_lever(shares, fc, cfg, a["quote_rules"]["competitor_lever"], cfg["horizons"], realised)
    band_q = cfg["competitor_lever"].get("band_scenario", "p90")
    ctx["band_scenario"] = band_q
    for h, g in comp[comp["brand"] == "competitor_average"].groupby("horizon"):
        r = g.iloc[0]
        ctx["competitor"][int(h)] = {"band": r["band"], "gap_pp": float(r[f"gap_pp_{band_q}"]),
                                     "gap_realised_12m_pp": float(r["gap_realised_12m_pp"]),
                                     "pass_through": {"protect_margin": float(r["pass_through_protect_margin"]),
                                                      "gain_share": float(r["pass_through_gain_share"])}}
    qr = quote_rules_table(ctx, cfg["horizons"], a["quote_rules"], ctx["default_basis"])
    for r in qr.itertuples():
        ctx["quote"][(r.product, int(r.horizon))] = {"indexation_recommended": bool(r.indexation_recommended),
                                                    "validity_days": int(r.validity_days),
                                                    "buffer_eur": float(r.buffer_eur)}
    ctx["_tables"] = {"shares": shares, "competitor": comp, "quote_rules": qr, "assumptions": a}
    return ctx


_CTX: dict | None = None


def _context() -> dict:
    global _CTX
    if _CTX is None:
        _CTX = build_context()
    return _CTX


# ---------------------------------------------------------------------------------------------- 6e
def recommend(product, horizon, scenario="p50", tier="standard", deal_value_eur=200_000, strategy=None,
              cost_basis=None, ctx: dict | None = None) -> dict:
    """Price recommendation for one deal, applying the levers in ``combination_order``."""
    ctx = _context() if ctx is None else ctx
    strategy = ctx["default_strategy"] if strategy is None else strategy
    basis = ctx["default_basis"] if cost_basis is None else cost_basis
    price, margin, floor = ctx["price"][product], ctx["margin"][product], ctx["floor"][product]
    cl = ctx["customer"]
    bucket = deal_bucket(deal_value_eur, cl["deal_size_eur"])
    gpts = giveup_pts(tier, deal_value_eur, cl)
    delta = ctx["cost_base"][(product, basis)] * (ctx["index"][(product, horizon)][scenario] / 100 - 1)
    notes = []
    pt = ctx["competitor"][horizon]["pass_through"][strategy] if product == "RM6" else 1.0
    absorbed, floor_binding = 0.0, False
    if delta <= 0:
        increase = 0.0
        notes.append(POS_NOTE)
    else:
        base_increase = pt * delta
        absorbed = min(base_increase, gpts / 100 * price)
        increase = base_increase - absorbed
    margin_after = margin - (delta - increase) / price
    if margin_after < floor:
        increase = delta - (margin - floor) * price
        floor_binding = True
        margin_after = floor
        notes.append("margin floor binding: price raised to keep the floor")
    q = ctx["quote"][(product, horizon)]
    flag = bool(cl["large_or_long_deals_require_indexation"])
    indexation = bool(q["indexation_recommended"] or (bucket == "large" and flag)
                      or (horizon == 9 and flag and bucket in ("medium", "large")))
    return {"product": product, "horizon": horizon, "scenario": scenario, "tier": tier, "deal_bucket": bucket,
            "deal_value_eur": deal_value_eur, "strategy": strategy, "cost_basis": basis, "delta_eur": delta,
            "pass_through": pt, "giveup_pts": gpts, "absorbed_eur": absorbed, "increase_eur": increase,
            "increase_pct_of_price": increase / price, "margin_after": margin_after, "floor_binding": floor_binding,
            "indexation": indexation, "validity_days": q["validity_days"], "buffer_eur": q["buffer_eur"],
            "notes": "; ".join(notes)}


def run(cfg: dict | None = None) -> dict:
    global _CTX
    cfg = load_business_config() if cfg is None else cfg
    ctx = build_context(cfg)
    _CTX = ctx
    t = ctx["_tables"]
    out = out_dir()
    mar = margin_at_risk(ctx, cfg["horizons"])
    mar.to_csv(out / "margin_at_risk.csv", index=False)
    t["quote_rules"].to_csv(out / "quote_rules.csv", index=False)
    t["competitor"].to_csv(out / "competitor_lever.csv", index=False)
    t["shares"].to_csv(out / "competitor_value_shares.csv", index=False)

    grid = [recommend(p, h, q, tier, deal, strat, cost_basis=basis, ctx=ctx)
            for basis in BASES for p in cfg["products"] for h in cfg["horizons"] for q in SCENARIOS
            for tier in TIERS_GRID for deal in DEALS_GRID for strat in STRATEGIES]
    rec = pd.DataFrame(grid)
    rec.to_csv(out / "pricing_recommendations.csv", index=False)

    sel = rec[(rec["product"] == "RM6") & (rec["horizon"] == 6) & (rec["scenario"] == "p90")
              & (rec["strategy"] == ctx["default_strategy"]) & (rec["cost_basis"] == ctx["default_basis"])]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    vals_all = [100 * v for v in sel["increase_pct_of_price"]]
    ymax = max(vals_all + [0.1])
    for k, tier in enumerate(TIERS_GRID):
        s_t = sel[sel["tier"] == tier].set_index("deal_value_eur")["increase_pct_of_price"]
        xs = np.arange(len(DEALS_GRID)) + (k - 1) * width
        vals = [100 * s_t[v] for v in DEALS_GRID]
        ax.bar(xs, vals, width, label=tier)
        for x, v in zip(xs, vals):
            if abs(v) < 1e-9:
                ax.text(x, ymax * 0.02, "0% (absorbed by give-up)", rotation=90, ha="center", va="bottom",
                        fontsize=7, color="dimgrey")
    ax.set_ylim(0, ymax * 1.15)
    ax.set_xticks(range(len(DEALS_GRID)), [f"{v / 1e3:,.0f}k EUR" if v < 1e6 else f"{v / 1e6:,.0f}M EUR"
                                           for v in DEALS_GRID])
    ax.set_xlabel("Deal value")
    ax.set_ylabel("Recommended price increase (% of price)")
    ax.set_title(f"RM6, 6 months, P90 cost scenario: price increase by customer tier\n"
                 f"({ctx['default_strategy']}, cost basis {ctx['default_basis']}, "
                 f"competitor band from {ctx['band_scenario']})")
    ax.legend(title="Customer tier")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir() / "pricing_levers_example.png", dpi=150)
    plt.close(fig)

    m9 = mar[(mar["horizon"] == 9) & (mar["scenario"] == "p90")].set_index("product")
    bands = {h: ctx["competitor"][h]["band"] for h in cfg["horizons"]}
    head = {}
    for b in BASES:
        for prod in cfg["products"]:
            head[f"{prod}_p90_margin_pts_lost_h9_{b}"] = round(float(m9.at[prod, f"margin_pts_lost_{b}"]), 2)
    head.update({
        "competitor_band_scenario": ctx["band_scenario"],
        "RM6_competitor_band": ", ".join(f"h{h}: {b}" for h, b in bands.items()),
        f"RM6_competitor_gap_{ctx['band_scenario']}_pp": ", ".join(
            f"h{h}: {ctx['competitor'][h]['gap_pp']:+.2f}" for h in cfg["horizons"]),
        "RM6_competitor_gap_realised_12m_pp": round(ctx["competitor"][cfg["horizons"][0]]["gap_realised_12m_pp"], 2),
        "grid_rows": int(len(rec))})
    for b in BASES:
        head[f"grid_rows_floor_binding_{b}"] = int(rec.loc[rec["cost_basis"] == b, "floor_binding"].sum())
    head["airset_price_assumed"] = bool(ctx["airset_price_assumed"])
    return head


if __name__ == "__main__":
    print(run())
