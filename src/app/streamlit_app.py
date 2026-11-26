"""Quote risk simulator: material and energy shocks -> BOM cost -> margin by quote rule. LOCAL ONLY (licensed data).

    make app        (uv run --group app streamlit run src/app/streamlit_app.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # streamlit runs the file as a script

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.app import goal_b as gb
from src.app import inputs as I
from src.app import scenarios as sc
from src.business.margin_replay import Economics
from src.testbench.targets import TARGET_IDS

st.set_page_config(page_title="Quote risk simulator", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")

# Schneider Electric greens + neutrals. Chart pair validated for colour-blind readers (dataviz validator).
GREEN, LIFE_GREEN, BLUE, RED, INK, MUTED, GRID = "#009530", "#3DCD58", "#42B4E6", "#DC0A0A", "#333333", "#626469", "#E6E8E6"
ORANGE = "#E47F00"
PRODUCT_LABEL = {"RM6": "RM6", "AirSeT": "RM AirSeT"}
SCENARIOS = {  # label -> shock definition; "plant" = the product's plant power price (local config)
    "No shock": dict(kind="sigma", k=0.0),
    "Normal noise": dict(kind="sigma", k=1.0),
    "Rare event": dict(kind="sigma", k=2.0),
    "Worst in history": dict(kind="worst", k=0.0),
    "Extreme stress": dict(kind="sigma", k=10.0),
    "Plant power spike": dict(kind="worst", k=0.0, driver="plant"),
}
SCENARIO_HELP = {
    "No shock": "Our forecast as it is.",
    "Normal noise": "A typical move (1 sigma) over the offer.",
    "Rare event": "A 1-in-20 move (2 sigma).",
    "Worst in history": "The largest move ever seen over the same number of months.",
    "Extreme stress": "10 sigma, far beyond anything in history. A stress test only.",
    "Plant power spike": "The worst power-price move at the plant, passed into materials.",
}
RULES = {"spot": "Fixed price", "p50_model": "Fixed, on our median", "p90_model": "Fixed + forecast buffer",
         "p90_hist": "Fixed + historical buffer", "clause_3": "Index clause ±3%", "clause_5": "Index clause ±5%",
         "clause_10": "Index clause ±10%", "hedge_cu": "Copper bought forward",
         "hedge_cu_clause_5": "Copper forward + clause ±5%"}
POINTS = ("favourable", "central", "adverse")
DEALS = {150_000: "under 250k EUR", 600_000: "250k-1M EUR", 2_000_000: "over 1M EUR"}
TOUR = [
    ("Welcome", "This tool shows what happens to the **margin of a fixed-price offer** if raw-material prices move "
                "between the quote and the purchase. It takes 30 seconds to learn."),
    ("1 · Choose the offer", "Pick the **product** and **how many months** the price stays fixed."),
    ("2 · Choose a scenario", "Pick what might happen to prices, from **no shock** to a **worst-in-history** move."),
    ("3 · Read the decision", "Four cards answer the pricing questions: **how to quote**, the **longest safe fixed "
                              "price**, **how much to raise** the price, and whether to **reopen** it."),
    ("Go further", "The tabs hold the evidence behind each decision, the **AirSeT** selling guide and "
                   "**sustainability**. **Settings** (top-left arrow) hold the assumptions: margins, confidence level, "
                   "which price moves."),
]


@st.cache_resource(show_spinner="Loading forecasts and price history...")
def load(tag: str) -> I.Inputs:
    return I.load(tag)


def pct(x: float, signed: bool = True) -> str:
    return f"{x:+.1%}" if signed else f"{x:.1%}"


def close_tour():
    st.session_state.tour_open = False


@st.dialog("Quick tour", on_dismiss=close_tour)
def tour():
    i = st.session_state.get("tour_i", 0)
    title, body = TOUR[i]
    st.markdown(f"#### {title}")
    st.markdown(body)
    st.progress((i + 1) / len(TOUR), text=f"Step {i + 1} of {len(TOUR)}")
    back, nxt = st.columns(2)
    if i > 0 and back.button("Back", width="stretch"):
        st.session_state.tour_i = i - 1
        st.rerun()
    last = i == len(TOUR) - 1
    if nxt.button("Start" if last else "Next", type="primary", width="stretch"):
        st.session_state.tour_i = 0 if last else i + 1
        st.session_state.tour_open = not last
        st.rerun()


st.session_state.setdefault("tour_open", st.query_params.get("tour") != "0")  # ?tour=0 skips it (demos)
if st.session_state.tour_open:
    tour()

# ------------------------------------------------------------------------------------------------ settings (sidebar)
sb = st.sidebar
sb.header("Settings")
sb.caption("Runs on this computer only. Licensed data: never deploy it on a server.")
tag = sb.selectbox("Forecast", I.sources(), format_func=lambda t: "Compute here (first run ~20 s)" if t == I.FRESH
                   else f"Graded run {t}", help="Graded runs come from the frozen backtest, newest first.")
inp = load(tag)

# ------------------------------------------------------------------------------------------------ header + steps 1, 2
head, tour_btn = st.columns([6, 1], vertical_alignment="center")
head.markdown(f"<h2 style='margin:0'><span style='color:{LIFE_GREEN}'>●</span> Quote risk simulator</h2>"
              f"<span style='color:{MUTED}'>Set the offer and a price scenario. Get the pricing decision that protects "
              "the margin.</span>", unsafe_allow_html=True)
if tour_btn.button("Quick tour", icon=":material/play_circle:", width="stretch"):
    st.session_state.tour_open, st.session_state.tour_i = True, 0
    st.rerun()

c1, c2, c3 = st.columns([1.1, 1.1, 3])
product = c1.segmented_control("1 · Product", list(TARGET_IDS), default="RM6", format_func=PRODUCT_LABEL.get,
                               required=True)
h = c2.segmented_control("Price fixed for", list(I.HORIZONS), default=6, format_func=lambda m: f"{m} months",
                         required=True)
scenario = c3.pills("2 · Scenario", list(SCENARIOS), default="Normal noise", required=True,
                    help="\n".join(f"- **{k}**: {v}" for k, v in SCENARIO_HELP.items()))
st.caption(SCENARIO_HELP[scenario])

w = inp.weights[inp.weights["product"] == product].set_index("series")
names = {s: m.replace("_", " ").capitalize() for s, m in w["material"].items()} | I.ENERGY
plant = I.plant_power(inp.assumptions, product)  # from the local config (NDA), may be None

# advanced settings
sb.subheader("Scenario")
p = dict(SCENARIOS[scenario])
if p.get("driver") == "plant" and plant not in inp.panel:
    sb.warning("Set plant_power_series in config/assumptions.yaml to use the plant power scenario.")
    p = dict(kind="sigma", k=0.0)
up = sb.segmented_control("Direction", ["Prices up", "Prices down"], default="Prices up", required=True) == "Prices up"
drivers = [sc.MARKET] + list(w.index) + [s for s in I.ENERGY if s in inp.panel]
label = {sc.MARKET: "Everything at once"} | names
default_driver = plant if p.get("driver") == "plant" else sc.MARKET
driver = sb.selectbox("What moves", drivers, index=drivers.index(default_driver), format_func=label.get,
                      help="'Everything at once' moves every price by the same number of sigmas (cautious).")
follow = True
if driver != sc.MARKET:
    follow = sb.toggle("Other prices follow (historical co-movement)", value=True)
if sb.toggle("Custom shock size", value=False):
    p = dict(kind="sigma", k=sb.slider("Shock size (sigma)", 0.0, 10.0, 2.0, 0.5))
level = sb.select_slider("Confidence of the range", [0.50, 0.80, 0.90, 0.98], value=0.80,
                         format_func=lambda v: f"{v:.0%}", help="80% = 1-in-10 chance of being above the high end.")
shock = sc.Shock(driver=driver, kind=p["kind"], k=p["k"], up=up)

sb.subheader("Economics (assumptions)")
eco0 = I.default_economics(inp, product)
share = sb.slider("Materials, share of price", 0.10, 0.80, float(eco0.material_share), 0.01)
margin0 = sb.slider("Target margin", 0.0, 0.50, float(eco0.target_margin), 0.01)
floor = sb.slider("Margin floor", 0.0, 0.40, float(eco0.floor), 0.01)
e_share = sb.slider("Plant energy, share of price", 0.0, 0.10, 0.0, 0.005, format="%.3f",
                    help="Assumption, Schneider data not available. 0 by default.")
elec_levels = [0.0] + [float(v) for v in inp.bcfg["electronics_slider"]["eur_per_box"][product]]
elec = sb.select_slider("Electronics per unit (not in Schneider's BOM)", elec_levels, value=0.0,
                        format_func=lambda v: "None, as in our submitted index" if v == 0 else f"{v:,.0f} EUR",
                        help="Sensors and boards are not in the official BOM. They move with the electronics price "
                             "index (EL01). Range: our assumption, to confirm with Schneider.")
show_eur = sb.toggle("Show EUR levels (local screen only)", value=False)
eco = Economics(material_share=share, target_margin=margin0, floor=floor)

# ------------------------------------------------------------------------------------------------ computation
fc = inp.fc.set_index(["series", "horizon"])
w_all = inp.weights.set_index("series")
COMP_SERIES = ["D01", "D02", "L01", "EP01", "S11", "S05", "EL01"]  # competitor groups (incl. zinc for brass), electronics
names = {"L01": "Carbon steel (competitor teardowns)", "S05": "Zinc", "EL01": "Electronics"} | names
series = list(dict.fromkeys(list(w_all.index) + [s for s in I.ENERGY if s in inp.panel and (s, h) in fc.index]
                            + [s for s in COMP_SERIES if (s, h) in fc.index]))
moves = {hh: sc.scenario_moves(inp.panel, series, hh, I.ORIGIN, shock, follow, set(I.ENERGY)) for hh in I.HORIZONS}


def band(s: str, hh: int, shocked: bool = True):
    r = fc.loc[(s, hh)]
    lo, hi = sc.scale_band(r.p10, r.p50, r.p90, level)
    f = float(np.exp(moves[hh].at[s, "move"])) if shocked and s in moves[hh].index else 1.0
    return lo / r["last"] * f, r.p50 / r["last"] * f, hi / r["last"] * f


bom_id = TARGET_IDS[product]
V = float(w["value"].sum())  # priced BOM materials, EUR per unit at the origin


def bom_for(prod: str, hh: int, shocked: bool = True):
    """(low, expected, high) raw-material cost ratio of a product vs September 2026."""
    wp = inp.weights[inp.weights["product"] == prod].set_index("series")
    comps = pd.DataFrame({"weight": wp["weight"],
                          "ratio50": [fc.loc[(s, hh)].p50 / fc.loc[(s, hh)]["last"] for s in wp.index]}, index=wp.index)
    q = fc.loc[(TARGET_IDS[prod], hh)]
    mv = moves[hh]["move"] if shocked else pd.Series(0.0, index=comps.index)
    return sc.bom_band((q.p10, q.p50, q.p90), comps, mv, level)


def bom(hh: int, shocked: bool = True):
    """Selected product, with the electronics slider: (V x BOM ratio + E x electronics ratio) / (V + E)."""
    b = bom_for(product, hh, shocked)
    if elec == 0 or ("EL01", hh) not in fc.index:
        return b
    e = band("EL01", hh, shocked)
    return tuple((V * x + elec * y) / (V + elec) for x, y in zip(b, e))


base, scen = bom(h, shocked=False), bom(h)


def margins(hh: int) -> pd.DataFrame:
    """Margin of every quote rule at the three points (favourable, central, adverse), offer fixed for hh months."""
    s_h = bom(hh)
    cu_h = band("D01", hh) if "D01" in w.index else (1.0, 1.0, 1.0)
    en_h = band(plant, hh) if plant and (plant, hh) in fc.index else (1.0, 1.0, 1.0)
    q = fc.loc[(bom_id, hh)]
    return sc.margin_table(eco, hh, dict(zip(POINTS, zip(s_h, cu_h))),
                           {"p50": q.p50 / 100, "p90": q.p90 / 100, "hist_p90": inp.hist_p90.get((product, hh), np.nan)},
                           w_cu=float(w.loc["D01", "weight"]) * V / (V + elec) if "D01" in w.index else 0.0,
                           cu_fwd_ratio=float(np.exp(inp.cu_term * hh / 3)), energy_share=e_share,
                           energy_ratio=dict(zip(POINTS, en_h)))


def spot_bad(t: pd.DataFrame) -> float:
    return float(t[(t.policy == "spot") & (t.point == "adverse")]["margin"].iloc[0])


tab = margins(h)
mt = tab.pivot_table(index="policy", columns="point", values="margin").reindex(list(RULES))
rec = I.recommended_policy(inp, product, share, h)
rec_name = RULES.get(rec[0], rec[0]) if rec else None
spot = mt.loc["spot"]
safe_fixed = 0  # longest offer (3, 6, 9 months) whose fixed price keeps the bad case above the floor
for hh in I.HORIZONS:
    if spot_bad(tab if hh == h else margins(hh)) < floor:
        break
    safe_fixed = hh
# cheapest safe rule: the lowest price markup that keeps the bad case above the floor, as in our backtest rule
mk = tab[tab.point == "central"].set_index("policy")["markup"].reindex(mt.index)
safe = mt.index[mt["adverse"] >= floor]
FRICTION = ["spot", "p50_model", "hedge_cu", "p90_hist", "p90_model", "clause_10", "hedge_cu_clause_5", "clause_5",
            "clause_3"]  # ties on markup: fewer price revisions first (backtest tie-break)
order = pd.DataFrame({"mk": mk.round(4), "fr": [FRICTION.index(r) for r in mk.index]}, index=mk.index)
best = order.loc[safe].sort_values(["mk", "fr"]).index[0] if len(safe) else mt["adverse"].idxmax()

# price advice (Goal A levers): competitors first (RM6 only), then the client, never below the floor
from src.business import pricing_levers as pl  # noqa: E402

q_rules = inp.assumptions["quote_rules"]
comp_cfg, cust_cfg = q_rules["competitor_lever"], q_rules["customer_lever"]
trig = float(q_rules.get("indexation_trigger_pct", 5)) / 100
price = float(inp.prices[product])
cost_base = V + elec


def advice(tier: str, deal: float, strategy: str) -> dict:
    out = {"pt": 1.0, "cg": None, "gpts": pl.giveup_pts(tier, deal, cust_cfg),
           "verdict": "The competitor check applies to RM6 only: AirSeT has no comparable teardowns."}
    if product == "RM6":
        grp_cfg = inp.bcfg["competitor_lever"]
        hi = {s: band(s, h)[2] - 1 for s in ("D01", "D02", "L01", "EP01", "S11", "S05")}
        mix = grp_cfg["brass_mix"]
        chg = {"copper": hi["D01"], "carbon_steel": hi["L01"], "aluminium": hi["D02"], "epoxy": hi["EP01"],
               "plastics": hi["S11"], "brass": mix["copper"] * hi["D01"] + mix["zinc"] * hi["S05"]}
        gaps, ours = sc.competitor_gaps(inp.comp_shares, {g: chg[g] for g in grp_cfg["priced_groups"]},
                                        comp_cfg["competitors"])
        avg_gap = float(gaps.set_index("brand").at["Competitor average", "gap_pp"])
        bnd = pl.band_for(avg_gap, comp_cfg["bands_pct_points"])
        pt = pl.pass_through_for(bnd, strategy, comp_cfg)
        out["pt"] = pt
        out["verdict"] = {
            "competitors_hit_harder": f"Competitors' costs rise {avg_gap:+.1f} points more than ours, so we can pass "
                                      f"on {pt:.0%} of our increase and stay competitive.",
            "similar": f"Competitors' costs move like ours ({avg_gap:+.1f} points), so we pass on {pt:.0%}.",
            "we_hit_harder": f"Our costs rise {-avg_gap:.1f} points more than competitors', so we pass on only "
                             f"{pt:.0%} to stay competitive."}[bnd]
        cg = gaps.assign(Brand=gaps.brand.where(gaps.brand != "Competitor average", "Competitor avg."))
        out["cg"] = pd.concat([cg, pd.DataFrame({"Brand": ["Schneider (teardown)"], "change": [ours]})],
                              ignore_index=True)
    out["res"] = {lab: sc.apply_levers(cost_base * (ratio - 1), price, margin0, floor, out["pt"], out["gpts"])
                  for lab, ratio in (("Expected", scen[1]), ("Bad case", scen[2]))}
    return out


ADV_DEFAULT = {"adv_tier": "standard", "adv_deal": 150_000,
               "adv_strategy": comp_cfg["strategy_when_competitors_hit_harder"]}
adv_now = {k: st.session_state.get(k) or v for k, v in ADV_DEFAULT.items()}  # same values as the Price increase tab
adv = advice(*adv_now.values())
bad_adv = adv["res"]["Bad case"]


# ------------------------------------------------------------------------------------------------ step 3: decision
def card(col, label: str, value: str, note: str, tone: str = "ok"):
    edge = {"ok": GREEN, "warn": ORANGE, "bad": RED}[tone]
    col.markdown(f"<div style='border:1px solid {GRID};border-left:6px solid {edge};border-radius:8px;"
                 f"padding:10px 14px;min-height:104px'><div style='color:{MUTED};font-size:.85rem'>{label}</div>"
                 f"<div style='color:{INK};font-size:1.3rem;font-weight:700;line-height:1.35'>{value}</div>"
                 f"<div style='color:{MUTED};font-size:.85rem'>{note}</div></div>", unsafe_allow_html=True)


st.markdown("##### 3 · Decision for this offer")
k1, k2, k3, k4 = st.columns(4)
card(k1, "How to quote", RULES[best] + ("  ★" if rec and best == rec[0] else ""),
     f"bad-case margin {pct(mt.loc[best, 'adverse'], False)}, floor {floor:.0%}",
     "ok" if spot["adverse"] >= floor else "warn" if len(safe) else "bad")
card(k2, "Longest safe fixed price", f"{safe_fixed} months" if safe_fixed else "Under 3 months",
     "bad case stays above the floor", "ok" if safe_fixed >= h else "warn" if safe_fixed else "bad")
card(k3, "Raise the price by", pct(bad_adv["increase_eur"] / price),
     f"bad case; expected case {pct(adv['res']['Expected']['increase_eur'] / price)}",
     "warn" if bad_adv["floor_binding"] else "ok")
review = abs(scen[2] - 1) > trig
card(k4, "Reopen the price", "Likely before delivery" if review else "Not expected",
     f"bad-case cost {pct(scen[2] - 1)}, trigger ±{trig:.0%}", "warn" if review else "ok")

if spot["adverse"] >= floor:
    st.success(f"**A fixed price holds.** Even in the bad case the margin stays at {pct(spot['adverse'], False)}, "
               f"above the {floor:.0%} floor.", icon=":material/check_circle:")
elif len(safe):
    st.warning(f"**A fixed price is risky here.** In the bad case the margin falls to {pct(spot['adverse'], False)}. "
               f"Quote with **{RULES[best]}**, or fix the price for {safe_fixed} months at most."
               if safe_fixed else f"**A fixed price is risky here.** In the bad case the margin falls to "
               f"{pct(spot['adverse'], False)}. Quote with **{RULES[best]}**.", icon=":material/warning:")
else:
    st.error(f"**No quote rule keeps the margin above the {floor:.0%} floor in this scenario.** "
             f"**{RULES[best]}** loses the least. Re-quote or shorten the offer.", icon=":material/error:")
if "Extreme" in scenario:
    largest = max(abs(sc.worst_move(inp.panel[s], h, I.ORIGIN, up)) / sc.sigma(inp.panel[s], h, I.ORIGIN)
                  for s in w.index)
    st.info(f"Stress test only. The largest {h}-month move in history was {largest:.1f} sigma, not 10.",
            icon=":material/science:")
st.caption((f"★ Default rule from our 2010-21 backtest for this offer: {rec_name}. " if rec_name else "")
           + f"Price increase for a {adv_now['adv_tier']} client and an order of {DEALS[adv_now['adv_deal']]} "
             "(set them in the Price increase tab).")
if e_share == 0 and scenario == "Plant power spike":
    st.caption("The spike reaches the margin through material prices only. Add the plant energy share in Settings "
               "to include the direct cost.")


# ------------------------------------------------------------------------------------------------ charts
def style(chart):
    return (chart.configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID, domainColor=GRID,
                                 tickColor=GRID, labelFontSize=12, titleFontSize=12)
            .configure_view(strokeWidth=0).configure_legend(orient="top", title=None, labelColor=INK))


t1, t2, t_adv, t3, t_b, t_s, t4 = st.tabs([
    ":material/show_chart: Cost outlook", ":material/rule: Quote rules", ":material/sell: Price increase",
    ":material/category: Materials", ":material/eco: Sell AirSeT", ":material/energy_savings_leaf: Sustainability",
    ":material/account_tree: Principle"])
own_series = list(w.index) + [s for s in I.ENERGY if s in inp.panel and (s, h) in fc.index]

with t1:
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Material cost, expected", pct(scen[1] - 1), f"{scen[1] - base[1]:+.1%} from the shock",
              delta_color="inverse", border=True, help="Change vs September 2026, central case.")
    e2.metric("Material cost, bad case", pct(scen[2] - 1), f"{scen[2] - base[2]:+.1%} from the shock",
              delta_color="inverse", border=True,
              help=f"High end of the {level:.0%} range: a 1-in-{round(2 / (1 - level))} chance of being worse.")
    e3.metric("Margin at a fixed price", pct(spot["central"], False), f"{spot['central'] - margin0:+.1%} vs target",
              border=True)
    e4.metric("Margin at a fixed price, bad case", pct(spot["adverse"], False),
              f"{spot['adverse'] - floor:+.1%} vs floor ({floor:.0%})", border=True)
    rows = [{"Months": 0, "lo": 100, "mid": 100, "hi": 100, "case": c} for c in ("Our forecast", "With the shock")]
    for hh in I.HORIZONS:
        for case, v in (("Our forecast", bom(hh, False)), ("With the shock", bom(hh))):
            rows.append({"Months": hh, "lo": 100 * v[0], "mid": 100 * v[1], "hi": 100 * v[2], "case": case})
    fan = pd.DataFrame(rows)
    color = alt.Color("case:N", scale=alt.Scale(domain=["Our forecast", "With the shock"], range=[GREEN, BLUE]),
                      legend=alt.Legend(title=None))
    x = alt.X("Months:Q", title="Months after the quote", scale=alt.Scale(domain=[0, 11.5]),
              axis=alt.Axis(values=[0, 3, 6, 9]))
    tip = [alt.Tooltip("case:N", title=" "), alt.Tooltip("Months:Q"), alt.Tooltip("mid:Q", title="Expected", format=".1f"),
           alt.Tooltip("lo:Q", title="Low", format=".1f"), alt.Tooltip("hi:Q", title="High", format=".1f")]
    area = alt.Chart(fan).mark_area(opacity=0.15).encode(x=x, y=alt.Y("lo:Q", title="Material cost (Sept 2026 = 100)",
                                                                          scale=alt.Scale(zero=False)), y2="hi:Q", color=color)
    line = alt.Chart(fan).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True)).encode(
        x=x, y="mid:Q", color=color, tooltip=tip)
    end = fan[fan.Months == 9].assign(txt=lambda d: d["mid"].round().astype(int).astype(str) + "  " + d["case"].str.lower())
    lab = alt.Chart(end).mark_text(align="left", dx=8, fontSize=12, color=INK).encode(x=x, y="mid:Q", text="txt:N")
    st.altair_chart(style(area + line + lab).properties(height=340), width="stretch")
    st.caption(f"Shaded: the {level:.0%} range. Line: the expected path.")
    if show_eur:
        eur0 = float(w["value"].sum())
        st.caption(f"Priced materials today: {eur0:,.0f} EUR per unit. In {h} months: {eur0 * scen[0]:,.0f} / "
                   f"{eur0 * scen[1]:,.0f} / {eur0 * scen[2]:,.0f} EUR (low / expected / high).")

with t2:
    case = st.segmented_control("Case", ["Bad case", "Expected", "Good case"], default="Bad case", required=True)
    col = {"Bad case": "adverse", "Expected": "central", "Good case": "favourable"}[case]
    bars = pd.DataFrame({"Rule": [RULES[r] + ("  ★" if r == (rec[0] if rec else None) else "") for r in mt.index],
                         "Margin": mt[col].to_numpy()})
    bars["Status"] = np.where(bars.Margin >= floor, "Above floor", "Below floor")
    order = bars.sort_values("Margin", ascending=False).Rule.tolist()
    y = alt.Y("Rule:N", sort=order, title=None, axis=alt.Axis(labelLimit=260, labelFontSize=13))
    b = alt.Chart(bars).mark_bar(cornerRadiusEnd=4, height=18).encode(
        x=alt.X("Margin:Q", title="Margin", axis=alt.Axis(format="%")), y=y,
        color=alt.Color("Status:N", scale=alt.Scale(domain=["Above floor", "Below floor"], range=[GREEN, RED])),
        tooltip=[alt.Tooltip("Rule:N"), alt.Tooltip("Margin:Q", format=".1%"), alt.Tooltip("Status:N")])
    inside = bars.Margin >= 0.04  # long bars: white label inside the end; short bars: dark label just right of them
    bars["xpos"] = bars.Margin.clip(lower=0)
    txt = (alt.Chart(bars[inside]).mark_text(align="right", dx=-6, fontSize=12, color="white", fontWeight="bold")
           .encode(x="Margin:Q", y=y, text=alt.Text("Margin:Q", format=".1%"))
           + alt.Chart(bars[~inside]).mark_text(align="left", dx=6, fontSize=12, color=INK, fontWeight="bold")
           .encode(x="xpos:Q", y=y, text=alt.Text("Margin:Q", format=".1%")))
    fl = pd.DataFrame({"x": [floor], "t": [f"floor {floor:.0%}"]})
    rule_floor = (alt.Chart(fl).mark_rule(color=INK, strokeDash=[4, 4], strokeWidth=2).encode(x="x:Q")
                  + alt.Chart(fl).mark_text(align="left", dx=4, dy=-8, fontSize=12, color=INK, baseline="bottom")
                  .encode(x="x:Q", y=alt.value(0), text="t:N"))
    st.altair_chart(style(b + txt + rule_floor).properties(height=36 * len(bars)), width="stretch")
    st.caption(f"★ = rule chosen on 2010-21 data for a material share near {rec[1]:.0%}." if rec else
               "Run `uv run python -m src.business.run_margin_replay` once to see the recommended rule.")

with t3:
    mrows = []
    for s in own_series:
        if (s, h) not in fc.index:
            continue
        b0, b1 = band(s, h, False), band(s, h)
        mrows.append({"Material": names.get(s, s), "Share of cost": 100 * w["weight"].get(s, np.nan),
                      "Our forecast": 100 * (b0[1] - 1), "With the shock": 100 * (b1[1] - 1),
                      "Low": 100 * (b1[0] - 1), "High": 100 * (b1[2] - 1)})
    mdf = pd.DataFrame(mrows)
    st.dataframe(mdf, hide_index=True, width="stretch", column_config={
        "Share of cost": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100,
                                                         help="Energy prices are not BOM materials (blank)."),
        **{c: st.column_config.NumberColumn(format="%+.1f%%") for c in ("Our forecast", "With the shock", "Low", "High")}})
    st.caption(f"Change by {h} months vs September 2026. Low and high: the {level:.0%} range with the shock.")
    pick = st.selectbox("Price path", [s for s in own_series if (s, h) in fc.index], format_func=lambda s: names.get(s, s))
    hist = inp.panel[pick].loc[:I.ORIGIN].dropna().iloc[-48:]
    last = float(fc.loc[(pick, h), "last"])
    hdf = pd.DataFrame({"date": hist.index, "value": 100 * hist.to_numpy() / last})
    fr = pd.DataFrame([{"date": I.ORIGIN, "lo": 100, "mid": 100, "hi": 100}] +
                      [dict(zip(("date", "lo", "mid", "hi"), (I.ORIGIN + pd.DateOffset(months=hh),
                                                              *(100 * v for v in band(pick, hh))))) for hh in I.HORIZONS])
    yy = alt.Y("value:Q", title="Price (Sept 2026 = 100)", scale=alt.Scale(zero=False))
    pc = (alt.Chart(hdf).mark_line(color=MUTED, strokeWidth=2).encode(x=alt.X("date:T", title=None), y=yy,
                                                                       tooltip=[alt.Tooltip("date:T"), alt.Tooltip("value:Q", format=".1f")])
          + alt.Chart(fr).mark_area(opacity=0.15, color=BLUE).encode(x="date:T", y="lo:Q", y2="hi:Q")
          + alt.Chart(fr).mark_line(color=BLUE, strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True)).encode(
              x="date:T", y="mid:Q", tooltip=[alt.Tooltip("date:T"), alt.Tooltip("mid:Q", format=".1f", title="Expected")]))
    st.altair_chart(style(pc).properties(height=300), width="stretch")
    st.caption(f"Grey: last 4 years. Blue: with the shock, {level:.0%} range. "
               f"Source: {fc.loc[(pick, h), 'source']}.")

with t4:
    STEPS = [("1 · You set", ["The offer: product, months the price is fixed", "A price scenario",
                              "The client: tier, order size, country, industry"]),
             ("2 · We simulate", ["Our frozen forecast of every material and energy price, scored once on 2022-26",
                                  "The shock, with the co-movement seen in past crises",
                                  "10,000 lifetimes of AirSeT vs RM6"]),
             ("3 · You see", ["The material cost range", "The margin under 9 quote rules", "Competitors' cost moves",
                              "AirSeT's lifetime cost and the SF6 it avoids"]),
             ("4 · You decide", ["How to quote, and for how long", "How much to raise the price", "When to reopen it",
                                 "Where to push AirSeT, at what premium", "Which suppliers to ask for data first"])]
    for col, (title, items) in zip(st.columns(4), STEPS):
        with col.container(border=True, height=260):
            st.markdown(f"**{title}**\n" + "\n".join(f"- {x}" for x in items))
    st.caption("Same maths as our backtests. Nothing leaves this computer.")
    st.markdown("##### Method")
    st.markdown(
        "- **Forecast.** Our submitted model, frozen and scored on 2022-26. Its range widens when markets are volatile.\n"
        "- **Shock.** One sigma is a typical move over the same number of months. *Worst in history* uses the "
        "largest move seen, with co-movement measured in crises (2008-09, 2020, 2021-22).\n"
        "- **Energy.** Reaches material prices with the delay that fits history best.\n"
        "- **Margin.** Price fixed at the quote, materials bought at the end. Same maths as our margin backtest.\n"
        "- **★ rule.** Chosen on 2010-21 data with a rule fixed in advance.")
    with st.expander("How each price was moved"):
        st.dataframe(moves[h].assign(Material=[names.get(s, s) for s in moves[h].index])
                     .set_index("Material")[["move", "beta", "lag", "sample"]].round(3), width="stretch")

# ------------------------------------------------------------------------------------------------ price advice (Goal A)
with t_adv:
    st.caption("How much of the cost increase to put into the price of one deal: competitors first (RM6), then the "
               "client, never below the margin floor. Same rules as our pricing module.")
    a1, a2, _ = st.columns(3)
    tier = a1.segmented_control("Client", list(cust_cfg["customer_tiers"]), default="standard", required=True,
                                format_func=str.capitalize, key="adv_tier",
                                help="\n".join(f"- **{k.capitalize()}**: {v['examples']}, up to "
                                               f"{v['max_margin_giveup_pts']} margin points"
                                               for k, v in cust_cfg["customer_tiers"].items()))
    deal = a2.segmented_control("Order size", list(DEALS), default=150_000, required=True, key="adv_deal",
                                format_func=lambda d: DEALS[d].capitalize())
    STANCE = {"protect_margin": "Keep our margin", "gain_share": "Look cheaper"}
    if product == "RM6":
        # both stances come from the same advice(); the chart and the gaps do not depend on the stance
        keep, cheap = advice(tier, deal, "protect_margin"), advice(tier, deal, "gain_share")
        cg = keep["cg"]
        cg["Who"] = np.where(cg.Brand.str.startswith("Schneider"), "Us", "Competitors")
        brand_order = cg.sort_values("change", ascending=False)["Brand"].tolist()
        cc = (alt.Chart(cg).mark_bar(cornerRadiusEnd=4, height=18).encode(
                  x=alt.X("change:Q", title=f"Material cost change in {h} months, bad case", axis=alt.Axis(format="%")),
                  y=alt.Y("Brand:N", sort=brand_order, title=None),
                  color=alt.Color("Who:N", scale=alt.Scale(domain=["Us", "Competitors"], range=[GREEN, BLUE]),
                                  legend=None),
                  tooltip=[alt.Tooltip("Brand:N"), alt.Tooltip("change:Q", format=".1%")])
              + alt.Chart(cg).mark_text(align="left", dx=6, fontSize=12, color=INK).encode(
                  x="change:Q", y=alt.Y("Brand:N", sort=brand_order), text=alt.Text("change:Q", format=".1%")))
        st.altair_chart(style(cc).properties(height=34 * len(cg)), width="stretch")
        st.caption("Each brand's material mix from the competitor teardowns, moved by the same scenario as ours.")

        # per-competitor readout: gap vs Schneider (teardown) in points, same values as the chart
        comp = cg[cg.brand.notna() & (cg.brand != "Competitor average")].sort_values("change", ascending=False)
        gap = dict(zip(comp.brand, comp.gap_pp))
        harder = [b for b, g in gap.items() if g > 0]
        less = [b for b, g in gap.items() if g < 0]

        def pts(g: float) -> str:
            return f"{g:+.1f}".replace("-", "−") + " pts"

        def join_names(xs: list, conj: str = "and") -> str:
            return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + f" {conj} " + xs[-1]

        with st.container(border=True):
            st.markdown("  \n".join(f":{'green' if g > 0 else 'red' if g < 0 else 'gray'}[**{b}** {pts(g)}]"
                                    for b, g in gap.items()))
            summary = f"Median competitor: {pts(float(np.median(list(gap.values()))))} vs us. "
            summary += (f"{len(harder)} of {len(gap)} competitors are hit harder than us ({', '.join(harder)}). "
                        if harder else f"None of the {len(gap)} competitors is hit harder than us. ")
            if less:
                summary += f"{join_names(less)} {'is' if len(less) == 1 else 'are'} hit less."
            st.markdown(summary)
            st.caption("Gap = competitor's material cost change minus Schneider's (teardown), bad case, in points. "
                       "Green: hit harder than us. Red: hit less.")

        # recommended stance, from the count above; X, Y, Z are the bad-case values of the two stances
        k_bad, c_bad = keep["res"]["Bad case"], cheap["res"]["Bad case"]
        # the pricing rule sets the pass-through from the AVERAGE gap; the buttons only differ when it is >= +2 pts
        same = (f"Both buttons give the same price here, because on the average competitor our rule passes on "
                f"{keep['pt']:.0%} of the increase (margin {pct(k_bad['margin_after'], False)}, floor {floor:.0%})."
                if keep["pt"] == cheap["pt"] else None)
        if len(harder) > len(gap) / 2:
            rec_key = "protect_margin"
            rec_msg = ("**Recommended: Keep our margin.** Most rivals' costs rise more than ours, so they'll have to "
                       "raise prices too. " + (same or f"Pass on {keep['pt']:.0%}."))
        elif len(less) > len(gap) / 2:
            rec_key = "gain_share"
            rec_msg = ("**Recommended: Look cheaper.** Most rivals' costs rise less than ours. " + (
                same or f"Absorb part of the increase: this costs "
                        f"{(k_bad['margin_after'] - c_bad['margin_after']) * 100:.1f} margin points (margin stays at "
                        f"{pct(c_bad['margin_after'], False)}, floor {floor:.0%})."))
        else:
            rec_key = None
            rec_msg = (f"**Mixed:** {join_names(harder)} {'is' if len(harder) == 1 else 'are'} hit harder, "
                       f"{join_names(less)} {'is' if len(less) == 1 else 'are'} hit less. Pricing against "
                       f"{join_names(less, 'or')}? Consider Look cheaper. Against the others: Keep our margin."
                       if harder and less else "**Mixed:** competitors move like us.")
        st.info(rec_msg, icon=":material/recommend:")
        strategy = st.columns(3)[0].segmented_control(
            "If competitors are hit harder", list(STANCE), default=comp_cfg["strategy_when_competitors_hit_harder"],
            required=True, key="adv_strategy",
            format_func=lambda k: STANCE[k] + (" (recommended)" if k == rec_key else ""))
        adv = keep if strategy == "protect_margin" else cheap
    else:
        st.info("No teardown data for SF6-free rivals.", icon=":material/groups:")
        adv = advice(tier, deal, adv_now["adv_strategy"])  # stance unused for AirSeT: no competitor pass-through
    gpts, res = adv["gpts"], adv["res"]
    b1, b2, b3 = st.columns(3)
    bad = res["Bad case"]
    b1.metric("Raise the price by (bad case)", pct(bad["increase_eur"] / price),
              f"expected case {pct(res['Expected']['increase_eur'] / price)}", delta_color="off", delta_arrow="off", border=True)
    b2.metric("Margin after the increase", pct(bad["margin_after"], False), f"floor {floor:.0%}", delta_color="off", delta_arrow="off",
              border=True)
    b3.metric("Margin points given to the client", f"{gpts:g}", f"{tier} client, order {DEALS[deal]}",
              delta_color="off", delta_arrow="off", border=True)
    if bad["floor_binding"]:
        st.warning("The margin floor binds: the price goes up enough to keep the floor, whatever the levers say.",
                   icon=":material/vertical_align_top:")
    if show_eur:
        assumed = ", AirSeT assumed" if product == "AirSeT" and inp.prices["airset_price_assumed"] else ""
        st.caption(f"Selling price {price:,.0f} EUR (synthetic benchmark{assumed}); "
                   f"cost increase in the bad case {cost_base * (scen[2] - 1):,.0f} EUR per unit.")
    st.markdown("##### Price-review trigger")
    cm = inp.biz.get("cost_monthly")
    col = f"{product}_cost_eur"
    if cm is None or col not in cm:
        st.info("Run `make business` once to see the realised cost moves.")
    else:
        cost = cm.assign(month=pd.to_datetime(cm["month"])).set_index("month")[col].dropna().loc[:I.ORIGIN]
        months = [m.strftime("%b %Y") for m in cost.index[-13:-1]]
        base_m = pd.Timestamp(st.select_slider("Base month written in the contract", months, value=months[-6]))
        realised = float(cost.iloc[-1] / cost.loc[base_m] - 1)
        r1, r2, r3 = st.columns(3)
        r1.metric(f"Cost moved since {base_m:%b %Y}", pct(realised), "review now" if abs(realised) > trig else
                  "inside the band", delta_color="inverse" if abs(realised) > trig else "off", border=True)
        r2.metric(f"Expected move in {h} months", pct(scen[1] - 1), border=True)
        r3.metric(f"Bad case in {h} months", pct(scen[2] - 1), "review likely before delivery"
                  if abs(scen[2] - 1) > trig else "inside the band",
                  delta_color="inverse" if abs(scen[2] - 1) > trig else "off", border=True)
        st.caption(f"Rule: either side can open a price review when the product's raw-material cost moves more than "
                   f"±{trig:.0%} from the contract's base month (completed months only).")
    lags = inp.biz.get("lags")
    if lags is not None:
        st.markdown("##### When suppliers pass on a market move")
        lt = lags[lags["sample"] == "full"]
        st.dataframe(pd.DataFrame({"Market move → price we pay": lt["name"].str.replace("_", " "),
                                   "Half passed on (months)": lt["lag_50"], "80% passed on (months)": lt["lag_80"]}),
                     hide_index=True, width="stretch")
        st.caption("Negotiate within this window after an index move. Distributed-lag regressions on monthly changes.")

# ------------------------------------------------------------------------------------------------ AirSeT strategy (Goal B)
with t_b:
    gcfg = gb.load()
    st.caption(f"Markets with no SF6 ban: why a client should choose AirSeT, and what Schneider should charge. "
               f"Market facts as of {gcfg['as_of']}, each with its source.")
    g1, g2 = st.columns(2)
    country = g1.selectbox("Country", list(gcfg["countries"]), format_func=lambda k: gcfg["countries"][k]["label"])
    industry = g2.selectbox("Client industry", list(gcfg["industries"]),
                            format_func=lambda k: gcfg["industries"][k]["label"])
    C, IND = gcfg["countries"][country], gcfg["industries"][industry]
    badge, score = gb.strength(gcfg, country, industry)
    {"STRONG": st.success, "MEDIUM": st.warning, "WEAK": st.error}[badge](
        f"**{C['label']} · {IND['label']}: {badge} case.** {gcfg['actions'][badge]}", icon={"STRONG": ":material/trending_up:", "MEDIUM": ":material/trending_flat:",
                                         "WEAK": ":material/trending_down:"}[badge])
    st.caption(f"{C['selling_tip']} Score {score} of 5 (reporting pull, SF6 rules momentum, client's climate focus).")

    be0 = inp.tco.loc[inp.tco["carbon_price_eur_t"] == 0, "breakeven_premium"]  # no carbon price in these markets
    d0 = inp.tco.loc[inp.tco["carbon_price_eur_t"] == 0, "diff"]
    gap_now = float(inp.weights[inp.weights["product"] == "AirSeT"]["value"].sum()
                    / inp.weights[inp.weights["product"] == "RM6"]["value"].sum() - 1)
    gap_h = (1 + gap_now) * bom_for("AirSeT", h)[1] / bom_for("RM6", h)[1] - 1
    cu = inp.weights.set_index(["product", "series"])["weight"]
    conv = (f"AirSeT's raw materials cost {gap_now:+.0%} vs RM6 today ({gap_h:+.0%} in {h} months in this scenario), "
            f"with less copper ({cu.get(('AirSeT', 'D01'), 0):.0%} of cost vs {cu.get(('RM6', 'D01'), 0):.0%}), so "
            f"the gap narrows when copper rises.")

    def link(text, url):
        return f"{text} ([source]({url}))"

    pitch = [link(o["note"], o["source"]) for o in gb.overrides(gcfg, country, industry)] + list(IND["pitch"])
    left, right = st.columns([3, 2])
    with left:
        if industry == "price_only":
            st.markdown("**Lead argument: cost convergence.** " + conv)
        st.markdown("##### What to tell the client")
        st.markdown("\n".join(f"- {p}" for p in pitch[:4]))
        if IND.get("note"):
            st.caption(IND["note"])
        if IND.get("sources"):
            st.caption("Industry sources: " + ", ".join(f"[{i + 1}]({u})" for i, u in enumerate(IND["sources"])))
        st.markdown("##### Why now, here")
        st.markdown(f"- {C['regulation_outlook']}\n- {link(C['disclosure']['text'], C['disclosure']['source'])}\n"
                    + "\n".join(f"- {link(p['text'], p['source'])}" for p in C["local_proof"])
                    + "\n- " + gcfg["pfas_point"]["text"] + " ("
                    + ", ".join(f"[source]({u})" for u in gcfg["pfas_point"]["sources"]) + ")")
        with st.expander("SF6 rules spreading elsewhere (upside only, no date assumed)"):
            st.markdown("\n".join(f"- {link(r['text'], r['source'])}" for r in gcfg["rules_spreading"]))
    with right:
        eol_opts = gcfg["eol_scenarios"]["options"]
        eol_key = st.segmented_control("Gas lost when the RM6 is scrapped", list(eol_opts), default="ours",
                                       required=True, format_func=lambda k: eol_opts[k]["label"],
                                       help="Fraunhofer IEE study of medium-voltage switchgear: 1.5% with best "
                                            "practice, up to 40% where recovery is not enforced, best guess 10%.")
        tco_e = gb.sf6_scenario(inp.tco, float(inp.prices["RM6"]), eol_opts[eol_key].get("pct"))
        st.metric("Lifetime break-even premium", pct(float(be0.median()), False),
                  f"range {be0.quantile(.1):.0%}-{be0.quantile(.9):.0%}, no carbon price", delta_color="off", delta_arrow="off",
                  border=True, help="Below this premium over RM6, AirSeT pays for itself through lower maintenance "
                                    "over 30-40 years (simulated lifetimes with no carbon price, our model).")
        st.metric("SF6 avoided per unit", f"{tco_e['sf6_lifetime_tco2e'].median():.1f} t CO₂e",
                  f"gas held: {(inp.tco.sf6_charge_kg * inp.tco.gwp_sf6).min() / 1000:.0f}-"
                  f"{(inp.tco.sf6_charge_kg * inp.tco.gwp_sf6).max() / 1000:.0f} t CO₂e", delta_color="off", delta_arrow="off",
                  border=True, help="Leaks and end-of-life losses over the unit's life (our model); 'gas held' is the "
                                    "total if all of it escaped.")
        st.caption(f"End-of-life losses: [Fraunhofer IEE]({gcfg['eol_scenarios']['source']}).")
        if industry != "price_only":
            st.caption(conv)

    st.markdown("##### What to charge (Schneider only, not for the client)")
    lo_b, hi_b = gb.premium_band(gcfg, country, industry)
    prem = st.slider("AirSeT price premium over RM6", 0, 30, int(round((lo_b + hi_b) / 2)), 1, format="%d%%") / 100
    floor_p, ceil_p = gap_h * share, float(be0.median())
    p1, p2, p3 = st.columns(3)
    p1.metric("Suggested premium", f"{lo_b:.0f}-{hi_b:.0f}%", f"{IND['label'].split(' /')[0].lower()}, "
              f"{C['label'].split(' (')[0]}", delta_color="off", delta_arrow="off", border=True,
              help="Our proposal: industry band, shifted by the country (e.g. Brazil -2 points).")
    p2.metric(f"AirSeT cheaper to own at +{prem:.0%}", pct(float((be0 > prem).mean()), False),
              "of simulated lifetimes", delta_color="off", delta_arrow="off", border=True)
    p3.metric("Extra build cost to cover", f"{floor_p:.1%} of the price", f"raw materials, in {h} months",
              delta_color="off", delta_arrow="off", border=True, help="AirSeT's extra raw-material cost x the materials share of the "
                                                   "price. Electronics are not in the BOM.")
    if prem < floor_p:
        st.error("Below AirSeT's extra build cost: every unit sold loses margin.", icon=":material/error:")
    elif prem > ceil_p:
        st.warning("Above the lifetime break-even: only clients who value the climate case gain from it.",
                   icon=":material/warning:")
    else:
        st.success(f"Inside the window: the premium covers the extra build cost, and AirSeT is still cheaper to own "
                   f"in {float((be0 > prem).mean()):.0%} of cases.", icon=":material/check_circle:")
    win = pd.DataFrame({"x": [floor_p], "x2": [ceil_p]})
    band_df = pd.DataFrame({"x": [lo_b / 100], "x2": [hi_b / 100]})
    wc = (alt.Chart(win).mark_rect(color=GREEN, opacity=0.18, height=40).encode(
              x=alt.X("x:Q", title="AirSeT premium over RM6", scale=alt.Scale(domain=[0, 0.30]),
                      axis=alt.Axis(format="%")), x2="x2:Q")
          + alt.Chart(band_df).mark_rect(color=BLUE, opacity=0.35, height=10).encode(x="x:Q", x2="x2:Q")
          + alt.Chart(pd.DataFrame({"x": [prem]})).mark_rule(color=INK, strokeWidth=3).encode(x="x:Q")
          + alt.Chart(pd.DataFrame({"x": [floor_p, ceil_p], "t": ["extra build cost", "lifetime break-even"]}))
          .mark_text(dy=-28, fontSize=12, color=MUTED).encode(x="x:Q", text="t:N"))
    st.altair_chart(style(wc).properties(height=110), width="stretch")
    st.caption(f"Green: the profitable window. Blue: suggested band for this client. Line: your premium. "
               f"Contract: {C['contract_note']}" + (f" {C['premium_note']}" if C.get("premium_note") else ""))
    if show_eur:
        at_p = gb.sf6_scenario(inp.tco[inp.tco["carbon_price_eur_t"] == 0], float(inp.prices["RM6"]),
                               eol_opts[eol_key].get("pct"), prem)
        per_t = float(at_p["diff"].median() / at_p["sf6_lifetime_tco2e"].median())
        st.caption(f"At +{prem:.0%} the client pays {per_t:,.0f} EUR per tonne of CO₂e avoided over the life "
                   f"(median, local screen only)." if per_t > 0 else
                   f"At +{prem:.0%} the client saves money and avoids the SF6 (median {-per_t:,.0f} EUR per tonne).")
    with st.expander("What not to claim"):
        st.markdown("\n".join(f"- {link(d['text'], d['source'])}" for d in gcfg["do_not_claim"]))
    with st.expander("Caveats"):
        st.markdown("\n".join(f"- {c}" for c in gcfg["caveats"])
                    + "\n- Costed on RM6 → RM AirSeT; we had no SM6 / SM AirSeT cost data, the same logic applies.")
    held_t = inp.tco.sf6_charge_kg * inp.tco.gwp_sf6 / 1000
    brief = gb.client_brief(gcfg, country, industry, {
        "breakeven": float(be0.median()), "sf6_t": float(tco_e["sf6_lifetime_tco2e"].median()),
        "held_lo": float(held_t.min()), "held_hi": float(held_t.max())})
    st.download_button("Download the client brief", brief, file_name=f"airset_brief_{country}_{industry}.md",
                       mime="text/markdown", icon=":material/download:",
                       help="Pitch, our model numbers and the local context, with sources. No Schneider pricing.")

# ------------------------------------------------------------------------------------------------ sustainability
with t_s:
    st.markdown("##### SF6 in one RM6")
    sf6 = inp.tco["sf6_lifetime_tco2e"]
    held = inp.tco.sf6_charge_kg * inp.tco.gwp_sf6 / 1000
    s1, s2, s3 = st.columns(3)
    s1.metric("Gas held, if it all escaped", f"{held.min():.0f}-{held.max():.0f} t CO₂e", "SF6 is 24,300 x CO₂",
              delta_color="off", delta_arrow="off", border=True)
    s2.metric("Emitted over its life", f"{sf6.median():.1f} t CO₂e", f"range {sf6.quantile(.1):.1f}-"
              f"{sf6.quantile(.9):.1f}, leaks and end of life", delta_color="off", delta_arrow="off", border=True)
    s3.metric("RM AirSeT", "0 t CO₂e", "no SF6 inside", delta_color="off", delta_arrow="off", border=True)
    fr = gb.sf6_scenario(inp.tco, float(inp.prices["RM6"]), 10)["sf6_lifetime_tco2e"].median()
    st.caption(f"These emissions sit in the client's own direct emissions (Scope 1), because the client owns the "
               f"switchgear. With Fraunhofer's 10% end-of-life loss, one RM6 emits about {fr:.1f} t CO₂e.")

    st.markdown("##### Supplier emissions (estimates)")
    if inp.suppliers is None:
        st.info("Run `make business` once to estimate supplier emissions.")
    else:
        names_ok = st.toggle("Show supplier names (NDA, local screen only)", value=False)
        scope = st.segmented_control("Scope", ["s12", "s3"], default="s12", required=True,
                                     format_func={"s12": "Scope 1+2", "s3": "Scope 3"}.get)
        su = inp.suppliers.sort_values("priority_rank").reset_index(drop=True)
        su["Supplier"] = su["name"] if names_ok else [f"Supplier {i}" for i in range(1, len(su) + 1)]
        order = su["Supplier"].tolist()
        st.success(f"**Ask first:** {', '.join(order[:3])}. They have the largest estimated emissions (Scope 1, 2 and 3).",
                   icon=":material/person_search:")
        conf_colors = alt.Scale(domain=["high", "medium", "low"], range=[GREEN, BLUE, "#9FA0A4"])
        base_c = alt.Chart(su).encode(y=alt.Y("Supplier:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False)))
        whisk = (base_c.mark_rule(strokeWidth=2, color=MUTED).encode(
                     x=alt.X(f"{scope}_p10_t:Q", title="t CO₂e per year (whole company)", scale=alt.Scale(type="log")),
                     x2=f"{scope}_p90_t:Q")
                 + base_c.mark_point(filled=True, size=90).encode(
                     x=f"{scope}_p50_t:Q", color=alt.Color("confidence:N", scale=conf_colors, title="Confidence"),
                     tooltip=[alt.Tooltip("Supplier:N"), alt.Tooltip(f"{scope}_p50_t:Q", title="Estimate", format=",.0f"),
                              alt.Tooltip(f"{scope}_p10_t:Q", title="Low", format=",.0f"),
                              alt.Tooltip(f"{scope}_p90_t:Q", title="High", format=",.0f"),
                              alt.Tooltip("confidence:N")]))
        st.altair_chart(style(whisk).properties(height=42 * len(su)), width="stretch")
        st.caption("Dot: estimate. Line: likely range (log scale). Sorted by engagement priority. Whole-company "
                   "figures, not Schneider's share.")
        val = inp.biz.get("sup_validation")
        with st.expander("How we estimate, and how sure we are"):
            st.markdown(
                "- **Scope 1 and 2**: a company's own direct emissions and the electricity it buys. A supplier's Scope "
                "1+2 become part of our Scope 3 (purchased goods).\n"
                "- **Method**: each supplier's revenue (PitchBook) times the emissions per euro of similar companies "
                "(same sector and country, else same sector). The comparison pool is **synthetic**, a demo of the "
                "method.\n"
                "- **Spend-based estimate**: a cross-check only, never added, to avoid double counting.\n"
                "- **Next step**: replace estimates with the suppliers' own data.")
            if val is not None:
                v = val.set_index("target")
                st.markdown(f"- **Accuracy on the comparison pool** (leave one out): typical error "
                            f"{v.at['s12', 'median_abs_pct_error']:.0%} (Scope 1+2) and "
                            f"{v.at['s3', 'median_abs_pct_error']:.0%} (Scope 3); the likely range contains the truth "
                            f"{v.at['s12', 'band_coverage_p10_p90']:.0%} and {v.at['s3', 'band_coverage_p10_p90']:.0%} "
                            f"of the time.")
        st.markdown("##### What to ask suppliers, in priority order")
        st.markdown("Their energy and fuel use, a carbon footprint for the parts they supply, and their CDP or "
                    f"science-based-target status. {int((su['confidence'] == 'low').sum())} of {len(su)} estimates are "
                    "low confidence, so their own data would change the picture most.")

    st.markdown("##### What our models cost to run")
    st.dataframe(inp.footprint, hide_index=True, width="stretch", column_config={
        "Minutes": st.column_config.NumberColumn(format="%.1f"), "kWh": st.column_config.NumberColumn(format="%.3f"),
        "gCO2e": st.column_config.NumberColumn("g CO₂e", format="%.1f")})
    st.caption("Laptop CPU at about 30 W, EU electricity at about 0.25 kg CO₂e per kWh (estimates). We chose small "
               "statistical models partly for this reason. This app adds seconds per use.")
