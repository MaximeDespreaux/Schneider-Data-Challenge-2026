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

from src.app import inputs as I
from src.app import scenarios as sc
from src.business.margin_replay import Economics
from src.testbench.targets import TARGET_IDS

st.set_page_config(page_title="Quote risk simulator", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")

# Schneider Electric greens + neutrals. Chart pair validated for colour-blind readers (dataviz validator).
GREEN, LIFE_GREEN, BLUE, RED, INK, MUTED, GRID = "#009530", "#3DCD58", "#42B4E6", "#DC0A0A", "#333333", "#626469", "#E6E8E6"
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
TOUR = [
    ("Welcome", "This tool shows what happens to the **margin of a fixed-price offer** if raw-material prices move "
                "between the quote and the purchase. It takes 30 seconds to learn."),
    ("1 · Choose the offer", "Pick the **product** and **how many months** the price stays fixed."),
    ("2 · Choose a scenario", "Pick what might happen to prices, from **no shock** to a **worst-in-history** move."),
    ("3 · Read the verdict", "The coloured box tells you if a fixed price is safe and, if not, the **cheapest quote "
                             "rule** that keeps the margin above the floor."),
    ("Go further", "The tabs show the cost outlook, every quote rule and each material. "
                   "**Settings** (top-left arrow) hold the assumptions: margins, confidence level, which price moves."),
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
              f"<span style='color:{MUTED}'>Price fixed in September 2026, materials bought later. "
              "What happens to the margin?</span>", unsafe_allow_html=True)
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
show_eur = sb.toggle("Show EUR levels (local screen only)", value=False)
eco = Economics(material_share=share, target_margin=margin0, floor=floor)

# ------------------------------------------------------------------------------------------------ computation
fc = inp.fc.set_index(["series", "horizon"])
series = list(w.index) + [s for s in I.ENERGY if s in inp.panel and (s, h) in fc.index]
moves = {hh: sc.scenario_moves(inp.panel, series, hh, I.ORIGIN, shock, follow, set(I.ENERGY)) for hh in I.HORIZONS}


def band(s: str, hh: int, shocked: bool = True):
    r = fc.loc[(s, hh)]
    lo, hi = sc.scale_band(r.p10, r.p50, r.p90, level)
    f = float(np.exp(moves[hh].at[s, "move"])) if shocked and s in moves[hh].index else 1.0
    return lo / r["last"] * f, r.p50 / r["last"] * f, hi / r["last"] * f


bom_id = TARGET_IDS[product]


def bom(hh: int, shocked: bool = True):
    comps = pd.DataFrame({"weight": w["weight"],
                          "ratio50": [fc.loc[(s, hh)].p50 / fc.loc[(s, hh)]["last"] for s in w.index]}, index=w.index)
    q = fc.loc[(bom_id, hh)]
    mv = moves[hh]["move"] if shocked else pd.Series(0.0, index=comps.index)
    return sc.bom_band((q.p10, q.p50, q.p90), comps, mv, level)


base, scen = bom(h, shocked=False), bom(h)
cu_pts = band("D01", h) if "D01" in w.index else (1.0, 1.0, 1.0)
en_pts = band(plant, h) if plant and (plant, h) in fc.index else (1.0, 1.0, 1.0)
q0 = fc.loc[(bom_id, h)]
tab = sc.margin_table(eco, h, dict(zip(POINTS, zip(scen, cu_pts))),
                      {"p50": q0.p50 / 100, "p90": q0.p90 / 100, "hist_p90": inp.hist_p90.get((product, h), np.nan)},
                      w_cu=float(w.loc["D01", "weight"]) if "D01" in w.index else 0.0,
                      cu_fwd_ratio=float(np.exp(inp.cu_term * h / 3)), energy_share=e_share,
                      energy_ratio=dict(zip(POINTS, en_pts)))
mt = tab.pivot_table(index="policy", columns="point", values="margin").reindex(list(RULES))
rec = I.recommended_policy(inp, product, share, h)
rec_name = RULES.get(rec[0], rec[0]) if rec else None
spot = mt.loc["spot"]

# ------------------------------------------------------------------------------------------------ step 3: verdict
st.markdown("##### 3 · Result")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Material cost, expected", pct(scen[1] - 1), f"{scen[1] - base[1]:+.1%} from the shock",
          delta_color="inverse", border=True, help="Change vs September 2026, central case.")
k2.metric("Material cost, bad case", pct(scen[2] - 1), f"{scen[2] - base[2]:+.1%} from the shock",
          delta_color="inverse", border=True,
          help=f"High end of the {level:.0%} range: a 1-in-{round(2 / (1 - level))} chance of being worse.")
k3.metric("Margin at a fixed price", pct(spot["central"], False), f"{spot['central'] - margin0:+.1%} vs target",
          border=True)
k4.metric("Margin at a fixed price, bad case", pct(spot["adverse"], False),
          f"{spot['adverse'] - floor:+.1%} vs floor ({floor:.0%})", border=True)

if "Extreme" in scenario:
    largest = max(abs(sc.worst_move(inp.panel[s], h, I.ORIGIN, up)) / sc.sigma(inp.panel[s], h, I.ORIGIN)
                  for s in w.index)
    st.info(f"Stress test only. The largest {h}-month move in history was {largest:.1f} sigma, not 10.",
            icon=":material/science:")
# safer rule here: the cheapest one (lowest price markup) that keeps the bad case above the floor, as in our backtest rule
mk = tab[tab.point == "central"].set_index("policy")["markup"].reindex(mt.index)
safe = mt.index[mt["adverse"] >= floor]
FRICTION = ["spot", "p50_model", "hedge_cu", "p90_hist", "p90_model", "clause_10", "hedge_cu_clause_5", "clause_5",
            "clause_3"]  # ties on markup: fewer price revisions first (backtest tie-break)
order = pd.DataFrame({"mk": mk.round(4), "fr": [FRICTION.index(r) for r in mk.index]}, index=mk.index)
best = order.loc[safe].sort_values(["mk", "fr"]).index[0] if len(safe) else mt["adverse"].idxmax()
if spot["adverse"] >= floor:
    st.success(f"**A fixed price holds.** Even in the bad case the margin stays at {pct(spot['adverse'], False)}, "
               f"above the {floor:.0%} floor.", icon=":material/check_circle:")
elif len(safe):
    st.warning(f"**A fixed price is risky here.** In the bad case the margin falls to {pct(spot['adverse'], False)}, "
               f"below the {floor:.0%} floor. Cheapest safe rule: **{RULES[best]}**.", icon=":material/warning:")
else:
    st.error(f"**No quote rule keeps the margin above the {floor:.0%} floor in this scenario.** "
             f"**{RULES[best]}** loses the least. Re-quote or shorten the offer.", icon=":material/error:")
if rec_name:
    st.caption(f"★ Default rule from our 2010-21 backtest for this offer: {rec_name}. Quote rules tab for all options.")
if e_share == 0 and scenario == "Plant power spike":
    st.caption("The spike reaches the margin through material prices only. Add the plant energy share in Settings "
               "to include the direct cost.")


# ------------------------------------------------------------------------------------------------ charts
def style(chart):
    return (chart.configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID, domainColor=GRID,
                                 tickColor=GRID, labelFontSize=12, titleFontSize=12)
            .configure_view(strokeWidth=0).configure_legend(orient="top", title=None, labelColor=INK))


t1, t2, t3, t4 = st.tabs([":material/show_chart: Cost outlook", ":material/rule: Quote rules",
                          ":material/category: Materials", ":material/help: How it works"])

with t1:
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
    for s in series:
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
    pick = st.selectbox("Price path", [s for s in series if (s, h) in fc.index], format_func=lambda s: names.get(s, s))
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
