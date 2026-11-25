"""Scenario app: material and energy shocks -> BOM cost -> margin by quote policy. LOCAL ONLY (licensed data).

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

st.set_page_config(page_title="Material cost scenarios", layout="wide")

PRESETS = {
    "No shock (our forecast only)": dict(kind="sigma", k=0.0),
    "Market noise (1 sigma)": dict(kind="sigma", k=1.0),
    "Rare event (2 sigma)": dict(kind="sigma", k=2.0),
    "Worst move in history (spike or crash, see direction)": dict(kind="worst", k=0.0),
    "Extreme stress test (10 sigma, beyond history)": dict(kind="sigma", k=10.0),
    "Power spike at the plant": dict(kind="worst", k=0.0, driver="plant"),
    "Custom size": dict(kind="sigma", k=None),
}
POLICY_NAMES = {"spot": "Fixed price on today's cost", "p50_model": "Fixed price on our median forecast",
                "p90_model": "Fixed price with P90 buffer (our forecast)", "p90_hist": "Fixed price with historical P90 buffer",
                "clause_3": "Index clause +/-3%", "clause_5": "Index clause +/-5%", "clause_10": "Index clause +/-10%",
                "hedge_cu": "Copper bought forward", "hedge_cu_clause_5": "Copper forward + clause +/-5%"}
POINTS = ("favourable", "central", "adverse")


@st.cache_resource(show_spinner="Loading the graded forecasts and the price history...")
def load(tag: str) -> I.Inputs:
    return I.load(tag)


def pct(x: float) -> str:
    return f"{x:+.1%}"


# ------------------------------------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.caption("Runs on this computer only. It reads licensed and confidential data: never deploy it on a server.")
tag = sb.selectbox("Forecast source", I.sources(),
                   format_func=lambda t: "Compute the 2026-09 forecast here (first run ~1-2 min)" if t == I.FRESH
                   else f"Graded run {t}",
                   help="Graded runs (newest first) come from the frozen backtest. Without one, the same model is "
                        "run once at the 2026-09 origin and cached.")
inp = load(tag)
product = sb.radio("Product", list(TARGET_IDS), horizontal=True)
h = sb.radio("Horizon (months after the quote)", I.HORIZONS, index=1, horizontal=True)
level = sb.select_slider("Confidence level of the range", [0.50, 0.80, 0.90, 0.98], value=0.80,
                         format_func=lambda v: f"{v:.0%}")

sb.subheader("Shock scenario")
preset = sb.selectbox("Scenario", list(PRESETS))
p = dict(PRESETS[preset])
plant = I.plant_power(inp.assumptions, product)  # from the local config (NDA), may be None
if p.get("driver") == "plant":
    if plant in inp.panel:
        p["driver"] = plant
        sb.caption(f"Plant power price: {I.ENERGY.get(plant, plant)} (local config).")
    else:
        sb.warning("Set plant_power_series in config/assumptions.yaml to use this scenario. No shock applied.")
        p = dict(kind="sigma", k=0.0)
up = sb.radio("Direction", ["Prices up (cost shock)", "Prices down"], horizontal=True).startswith("Prices up")
w = inp.weights[inp.weights["product"] == product].set_index("series")
names = {s: m.replace("_", " ") for s, m in w["material"].items()} | I.ENERGY
drivers = [sc.MARKET] + list(w.index) + [s for s in I.ENERGY if s in inp.panel]
label = {sc.MARKET: "All materials and energy at once"} | names
default_driver = p.get("driver", sc.MARKET)
driver = sb.selectbox("What is shocked", drivers, index=drivers.index(default_driver), format_func=label.get,
                      disabled=p["k"] == 0.0 and p["kind"] == "sigma")
k = p["k"] if p["k"] is not None else sb.slider("Shock size (sigma)", 0.0, 10.0, 2.0, 0.5)
follow = True
if driver != sc.MARKET:
    follow = sb.radio("Other prices", ["Others follow (historical co-movement)", "Only this material moves"]
                      ).startswith("Others")
shock = sc.Shock(driver=driver, kind=p["kind"], k=k, up=up)

sb.subheader("Economics (assumptions)")
eco0 = I.default_economics(inp, product)
share = sb.slider("Material cost, share of selling price", 0.10, 0.80, float(eco0.material_share), 0.01)
margin0 = sb.slider("Target margin", 0.0, 0.50, float(eco0.target_margin), 0.01)
floor = sb.slider("Margin floor", 0.0, 0.40, float(eco0.floor), 0.01)
e_share = sb.slider("Plant energy cost, share of price", 0.0, 0.10, 0.0, 0.005, format="%.3f",
                    help="Assumption, Schneider data not available. 0 by default.")
sb.caption("Plant energy share: assumption, Schneider data not available.")
show_eur = sb.checkbox("Show EUR levels (local screen only)", value=False,
                       help="Off: everything is indexed to 100 at the origin, safe for screenshots.")
eco = Economics(material_share=share, target_margin=margin0, floor=floor)

# ------------------------------------------------------------------------------------------------ scenario
fc = inp.fc.set_index(["series", "horizon"])
energy = set(I.ENERGY)
series = list(w.index) + [s for s in I.ENERGY if s in inp.panel and (s, h) in fc.index]
moves = {hh: sc.scenario_moves(inp.panel, series, hh, I.ORIGIN, shock, follow, energy) for hh in I.HORIZONS}


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
cu = "D01" if "D01" in w.index else None
cu_pts = band(cu, h) if cu else (1.0, 1.0, 1.0)
en_pts = band(plant, h) if (plant, h) in fc.index else (1.0, 1.0, 1.0)
q0 = fc.loc[(bom_id, h)]
tab = sc.margin_table(eco, h, dict(zip(POINTS, zip(scen, cu_pts))),
                      {"p50": q0.p50 / 100, "p90": q0.p90 / 100, "hist_p90": inp.hist_p90.get((product, h), np.nan)},
                      w_cu=float(w.loc[cu, "weight"]) if cu else 0.0, cu_fwd_ratio=float(np.exp(inp.cu_term * h / 3)),
                      energy_share=e_share, energy_ratio=dict(zip(POINTS, en_pts)))
rec = I.recommended_policy(inp, product, share, h)

# ------------------------------------------------------------------------------------------------ page
st.title(f"{product}: material cost and margin, {h} months after a quote made in 2026-09")
if "10 sigma" in preset:
    largest = max(abs(sc.worst_move(inp.panel[s], h, I.ORIGIN, up)) / sc.sigma(inp.panel[s], h, I.ORIGIN)
                  for s in w.index)
    st.warning(f"Extreme stress test. A 10 sigma move is far outside the price history: the largest {h}-month move "
               f"of any {product} material was {largest:.1f} sigma. Use it only to test the quote rules.")
if "plant" in preset and e_share == 0:
    st.info("The plant power spike reaches the margin through material prices only. Set the plant energy share "
            "(sidebar) to add the direct cost of power in the plant.")

spot = tab[tab.policy == "spot"].set_index("point")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Material cost, central", pct(scen[1] - 1), f"{scen[1] - base[1]:+.1%} vs no shock", delta_color="inverse")
c2.metric(f"Material cost, adverse ({level:.0%} range)", pct(scen[2] - 1))
c3.metric("Margin at a fixed price, central", f"{spot.at['central', 'margin']:.1%}",
          f"{spot.at['central', 'margin'] - margin0:+.1%} vs target")
c4.metric("Margin at a fixed price, adverse", f"{spot.at['adverse', 'margin']:.1%}",
          "below floor" if spot.at["adverse", "breach"] else "above floor",
          delta_color="inverse" if spot.at["adverse", "breach"] else "normal")

rec_txt = (f"The quote rule chosen on 2010-21 data for this product, horizon and a material share near "
           f"{rec[1]:.0%} is **{POLICY_NAMES.get(rec[0], rec[0])}**." if rec else
           "Run `uv run python -m src.business.run_margin_replay` once to add the recommended quote rule.")
st.markdown(
    f"Under **{preset.lower()}** ({label[driver].lower()}, {'up' if up else 'down'}), the {product} material cost "
    f"is expected to move {pct(scen[1] - 1)} in {h} months, within {pct(scen[0] - 1)} to {pct(scen[2] - 1)} at "
    f"{level:.0%} confidence. At a fixed price on today's cost, the margin would be {spot.at['central', 'margin']:.1%} "
    f"in the central case and {spot.at['adverse', 'margin']:.1%} in the adverse case, against a floor of "
    f"{floor:.0%}. {rec_txt}")

# BOM fan
rows = [{"h": 0, "lo": 100, "mid": 100, "hi": 100, "case": c} for c in ("Our forecast", "Scenario")]
for hh in I.HORIZONS:
    for case, v in (("Our forecast", bom(hh, False)), ("Scenario", bom(hh))):
        rows.append({"h": hh, "lo": 100 * v[0], "mid": 100 * v[1], "hi": 100 * v[2], "case": case})
fan = pd.DataFrame(rows)
colors = alt.Scale(domain=["Our forecast", "Scenario"], range=["#8c8c8c", "#d9480f"])
x = alt.X("h:Q", title="Months after the quote", scale=alt.Scale(domain=[0, 9]), axis=alt.Axis(values=[0, 3, 6, 9]))
chart = (alt.Chart(fan).mark_area(opacity=0.18).encode(x=x, y=alt.Y("lo:Q", title="BOM material cost (2026-09 = 100)",
                                                                       scale=alt.Scale(zero=False)),
                                                      y2="hi:Q", color=alt.Color("case:N", scale=colors, title=None))
         + alt.Chart(fan).mark_line(point=True).encode(x=x, y="mid:Q", color=alt.Color("case:N", scale=colors)))
left, right = st.columns([3, 2])
left.subheader("Material cost of one box")
left.altair_chart(chart, width="stretch")
if show_eur:
    eur0 = float(w["value"].sum())
    left.caption(f"Priced BOM materials at the origin: {eur0:,.0f} EUR per box. Scenario at {h} months: "
                 f"{eur0 * scen[0]:,.0f} / {eur0 * scen[1]:,.0f} / {eur0 * scen[2]:,.0f} EUR (low / central / high).")

# margin by policy
mt = tab.pivot_table(index="policy", columns="point", values="margin").reindex(list(POLICY_NAMES))
mk = tab[tab.point == "central"].set_index("policy")["markup"].reindex(mt.index)
view = pd.DataFrame({"Quote rule": [POLICY_NAMES[i] for i in mt.index], "Price vs today": mk.map(pct).to_numpy(),
                     "Margin, favourable": mt["favourable"].map("{:.1%}".format).to_numpy(),
                     "Margin, central": mt["central"].map("{:.1%}".format).to_numpy(),
                     "Margin, adverse": mt["adverse"].map("{:.1%}".format).to_numpy(),
                     "Adverse below floor": np.where(mt["adverse"] < floor, "yes", "no")})
right.subheader("Margin by quote rule")
right.dataframe(view.style.apply(lambda r: ["font-weight: bold; background-color: rgba(217,72,15,0.12)"] * len(r)
                                 if rec and r["Quote rule"] == POLICY_NAMES.get(rec[0]) else [""] * len(r), axis=1),
                hide_index=True, width="stretch")
right.caption("Copper forward priced from the LME 3-month spread at the origin; EUR/USD held flat. Margins in the "
              "favourable / central / adverse cases use the low / central / high end of the range.")

# per material
st.subheader(f"Prices at {h} months, per material and energy")
mrows = []
for s in series:
    if (s, h) not in fc.index:
        continue
    b0, b1 = band(s, h, False), band(s, h)
    mv = moves[h].loc[s]
    mrows.append({"Series": names.get(s, s), "Weight in BOM": f"{w['weight'].get(s, 0):.0%}" if s in w.index else "energy",
                  "Our forecast (median)": pct(b0[1] - 1), "Scenario (median)": pct(b1[1] - 1),
                  f"Scenario {level:.0%} range": f"{pct(b1[0] - 1)} to {pct(b1[2] - 1)}",
                  "Shock applied": pct(np.exp(mv["move"]) - 1),
                  "How": {"driver": "shocked", "held": "held", "own": "own shock"}.get(
                      mv["sample"], f"beta {mv['beta']:.2f}, lag {int(mv['lag'])} m, {mv['sample']} sample"),
                  "Forecast source": fc.loc[(s, h), "source"]})
st.dataframe(pd.DataFrame(mrows), hide_index=True, width="stretch")

pick = st.selectbox("Price path of one series", [s for s in series if (s, h) in fc.index], format_func=lambda s: names.get(s, s))
hist = inp.panel[pick].loc[:I.ORIGIN].dropna().iloc[-48:]
last = float(fc.loc[(pick, h), "last"])
hrows = [{"date": d_, "value": 100 * v / last, "kind": "History"} for d_, v in hist.items()]
frows = [{"date": I.ORIGIN, "lo": 100, "mid": 100, "hi": 100}]
for hh in I.HORIZONS:
    lo, mid, hi = band(pick, hh)
    frows.append({"date": I.ORIGIN + pd.DateOffset(months=hh), "lo": 100 * lo, "mid": 100 * mid, "hi": 100 * hi})
fr = pd.DataFrame(frows)
yt = "Price (2026-09 = 100)" if not show_eur else "Price (2026-09 = 100; level shown in the table below)"
pc = (alt.Chart(pd.DataFrame(hrows)).mark_line(color="#495057").encode(x=alt.X("date:T", title=None),
                                                                        y=alt.Y("value:Q", title=yt, scale=alt.Scale(zero=False)))
      + alt.Chart(fr).mark_area(opacity=0.2, color="#d9480f").encode(x="date:T", y="lo:Q", y2="hi:Q")
      + alt.Chart(fr).mark_line(point=True, color="#d9480f").encode(x="date:T", y="mid:Q"))
st.altair_chart(pc, width="stretch")
if show_eur:
    st.caption(f"Last observed level: {last:,.2f} ({names.get(pick, pick)}).")

with st.expander("How the scenario is built"):
    st.markdown(
        "- **Forecast.** The median and range come from the submitted model's graded run (origin 2026-09). Series "
        "without a model forecast (energy, some BOM components) use a volatility-scaled range around today's price.\n"
        "- **Confidence level.** The P10-P90 range is widened or narrowed in proportion to the normal quantiles.\n"
        "- **Shock size.** One sigma is the standard deviation of the series' historical moves over the same horizon "
        "(data up to 2026-09). The crash preset uses the worst move actually seen in the history.\n"
        "- **Co-movement.** Other prices follow the shocked one with their historical sensitivity (beta). For the "
        "crash preset the sensitivity is estimated on stress periods only (2008-09, 2020, 2021-22), because prices "
        "move together more in crises. Power prices start in 2016, so their stress sample is 2020-22.\n"
        "- **Energy.** Energy reaches material prices with the lag (0-6 months) that fits history best. Within the "
        "horizon only the part of the move that has passed through counts. The direct plant energy line is an "
        "assumption and is 0 by default.\n"
        "- **Margin.** Same calculation as the margin-protection replay: a fixed price set at the quote, materials "
        "bought at the horizon, index clauses and the copper forward as in the replay. The recommended rule is the "
        "one chosen on 2010-21 data with the pre-registered rule.")
    st.dataframe(moves[h].assign(series=[names.get(s, s) for s in moves[h].index]).set_index("series").round(3),
                 width="stretch")
