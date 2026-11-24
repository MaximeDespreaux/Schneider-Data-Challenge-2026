"""Build the PUBLIC (NDA-safe) versions of the jury reports (HTML + PDF).

    uv run python public/reports/make_public_figures.py              # figures first
    uv run python public/reports/build_public_reports.py             # all reports
    uv run python public/reports/build_public_reports.py 01_eda_report

Derived from reports/build_reports.py (private). Rules: no CRU/MEPS/epoxy EUR/t levels, no Schneider € figures,
no BOM masses, no supplier names or financials, no part-mapping counts, no personal data. Indices, % changes,
correlations and error metrics are kept. Check the output with `uv run python public/scan_leaks.py`.

Report 0 (data audit) reuses the inline figures of notebooks/00_data_profiling_preprocessing.ipynb
and computes its tables from data/processed/*.parquet.
Report 1 (EDA) quotes the verified outputs of notebooks/01_data_overview_eda.ipynb.
Report 2 (forecasting) computes its tables from data/processed/backtest.parquet
(produced by notebooks/02_forecasting_benchmark.ipynb), so the numbers cannot drift.
Report 3 (feature study) computes its tables from data/processed/feature_selection.parquet
(produced by notebooks/03_feature_study.ipynb) and reuses its saved figures 18–23.
PDFs are rendered with headless Google Chrome.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.backtest as bt  # noqa: E402

REPORTS = ROOT / "public" / "reports"
FIGS = REPORTS / "figures"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# --------------------------------------------------------------------------- styling
CSS = """
/* Layout: one reading column (~68ch) on a steel-tinted page; figures sit on light "plates"
   because the charts are rendered on a light surface; copper accent = the subject's own metal. */
:root {
  --bg: #f4f5f6; --surface: #ffffff; --ink: #15181c; --muted: #59626d; --rule: #d7dbe0;
  --accent: #a84f25; --accent-soft: #f4e4da; --good: #1d6b3a; --warn: #9a5b00; --plate: #fcfcfb;
  --font-display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --font-body: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --font-data: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111316; --surface: #191c20; --ink: #e7e9ec; --muted: #9ba4ae; --rule: #2b3036;
  --accent: #e0915f; --accent-soft: #3a2417; --good: #6cc68c; --warn: #e3a948; --plate: #fcfcfb; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #111316; --surface: #191c20; --ink: #e7e9ec; --muted: #9ba4ae; --rule: #2b3036;
  --accent: #e0915f; --accent-soft: #3a2417; --good: #6cc68c; --warn: #e3a948; --plate: #fcfcfb; color-scheme: dark }

* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: var(--font-body); font-size: 15.5px; line-height: 1.6;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  padding-block: 32px 64px; }
main { max-width: 820px; margin: 0 auto; display: grid; gap: 40px; }
section { display: grid; gap: 14px; min-width: 0; }
h1, h2, h3 { font-family: var(--font-display); text-wrap: balance; line-height: 1.15; margin: 0; }
h1 { font-size: clamp(30px, 5vw, 42px); font-weight: 800; letter-spacing: -0.01em; }
h2 { font-size: 23px; font-weight: 700; padding-top: 12px; border-top: 2px solid var(--ink); }
h3 { font-size: 17px; font-weight: 700; margin-top: 6px; }
p { margin: 0; max-width: 68ch; }
ul, ol { margin: 0; padding-left: 1.2em; display: grid; gap: 6px; max-width: 70ch; }
a { color: var(--accent); }
.eyebrow { font-family: var(--font-data); font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); }
.cover { display: grid; gap: 14px; padding-bottom: 22px; border-bottom: 1px solid var(--rule); }
.lede { font-size: 18px; color: var(--muted); max-width: 62ch; }
.meta { display: flex; flex-wrap: wrap; gap: 8px 28px; font-family: var(--font-data); font-size: 12.5px; color: var(--muted); }
.meta b { color: var(--ink); font-weight: 500; }
.keys { display: grid; gap: 12px; list-style: none; padding: 0; counter-reset: k; }
.keys li { display: grid; grid-template-columns: 2.2em 1fr; gap: 0 10px; align-items: baseline; }
.keys li::before { counter-increment: k; content: counter(k); font-family: var(--font-data); font-size: 13px; color: var(--accent);
  border: 1px solid var(--accent); border-radius: 50%; width: 1.9em; height: 1.9em; display: inline-grid; place-items: center; }
.keys b { font-weight: 600; }
.asks { background: var(--accent-soft); padding: 16px 20px; border-radius: 4px; display: grid; gap: 8px; }
.asks h3 { margin: 0; color: var(--accent); font-size: 14px; font-family: var(--font-data); text-transform: uppercase; letter-spacing: 0.07em; font-weight: 500; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.stat { background: var(--surface); border: 1px solid var(--rule); padding: 14px 16px; display: grid; gap: 4px; align-content: start; }
.stat .v { font-family: var(--font-display); font-size: 28px; font-weight: 800; font-variant-numeric: tabular-nums; }
.stat .l { font-size: 13px; color: var(--muted); line-height: 1.4; }
figure { margin: 0; background: var(--plate); border: 1px solid var(--rule); padding: 10px; display: grid; gap: 8px; break-inside: avoid; }
figure img { width: 100%; height: auto; display: block; }
figcaption { font-size: 13px; color: #59626d; line-height: 1.45; }
figcaption b { color: #15181c; }
.tbl { overflow-x: auto; break-inside: avoid; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-family: var(--font-data); font-weight: 500; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); border-bottom: 1.5px solid var(--ink); }
td.n, th.n { text-align: right; font-family: var(--font-data); font-size: 13px; }
td.win { color: var(--good); font-weight: 600; }
td.lose { color: var(--warn); }
code, .mono { font-family: var(--font-data); font-size: 0.88em; }
.note { font-size: 13.5px; color: var(--muted); border-left: 2px solid var(--rule); padding-left: 12px; max-width: 70ch; }
.tag { font-family: var(--font-data); font-size: 11px; padding: 1px 7px; border-radius: 10px; border: 1px solid currentColor; white-space: nowrap; }
.tag.ok { color: var(--good); } .tag.warn { color: var(--warn); } .tag.bad { color: var(--accent); }
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
.two > * { min-width: 0; }
footer { font-size: 12.5px; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 14px; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

@page { size: A4; margin: 14mm 15mm 16mm; }
@media print {
  :root, :root[data-theme="dark"] { --bg: #ffffff; --surface: #ffffff; --ink: #15181c; --muted: #59626d; --rule: #d7dbe0;
    --accent: #a84f25; --accent-soft: #f4e4da; --good: #1d6b3a; --warn: #9a5b00; color-scheme: light; }
  body { padding: 0; font-size: 10.5pt; }
  /* Chrome cannot fragment CSS grid across pages (items vanish or overlap): use block flow in print */
  main, section, ul, ol, figure, .asks { display: block; }
  main { max-width: none; }
  main > * + * { margin-top: 24px; }
  section > * + *, .asks > * + * { margin-top: 10px; }
  li + li { margin-top: 5px; }
  .keys li + li { margin-top: 9px; }
  figure img { max-height: 118mm; object-fit: contain; }
  figcaption { margin-top: 6px; }
  h2, h3 { break-after: avoid; }
  section.pb { break-before: page; }
  .stat, .asks, tr, li, figure, .tbl { break-inside: avoid; }
}
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@700;800&family=IBM+Plex+Mono:wght@400;500'
         '&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,400&display=swap">')


def page(title: str, body: str) -> str:
    return (f'<meta charset="utf-8"><title>{title}</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n{FONTS}\n'
            f"<style>{CSS}</style>\n<main>\n{body}\n</main>\n")


def fig(name: str, caption: str) -> str:
    data = base64.b64encode((FIGS / f"{name}.png").read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{caption.split("</b>")[0].replace("<b>", "")}"><figcaption>{caption}</figcaption></figure>'


def table(df: pd.DataFrame, num_cols=(), fmt="{:.2f}", rel_cols=()) -> str:
    head = "".join(f'<th class="{"n" if c in num_cols else ""}">{c}</th>' for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            cls = "n" if c in num_cols else ""
            if c in rel_cols and isinstance(v, (int, float)) and not pd.isna(v):
                cls += " win" if v < 0.97 else (" lose" if v > 1.03 else "")
            txt = (fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
                   else ("—" if (isinstance(v, float) and pd.isna(v)) else str(v)))
            tds.append(f'<td class="{cls.strip()}">{txt}</td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


# =========================================================================== REPORT 1
def report_eda() -> str:
    sources = pd.DataFrame([
        ["Core challenge panel (data.zip)", "15 RM + 30 energy series, monthly", "Nov-2016 → Oct-2026", '<span class="tag ok">primary</span>'],
        ["MEPS & CRU steel indexes", "CR, HDG, HR coil by country; 304/430/316 stainless; mapping of Schneider steel parts to their contract index (1-month lead time)", "2012 → Oct-2026 (stainless 2008 →)", '<span class="tag ok">high value</span>'],
        ["LME copper & aluminium", "Cash and 3-month prices, spreads, stocks, FX", "Jan-2016 → Nov-2026 (partial)", '<span class="tag ok">useful</span>'],
        ["Epoxy cost build-up", "Benzene, acetone, epichlorohydrin, resin (EUR/t)", "Jan-2018 → Oct-2026", '<span class="tag ok">useful</span>'],
        ["Competitor teardown benchmark", "Should-cost of RM6 vs 5 competitors", "Snapshot, 09-Nov-2026", '<span class="tag warn">provisional</span>'],
        ["Supplier financials (PitchBook)", "9 suppliers: P&amp;L, balance sheet, ratios", "2005 → 2025, uneven", '<span class="tag warn">patchy</span>'],
        ["BOM workbook at data/ root", "BOM only; masses differ from data.zip", "—", '<span class="tag bad">conflict</span>'],
        ["RMU IQI EU price benchmark", "Yearly P10 / P50 / P90 selling prices for 6 brands (synthetic); unlocked copy received 18 Nov", "2017 → 2026", '<span class="tag warn">synthetic, yearly</span>'],
        ["Kick-off presentation", "Business context, evaluation criteria, submission format", "Nov-2026", '<span class="tag ok">context</span>'],
    ], columns=["Source", "What it contains", "Coverage", "Status"])

    quality = pd.DataFrame([
        ["Publication lags", "Series end between Jul-2026 and Oct-2026; no gaps inside the history", "Treat the last 1–3 months as a nowcast, not as missing data"],
        ["Two BOM versions", "AirSeT masses differ between data.zip and the root file (more carbon steel, less stainless in the root file)", "This report used the data.zip version. The organisers have since sent a latest BOM (v1), now the reference; the cost-index chart below uses it"],
        ["Epoxy typo", "Epichlorohydrin in May 2026 is about ten times below the neighbouring months; the resin price that month is computed from it", "May 2026 excluded from the epoxy fit and correlations"],
        ["RM6 teardown detail", "Line items add up to more than twice the datasheet mass and give far less copper than the BOM", "Teardown totals used for comparison only; line items not used"],
        ["RM6 carbon-steel mass", "Labelled “to confirm”, no mass given", "Assumed as a share of the implied total mass; the latest BOM (v1) now gives it"],
        ["SGC should-cost", "Total two orders of magnitude below every other unit", "Excluded from comparisons"],
        ["PitchBook leverage", "“Debt to Equity” is 0 for every supplier", "Recomputed from total debt and equity"],
        ["Cross-source check", "Challenge copper/aluminium in EUR vs the LME workbook: mean gap 0.06%, max 1.1%", "Sources agree; no correction needed"],
    ], columns=["Issue", "What we found", "How we handled it"])

    vol = pd.DataFrame([
        ["Iron ore (S04)", 32.1, -6.0, "Jun-2021"], ["Nickel (S03)", 25.2, 8.2, "Mar-2022"], ["Zinc (S05)", 20.2, 27.6, "Apr-2022"],
        ["Aluminium, EUR (D02)", 16.0, 18.7, "Mar-2022"], ["Copper, EUR (D01)", 14.4, 36.1, "Sep-2026"],
        ["EU steel PPI (S09)", 7.4, 9.6, "May-2022"], ["EU plastics PPI (S11)", 6.9, 14.2, "Jun-2022"],
    ], columns=["Material", "Volatility % / yr", "Change YoY %", "Peak month"])

    leads = pd.DataFrame([
        ["CRU CR coil (DE)", "EU steel PPI S09", "Market leads by 1 month", "0.71"],
        ["CRU HDG coil (DE)", "EU galvanised PPI S10", "Market leads by 1 month", "0.62"],
        ["LME nickel", "Stainless 304 alloy surcharge", "Nickel leads by 1 month", "0.66"],
        ["Brent crude (EUR)", "EU plastics PPI S11", "Oil leads by 1 month", "0.60"],
        ["Iron ore", "CRU CR coil (DE)", "Ore leads by 1 month", "0.43"],
        ["EU energy PPI", "EU aluminium PPI S08", "Same month", "0.44"],
    ], columns=["Leading series", "Follows", "Timing", "Correlation of monthly changes"])

    suppliers = supplier_table()

    body = f"""
<header class="cover">
  <span class="eyebrow">Schneider Electric × X-HEC Data Challenge 2026 · Report 1 of 2</span>
  <h1>RMU Raw-Material Data Review</h1>
  <p class="lede">What the data says about the materials inside the RM AirSeT and RM6 NE-IQI ring main units, what is reliable, and what needs fixing before any forecast is trusted.</p>
  <div class="meta"><span>Data received <b>12–18 Nov 2026</b></span><span>Price history <b>Nov 2016 → Oct 2026</b></span><span>Report <b>18 Nov 2026</b></span></div>
</header>

<section>
  <h2>Summary</h2>
  <ol class="keys">
    <li><span><b>The data is solid enough to work with.</b> The core panel has 10 years of monthly prices with no gaps inside the history. The only missing values are the last one to three months, which have not been published yet.</span></li>
    <li><span><b>The steel prices the organisers left out are in the steel index workbook.</b> It holds the real CRU and MEPS market prices for cold-rolled, galvanised and stainless steel. The free Eurostat index trails these prices by about a month and swings 2.5× less.</span></li>
    <li><span><b>Copper is at an all-time high:</b> €12,571 per tonne in September 2026, up 36% in a year. Copper now makes up about 60% of the RM6's priced material cost. In the data.zip BOM the AirSeT has no copper and its exposure is carbon and stainless steel in equal parts; the latest BOM (v1) adds a small copper share.</span></li>
    <li><span><b>Some prices move before others.</b> Market steel prices lead the Eurostat indices, oil leads plastics, and nickel leads the stainless surcharge, each by about one month. These links are what make part of the forecasting work.</span></li>
    <li><span><b>Several inputs conflict or are incomplete.</b> There are two versions of the bill of materials, the RM6 carbon-steel mass is missing, and one competitor's cost is broken.</span></li>
    <li><span><b>Selling prices follow regulation, not material cost.</b> In the synthetic price benchmark, SF6 units rise 32–40% in 2026 while the only SF6-free unit (Eaton) rises 8%. Priced materials are only 6–9% of the RM6 selling price, so a 20% copper move is under 1% of price.</span></li>
  </ol>
  <div class="asks">
    <h3>Decisions we need</h3>
    <ul>
      <li>Confirm which bill of materials is correct, and give the RM6 carbon-steel mass. <i>(Answered since: the latest BOM v1 is the reference.)</i></li>
      <li>Share the two documents cited in the kick-off deck, “RM AirSeT and RM6 Product and Cost Overview” and “RM AirSeT vs RMX Product Cost Overview”: they should hold the AirSeT cost gap and margins.</li>
      <li>Confirm whether the migration question is about RM6 → RM AirSeT (as in the data) or SM6 → SM AirSeT (as on two slides).</li>
      <li>Confirm that the licensed CRU and MEPS prices may be used for modelling. We keep them out of anything shared outside the challenge.</li>
    </ul>
  </div>
</section>

<section>
  <h2>The data at a glance</h2>
  <p>We received nine sources, including the kick-off presentation. Two of them carry most of the value: the organisers' price panel and the steel index workbook from the purchasing team.</p>
  {table(sources)}
</section>

<section>
  <h2>Data quality and how we handled it</h2>
  {table(quality)}
  {fig("01_missingness", "<b>Observed months by series.</b> Blue is observed, grey is missing. The empty rows are the eight licensed series the organisers left blank; every other gap is at the recent edge.")}
</section>

<section class="pb">
  <h2>Ten years of prices</h2>
  <p>Almost every material roughly doubled during the 2021–22 commodity and energy shock and then fell back part of the way. Copper has since moved to new highs, and zinc and EU copper products are rising again.</p>
  {fig("02_rm_indexed", "<b>Prices rebased to November 2016 = 100.</b> Shaded band: the 2021–22 shock.")}
  <div class="two">
    {table(vol, num_cols=("Volatility % / yr", "Change YoY %"), fmt="{:.1f}")}
    <p class="note">Exchange-traded metals swing two to four times more than the Eurostat producer price indices. The indices move slowly because they include contract and list prices, which change less often than spot quotes.</p>
  </div>
  {fig("03_rolling_vol", "<b>Rolling 12-month volatility.</b> Volatility spiked in 2020–22 for every metal and has since fallen, except aluminium, which is rising again.")}
</section>

<section>
  <h2>What moves before what</h2>
  <p>Monthly price changes in one series are correlated with later changes in another. These lags are the useful signal for a 3- to 9-month forecast.</p>
  {table(leads, num_cols=("Correlation of monthly changes",))}
  {fig("07_steel_market_vs_proxy", "<b>Steel market prices (solid) against the Eurostat indices (dashed), year-on-year % change.</b> In 2021 the market rose about 150–160% on a year earlier against about 60% for the Eurostat indices, then fell faster (about −36% vs −20%). The market moves are about twice as large.")}
  <p class="note">Every price series is a random walk with trend in levels and stationary in monthly changes, and none shows meaningful seasonality. Forecasting should therefore work on monthly changes and can skip seasonal terms.</p>
</section>

<section class="pb">
  <h2>How exposed are the two products?</h2>
  <p>We valued each bill of materials at market prices month by month (cost-weighted: price × mass). Plastics have no euro-per-tonne price, so we used an assumed price in 2021 moved with the plastics index. This covers the priced materials only, not the full product cost. Product costs are confidential and are shown here as changes and indices only.</p>
  <div class="stats">
    <div class="stat"><span class="v">×2.0</span><span class="l">RM6 priced material cost today vs June 2020; 12% below its April 2022 peak (data.zip BOM)</span></div>
    <div class="stat"><span class="v">+56%</span><span class="l">AirSeT priced material cost today vs June 2020; 41% below its 2022 peak (data.zip BOM)</span></div>
    <div class="stat"><span class="v">60%</span><span class="l">Copper's share of RM6 priced cost, and 40% of its month-to-month swings</span></div>
    <div class="stat"><span class="v">49 / 49%</span><span class="l">Share of AirSeT cost swings from carbon steel and from stainless steel</span></div>
  </div>
  {fig("08_bom_cost_index", "<b>Left: priced material cost index per product</b>, cost-weighted with the latest BOM (v1), latest month = 100. <b>Right: share of priced cost by material.</b> Epoxy history starts in 2018. “Other” and unidentified material are excluded (not priceable); brass is priced as 65% copper + 35% zinc.")}
</section>

<section>
  <h2>Competitive and supplier context</h2>
  <h3>Should-cost against competitors</h3>
  <p>A teardown estimate compares the should-cost of the Schneider RM6 with five competitors. Materials are 80–86% of the estimate for every valid unit, so raw-material prices largely decide cost competitiveness. The RM6 is the lightest unit. The teardown's own line items are unreliable (their masses add up to more than twice the datasheet mass and give far less copper than the bill of materials), so we use only the totals. One competitor's total (SGC) is a calculation error and is excluded; Eaton's low figure reflects an SF6-free design and should be treated with caution. Should-cost values are confidential and not shown.</p>
  <h3>Selling prices</h3>
  <p>The price benchmark gives a synthetic yearly selling price per brand for a 3-function RMU in the EU. Until 2025 prices drift up slowly (Schneider +21% since 2017, against +31% for its material cost). In 2026 every SF6 unit jumps 32–40%, Schneider by 40%, while Eaton's SF6-free unit rises only 8%. The jump starts at the workbook's own “SF6 marker” and lines up with the EU ban on SF6 in new switchgear up to 24 kV from 2026. Schneider's RM6 is now priced above Eaton's SF6-free unit: the price gap that held SF6-free products back has closed.</p>
  <div class="stats">
    <div class="stat"><span class="v">6.6%</span><span class="l">RM6 priced material as a share of the 2026 median price; 6–9% in every year since 2017</span></div>
    <div class="stat"><span class="v">25%</span><span class="l">Teardown should-cost as a share of the 2026 median price</span></div>
    <div class="stat"><span class="v">0.8%</span><span class="l">Effect of a 20% copper move on the RM6 price</span></div>
    <div class="stat"><span class="v">+40% / +8%</span><span class="l">2026 price change: Schneider (SF6) vs Eaton (SF6-free)</span></div>
  </div>
  {fig("16_price_vs_cost_index", "<b>Schneider selling price (synthetic P50) vs RM6 priced material cost</b> (latest BOM), yearly means, first common year = 100. The 2021–22 cost spike barely moved prices; the 2026 jump comes with the SF6 regulation.")}
  <h3>Supplier financial health</h3>
  {table(suppliers)}
  <p class="note">Suppliers are anonymised and ordered by revenue (A = largest); names and financial figures are confidential. Flags are computed from the PitchBook statements, with debt/equity recomputed from the balance sheet (PitchBook's own ratio is unreliable). A debt of zero may mean “not reported”.</p>
  <h3>Epoxy resin</h3>
  <p>Epoxy prices follow their feedstocks almost exactly: a linear recipe of benzene, acetone, epichlorohydrin and other costs explains 97.7% of the price (R² = 0.977). The May 2026 epichlorohydrin value (about ten times too low) is a data-entry error and is excluded; left in, it produces a spurious 0.87 correlation between epichlorohydrin and resin. Resin moves with its inputs in the same month (correlation of monthly changes: benzene 0.66, other costs 0.60, acetone 0.59, epichlorohydrin 0.38), and no input leads it by one month or more (at most 0.13). Resin behaves like a same-month cost formula, so to forecast epoxy, forecast the feedstocks.</p>
</section>

<section class="pb">
  <h2>Appendix</h2>
  <h3>Methods</h3>
  <ul>
    <li><b>Volatility</b> is the annualised standard deviation of monthly log changes, Nov-2016 to Sep-2026.</li>
    <li><b>Lead–lag</b> is the correlation between the monthly log change of the leading series at month t−k and the follower at month t, for k = 0…9 (steel pairs tested for k = −6…6).</li>
    <li><b>Stationarity</b>: augmented Dickey–Fuller tests on log levels (no rejection) and log changes (rejection). <b>Seasonality</b>: robust STL decomposition; seasonal strength is at most 0.16 for every series.</li>
    <li><b>Cost exposure</b>: the share of each material in the variance of month-to-month changes in priced cost.</li>
  </ul>
  <h3>Assumptions</h3>
  <ul>
    <li>Stats in this report: bill of materials from <span class="mono">data.zip</span>, with the RM6 carbon-steel mass assumed. The cost-index chart uses the latest BOM (v1), now the reference.</li>
    <li>Material prices: carbon steel = CRU CR coil Germany; galvanised = CRU HDG Germany; stainless = CRU 304 Germany (base + alloy surcharge); copper and aluminium = LME in EUR; epoxy = epoxy resin series; brass = 65% copper + 35% zinc; plastics = an assumed 2021 price indexed on the EU plastics PPI.</li>
  </ul>
  <h3>Reproducing</h3>
  <p>Everything in this report comes from <span class="mono">public/notebooks/01_data_overview_eda.ipynb</span> (run locally with the NDA data). The file-by-file notes are in <span class="mono">public/DATA_OVERVIEW.md</span>.</p>
</section>
<footer>Public version (NDA-safe). Prepared for the Schneider Electric × X-HEC Data Challenge 2026 jury. Licensed CRU / MEPS / epoxy prices and Schneider-internal figures are shown only as indices, % changes or statistics.</footer>
"""
    return page("RMU Raw-Material Data Review", body)


# =========================================================================== REPORT 2
def report_forecast() -> str:
    res = pd.read_parquet(ROOT / "data" / "processed" / "backtest.parquet")
    sc = bt.score(res, by=("model", "h"))
    names = {"ensemble": "Average of five models", "arima": "ARIMA (momentum)", "naive": "Naive: last price",
             "dynreg": "Driver regression", "lgbm": "LightGBM (machine learning)", "theta": "Theta",
             "drift": "Naive + trend", "ets": "Exponential smoothing"}
    board = sc.relMAE_vs_naive.unstack("h").loc[list(names)]
    board.index = [names[m] for m in board.index]
    board = board.reset_index().rename(columns={"index": "Model", 3: "3 months", 6: "6 months", 9: "9 months"})

    per = bt.score(res, by=("series", "h", "model")).relMAE_vs_naive.unstack("model")
    label = {"S09": "EU steel index (S09)", "S10": "EU galvanised index (S10)", "S11": "EU plastics index (S11)",
             "CRU_SS304_DE": "Stainless 304, Germany", "CRU_CR_DE": "CR coil, Germany", "CRU_HDG_DE": "HDG coil, Germany",
             "D01": "Copper (EUR)", "D02": "Aluminium (EUR)", "S03": "Nickel", "S04": "Iron ore"}
    best = []
    for s, lab in label.items():
        row = [lab]
        for h in (3, 9):
            r = per.loc[(s, h)].drop(["naive", "drift"])
            row += [names[r.idxmin()], r.min()]
        best.append(row)
    best = pd.DataFrame(best, columns=["Price", "Best model, 3 months", "Error vs naive, 3m", "Best model, 9 months", "Error vs naive, 9m"])

    cov = bt.score(res[res.model == "ensemble"], by=("h",)).cover_80
    win = bt.score(res.assign(window=bt.window_label(res))[lambda d: d.model.isin(["naive", "ensemble"])],
                   by=("window", "model", "h")).relMAE_vs_naive.unstack("h").xs("ensemble", level="model")

    def hedge(series, model="arima", thr=0.02, carry=0.004, h=3):
        g = res[(res.series == series) & (res.model == model) & (res.h == h)]
        fwd, spot = g.y_origin * (1 + carry) ** h, g.y_true
        rule = np.where(g.pred > thr, fwd, spot)
        s_rule = 100 * (1 - rule.mean() / spot.mean())
        s_or = 100 * (1 - np.minimum(fwd, spot).mean() / spot.mean())
        return s_rule, s_or
    hed = []
    for s in ["S09", "S10", "CRU_CR_DE", "CRU_SS304_DE", "D01"]:
        r, o = hedge(s)
        hed.append([label[s], r, o, 100 * r / o])
    hed = pd.DataFrame(hed, columns=["Purchase priced on", "Saving with forecast rule %", "Saving with perfect hindsight %", "Share captured %"])

    s09 = per.loc[("S09", 3), "dynreg"]; s10 = per.loc[("S10", 3), "dynreg"]
    s09_9 = per.loc[("S09", 9), "dynreg"]; s10_9 = per.loc[("S10", 9), "dynreg"]
    ens3 = sc.loc[("ensemble", 3), "relMAE_vs_naive"]
    lo3, hi3 = sorted([100 * (1 - s09), 100 * (1 - s10)]); lo9, hi9 = sorted([100 * (1 - s09_9), 100 * (1 - s10_9)])
    cap = hed.set_index("Purchase priced on")["Share captured %"]
    idx_cap = cap.iloc[:2].mean(); mkt_lo, mkt_hi = cap.iloc[2:].min(), cap.iloc[2:].max()
    cr_cap = cap.loc[label["CRU_CR_DE"]]

    bom = pd.DataFrame([
        ["RM6 NE-IQI", "+3%", "+17%", "+4%", "+54%"],
        ["RM AirSeT", "+2%", "+18%", "+3%", "+58%"],
    ], columns=["Product", "3 months, central", "3 months, bad case (P90)", "9 months, central", "9 months, bad case (P90)"])

    body = f"""
<header class="cover">
  <span class="eyebrow">Schneider Electric × X-HEC Data Challenge 2026 · Report 2 of 2</span>
  <h1>RMU Price Forecasting Benchmark</h1>
  <p class="lede">We tested eight forecasting approaches on twelve raw-material prices at 3, 6 and 9 months ahead, against the simplest possible forecast: next price equals today's price. This report says where forecasting works, where it doesn't, and what that means for buying decisions.</p>
  <div class="meta"><span>Backtest <b>Jan 2020 → Jul 2026</b></span><span>Forecasts scored <b>{len(res):,}</b></span><span>Report <b>18 Nov 2026</b></span></div>
</header>

<section>
  <h2>Summary</h2>
  <ol class="keys">
    <li><span><b>Market prices cannot be predicted much better than “same as today”.</b> For steel coil, copper, aluminium, nickel and iron ore, the best models cut error by at most about 5%, and only at 3 months. The futures market does no better.</span></li>
    <li><span><b>The Eurostat price indices can be predicted.</b> These indices are common in index-linked supply contracts. Because they follow market prices with a delay, a regression on recent market moves cuts forecast error by {lo3:.0f}–{hi3:.0f}% at 3 months and {lo9:.0f}–{hi9:.0f}% at 9 months.</span></li>
    <li><span><b>The range matters more than the central forecast.</b> Our 80% ranges hold {100*cov[3]:.0f}% of outcomes at 3 months. In the bad case (90th percentile), priced material cost rises 54% for the RM6 and 58% for the AirSeT within 9 months, against a central forecast of only +3–4%.</span></li>
    <li><span><b>Forecasts earn money on index-linked purchases.</b> A simple rule that buys three months ahead when a rise is forecast captures about {idx_cap:.0f}% of the saving perfect hindsight would give on index-priced steel, against {mkt_lo:.0f}–{mkt_hi:.0f}% on market-priced steel and copper. Schneider's steel contracts are indexed on CRU market prices today, so the {cr_cap:.0f}% figure is the one that applies now; the higher figure needs Eurostat-indexed contracts.</span></li>
  </ol>
  <div class="asks">
    <h3>What we recommend</h3>
    <ul>
      <li>For market-priced materials, plan on today's price and manage the risk range: hedge or forward-buy copper for the RM6 and steel for the AirSeT when the bad case exceeds the budget.</li>
      <li>For contracts indexed on Eurostat producer indices, use the driver regression to forecast the index and time purchases.</li>
      <li>Consider moving steel contracts from CRU to Eurostat producer indices where the supplier allows it: they are smoother, lag the market, and can be forecast. Today the indexed steel references in the mapping follow CRU with a 1-month lead time.</li>
    </ul>
  </div>
</section>

<section>
  <h2>How we tested</h2>
  <div class="two">
    <div style="display:grid;gap:10px">
      <p>Each month from January 2020, we fitted every model on the data available at that time and forecast 3, 6 and 9 months ahead. We then compared each forecast with what actually happened. This covers the 2021–22 shock and the calmer years after it.</p>
      <p>Each input is delayed by its real publication lag (for example two months for Eurostat indices), so no model sees data that was not yet published. An automated check confirms this.</p>
    </div>
    <div style="display:grid;gap:10px">
      <p><b>The score</b> is each model's average error divided by the error of the naive forecast (next price = today's price). Below 1.00 the model is better; above 1.00 it is worse.</p>
      <p><b>The models</b>, from simple to complex: trend and momentum models (ARIMA, exponential smoothing, Theta); a driver regression using leading prices; a machine-learning model (LightGBM) trained on all series together; and an average of five of them.</p>
    </div>
  </div>
</section>

<section class="pb">
  <h2>Results</h2>
  <h3>Average over the twelve prices</h3>
  {table(board, num_cols=("3 months", "6 months", "9 months"), rel_cols=("3 months", "6 months", "9 months"))}
  <p class="note">Green: at least 3% better than naive. Amber: at least 3% worse. The average of five models is the best overall at 3 months ({ens3:.2f}), but no approach beats naive on average at 6 or 9 months.</p>
  <h3>Price by price</h3>
  {table(best, num_cols=("Error vs naive, 3m", "Error vs naive, 9m"), rel_cols=("Error vs naive, 3m", "Error vs naive, 9m"))}
  {fig("12_leaderboard_heatmap", "<b>Error relative to naive for every price and model.</b> Blue cells beat naive. The blue block at the top is the Eurostat indices; the market prices below stay close to 1 or turn red at longer horizons.")}
</section>

<section>
  <h3>Checks on these results</h3>
  <ul>
    <li><b>Timing.</b> Models help in trending markets and hurt in calm ones. During the 2021–22 shock the average of five models scored {win.loc['shock 2021-22', 3]:.2f} at 3 months; in calm periods it scored {win.loc['calm', 3]:.2f} at 3 months and {win.loc['calm', 9]:.2f} at 9 months.</li>
    <li><b>No hindsight.</b> When we choose the best model per price using only data before 2023 and score it from 2023 on, the driver regression still scores 0.64–0.76 on the steel indices.</li>
    <li><b>Statistical confidence.</b> With about 80 overlapping forecast dates, the gain on the steel indices is consistent but not significant at the 5% level (Diebold–Mariano p-values 0.05–0.25).</li>
    <li><b>The futures market.</b> Using the LME 3-month futures price as the forecast scores 0.995 for copper and 0.966 for aluminium at 3 months.</li>
    <li><b>Machine learning overfits.</b> With about 1,500 training rows, LightGBM scores 1.33 at 9 months, and its own 80% ranges hold only 56–64% of outcomes.</li>
  </ul>
</section>

<section class="pb">
  <h2>Why the index forecast works</h2>
  <p>The EU steel index (S09) is published two months late and adjusts slowly towards market prices. The regression picks this up. Its strongest inputs are last month's change in the CRU cold-rolled coil price and the size of the gap between the market price and the index.</p>
  {fig("14_dynreg_s09_coefs", "<b>Inputs to the S09 regression, ranked by weight</b> (standardised coefficients, 3-month horizon). <span class='mono'>x_CRU_CR_DE_d1</span> is the latest monthly change in the CRU coil price; <span class='mono'>gap_steel</span> is the market-to-index gap.")}
</section>

<section>
  <h2>How wide is the uncertainty?</h2>
  <p>Each forecast comes with an 80% range built from that price's own past forecast errors. The ranges hold {100*cov[3]:.0f}%, {100*cov[6]:.0f}% and {100*cov[9]:.0f}% of outcomes at 3, 6 and 9 months: close to the target, slightly narrow at long horizons.</p>
  {fig("15_forecast_fans_pct", "<b>Forecasts from the latest data</b>, as % change from the latest observation (central forecast and 80% range at 3, 6 and 9 months). The ranges are skewed upward because the 2021–22 shock is part of the error history.")}
</section>

<section class="pb">
  <h2>What this means for product cost</h2>
  <p>We combined the price forecasts with each bill of materials. The ranges are simulated by replaying the forecast errors of all materials from the same past month together, so that copper and steel shocks arrive at the same time, as they did in 2021–22.</p>
  {table(bom)}
  <p class="note">Change in priced material cost from today (priced materials only: steel, stainless, copper, aluminium, plastics). Bill of materials: the data.zip version with an assumed RM6 carbon-steel mass; to be re-run with the latest BOM (v1). Against the 2026 median RM6 selling price in the (synthetic) price benchmark, the 9-month bad case adds about 3.5% of the price and the central case 0.3%.</p>
  <h3>Does forecasting save money?</h3>
  <p>We tested a simple rule at each past month: if the forecast says the price will rise more than 2% in three months, buy three months of need now, paying 0.4% per month to hold stock; otherwise buy at the spot price later. Results use the ARIMA forecast.</p>
  {table(hed, num_cols=tuple(hed.columns[1:]), fmt="{:.1f}")}
  <p class="note">Perfect hindsight is the best any rule could do. On index-priced steel the rule gets most of the way there. On market prices it captures a small part of a larger opportunity, which is better pursued by hedging than by timing. Schneider's steel purchases are currently indexed on CRU coil prices, so the CR coil row is the one that applies to today's contracts.</p>
</section>

<section>
  <h2>Recommendations and next steps</h2>
  <h3>Use the right tool for each price</h3>
  <ul>
    <li><b>Market prices</b> (steel coil, copper, aluminium, nickel): use today's price or the futures price as the plan, and put the modelling effort into the risk range.</li>
    <li><b>Eurostat indices</b> (steel, galvanised, plastics): use the driver regression for budgets and purchase timing.</li>
  </ul>
  <h3>What to explore next</h3>
  <ol>
    <li>Models that switch between momentum and “no change” depending on market conditions, since models win in trends and lose in calm periods.</li>
    <li>More forward-looking data: full futures curves for copper and aluminium, steel futures, and mill price announcements.</li>
    <li>A longer price history (the public sources go back to the 1990s) so that machine-learning models have enough data.</li>
    <li>A pre-trained time-series forecasting model as an extra benchmark.</li>
  </ol>
</section>

<section class="pb">
  <h2>Appendix</h2>
  <h3>Models</h3>
  <ul>
    <li><b>Naive</b>: the forecast is today's price. <b>Naive + trend</b>: adds the average historical monthly change.</li>
    <li><b>ARIMA(1,1,0) with drift</b>, <b>exponential smoothing with damped trend</b> and <b>Theta</b>, fitted on log prices.</li>
    <li><b>Driver regression</b>: a ridge regression of the h-month log change on the price's recent changes and on lagged leading series, without an intercept, so it defaults to “no change”.</li>
    <li><b>LightGBM</b>: one gradient-boosted tree model per horizon trained on all twelve series together, plus 10th, 50th and 90th percentile models.</li>
    <li><b>Average of five</b>: equal weights on ARIMA, exponential smoothing, Theta, driver regression and LightGBM. Its 80% range uses the 10th and 90th percentiles of its own past errors for that price and horizon (online split-conformal).</li>
  </ul>
  <h3>Publication lags applied</h3>
  <p>Market prices (CRU, MEPS, LME, ECB exchange rates, power): 0 months. IMF metal and energy prices: 1 month. Eurostat producer indices: 2 months. Eurostat crude imports: 4 months.</p>
  <h3>Reproducing</h3>
  <p>Run <span class="mono">public/notebooks/02_forecasting_benchmark.ipynb</span> locally with the NDA data (about 25 seconds), then <span class="mono">uv run python public/reports/make_public_figures.py</span> and <span class="mono">uv run python public/reports/build_public_reports.py</span>. Code is in <span class="mono">src/features.py</span>, <span class="mono">src/models.py</span> and <span class="mono">src/backtest.py</span>.</p>
</section>
<footer>Public version (NDA-safe). Prepared for the Schneider Electric × X-HEC Data Challenge 2026 jury. Forecasts are statistical estimates, not price commitments. Licensed prices and Schneider-internal figures are shown only as indices, % changes or statistics.</footer>
"""
    return page("RMU Price Forecasting Benchmark", body)


# =========================================================================== REPORT 0
NB00 = ROOT / "notebooks" / "00_data_profiling_preprocessing.ipynb"
PROC = ROOT / "data" / "processed"
NB00_FIGS = {12: "00_rm_missingness", 18: "00_rm_rebased", 31: "00_energy_shock", 53: "00_stainless_base_alloy",
             71: "00_lme_spread", 75: "00_epoxy_typo"}


def _dlog(s: pd.Series) -> pd.Series:
    return np.log(s).diff()


def profiling_detail(raw: pd.DataFrame) -> dict[str, str]:
    """Per-source detail tables for report 0, all computed from data/processed/ so they track the pipeline."""
    out = {}
    # --- core RM panel: price behaviour
    names = {"D01": "Copper, EUR/t", "D02": "Aluminium, EUR/t", "S03": "Nickel, USD/t", "S04": "Iron ore, USD/dmt",
             "S05": "Zinc, USD/t", "S07": "EU copper products PPI", "S08": "EU aluminium PPI", "S09": "EU iron & steel PPI",
             "S10": "German steel PPI (HDG proxy)", "S11": "EU plastics PPI", "S12": "US steel PPI", "S13": "US plastics PPI"}
    rows = []
    for s, lab in names.items():
        x = raw[s].loc["2016-11":].dropna()
        d = _dlog(x).dropna()
        last = x.index[-1]
        rows.append([f"{lab} <span class='mono'>{s}</span>", x.iloc[-1], 100 * (x.iloc[-1] / x.loc[last - pd.DateOffset(months=12)] - 1),
                     f"{x.idxmax():%b-%Y}", 100 * (x.iloc[-1] / x.max() - 1), 100 * d.std() * np.sqrt(12), d.autocorr(1),
                     f"{100 * (np.exp(d.abs().max()) - 1):.0f}% ({d.abs().idxmax():%b-%y})"])
    out["prices"] = table(pd.DataFrame(rows, columns=["Series", "Last", "YoY %", "Peak", "Last vs peak %", "Vol % / yr",
                                                      "Momentum (AC1)", "Largest monthly move"]),
                          num_cols=("Last", "YoY %", "Last vs peak %", "Vol % / yr", "Momentum (AC1)"), fmt="{:,.1f}")
    ppi_ac1 = [_dlog(raw[s].dropna()).autocorr(1) for s in ("S08", "S09", "S10", "S11")]
    lme_ac1 = [_dlog(raw[s].dropna()).autocorr(1) for s in ("D01", "D02", "S03", "S05")]
    out["ac1"] = f"{min(ppi_ac1):.2f}–{max(ppi_ac1):.2f}", f"{min(lme_ac1):.2f}–{max(lme_ac1):.2f}"

    # --- energy
    ru = raw["ED06"].dropna().groupby(raw["ED06"].dropna().index.year).mean()
    ru = 100 * ru if ru.max() <= 1 else ru  # stored as a fraction
    ru_tbl = pd.DataFrame([ru.round(1).values], columns=[str(y) for y in ru.index])
    out["russia"] = table(ru_tbl, num_cols=tuple(ru_tbl.columns), fmt="{:.1f}")
    pw = raw[["E09", "E10", "E11"]].dropna().tail(3)
    pw = pd.DataFrame({"Month": [f"{m:%b %Y}" for m in pw.index], "Germany": pw.E09.values,
                       "France": pw.E10.values, "Hungary": pw.E11.values})
    out["power"] = table(pw, num_cols=("Germany", "France", "Hungary"), fmt="{:.1f}")
    hu = raw.E11.dropna()
    out["hu_jump"] = 100 * (hu.iloc[-1] / hu.iloc[-3] - 1)
    out["e_peak"] = (raw[["ED02", "E09", "ED04"]].loc["2022"].max() / raw[["ED02", "E09", "ED04"]].loc["2019"].mean())

    # --- BOM
    bom = pd.read_parquet(PROC / "bom.parquet").dropna(subset=["model_series"])
    prod = {"RMAirSeT IIQ 24kV IEC": "AirSeT", "RM6_NE_IQI_433520": "RM6"}
    bt_ = pd.DataFrame({"Product": bom.product_ref.map(prod), "Material": bom.material_family,
                        "Priced on": "<span class='mono'>" + bom.model_series + "</span>",
                        "Note": np.where(bom.is_assumption, '<span class="tag warn">mass assumed</span>', "")})
    out["bom"] = table(bt_)  # NDA: masses and shares are confidential

    # --- steel workbook
    mp = pd.read_parquet(PROC / "steel_index_mapping.parquet")
    vc = mp.groupby(["index", "panel_series"]).size().sort_values(ascending=False).rename("n").reset_index()
    vc["share"] = 100 * vc.n / len(mp)
    vc["cum"] = vc.share.cumsum()
    vc = vc.head(8)
    out["mapping"] = table(pd.DataFrame({"Rank": range(1, len(vc) + 1), "Index in the contract": vc["index"],
                                         "Panel series": "<span class='mono'>" + vc.panel_series + "</span>"}))  # NDA: no counts
    out["n_plants"] = mp.plant_name.nunique()
    out["lead"] = int((mp.lead_time_months == 1).sum())
    pairs = []
    for prod_ in ("HR", "CR", "HDG"):
        for cc in ("DE", "FR", "IT", "ES"):
            a, b = raw[f"CRU_{prod_}_{cc}"], raw[f"MEPS_{prod_}_{cc}"]
            ab = pd.concat([a, b], axis=1).dropna()
            pairs.append([f"{prod_} coil, {cc}", len(ab), ab.iloc[:, 0].corr(ab.iloc[:, 1]),
                          _dlog(ab.iloc[:, 0]).corr(_dlog(ab.iloc[:, 1])), 100 * (ab.iloc[:, 1] / ab.iloc[:, 0] - 1).mean()])
    cm = pd.DataFrame(pairs, columns=["Product", "Months", "Corr. of levels", "Corr. of monthly changes", "MEPS vs CRU, mean gap %"])
    out["cru_meps"] = table(cm, num_cols=tuple(cm.columns[1:]), fmt="{:.2f}")
    out["cm_dlog"] = f"{cm.iloc[:, 3].min():.2f}–{cm.iloc[:, 3].max():.2f}"
    ni = raw.S03 / raw.S06
    al = raw.CRU_SS304_DE_ALLOY
    out["ss_lags"] = [_dlog(al).corr(_dlog(ni).shift(k)) for k in (0, 1, 2)]
    sh = (raw.CRU_SS304_DE_ALLOY / raw.CRU_SS304_DE).dropna()
    out["ss_share"] = (100 * sh.mean(), 100 * sh.min(), 100 * sh.max())

    # --- LME
    lme = raw[["LME_CU_SPREAD_REL", "LME_AL_SPREAD_REL", "LME_CU_STOCKS", "LME_AL_STOCKS"]].dropna().tail(3)
    bpe = pd.read_parquet(PROC / "beyond_panel_end.parquet").set_index("series").value
    lt = pd.DataFrame({"Month": [f"{m:%b %Y}" for m in lme.index] + ["Nov 2026 (month to date)"],
                       "Copper 3M vs cash %": list(100 * lme.LME_CU_SPREAD_REL) + [100 * bpe["LME_CU_SPREAD_REL"]],
                       "Aluminium 3M vs cash %": list(100 * lme.LME_AL_SPREAD_REL) + [100 * bpe["LME_AL_SPREAD_REL"]],
                       "Copper stocks, kt": list(lme.LME_CU_STOCKS / 1e3) + [bpe["LME_CU_STOCKS"] / 1e3],
                       "Aluminium stocks, kt": list(lme.LME_AL_STOCKS / 1e3) + [bpe["LME_AL_STOCKS"] / 1e3]})
    out["lme"] = table(lt, num_cols=tuple(lt.columns[1:]), fmt="{:,.1f}")
    gap = pd.concat({"Cu": raw.LME_CU_CASH_EUR / raw.D01 - 1, "Al": raw.LME_AL_CASH_EUR / raw.D02 - 1}, axis=1).dropna()
    out["lme_gap"] = 100 * gap.abs().max().max()

    # --- epoxy recipe
    ep = raw[["EPOXY_BENZENE", "EPOXY_ACETONE", "EPOXY_ECH", "EPOXY_OTHER", "EPOXY_RESIN"]].dropna()
    Xe = np.c_[ep.iloc[:, :4].values, np.ones(len(ep))]
    coef, *_ = np.linalg.lstsq(Xe, ep.EPOXY_RESIN.values, rcond=None)
    r2 = 1 - ((ep.EPOXY_RESIN - Xe @ coef) ** 2).sum() / ((ep.EPOXY_RESIN - ep.EPOXY_RESIN.mean()) ** 2).sum()
    out["epoxy"] = table(pd.DataFrame([[r2, str(len(ep))]], columns=["R² of the linear recipe", "Months"]),
                         num_cols=("R² of the linear recipe", "Months"), fmt="{:,.3f}")  # NDA: no coefficients / €/t constant

    # --- teardown, price benchmark, suppliers
    cb = pd.read_parquet(PROC / "competitor_benchmark.parquet")
    brand = {"Schneider": "Schneider", "ormazabal": "Ormazabal", "ABB": "ABB", "Siemens": "Siemens", "eaton": "Eaton"}
    ct = pd.DataFrame({"Brand": [brand.get(b, "SGC") for b in cb.brand],
                       "Material % of should-cost": np.where(cb.is_valid, 100 * cb.material_share, np.nan),
                       "Insulation": cb.insulating_gas.str.replace(" ✓", ""),
                       "Valid": np.where(cb.is_valid, "", '<span class="tag bad">excluded</span>')})
    out["teardown"] = table(ct, num_cols=("Material % of should-cost",), fmt="{:,.0f}")  # NDA: no € or kg
    pb = pd.read_parquet(PROC / "price_benchmark.parquet").pivot(index="year", columns="brand", values="p50").sort_index()
    yoy = (100 * pb.pct_change()).loc[2023:].T.reset_index().rename(columns={"brand": "Brand"})
    yoy.columns = [str(c) for c in yoy.columns]
    out["price"] = table(yoy, num_cols=tuple(yoy.columns[1:]), fmt="{:,.1f}")
    out["suppliers"] = table(supplier_table())
    return out


def supplier_table() -> pd.DataFrame:
    """Anonymised supplier overlay: letters by revenue rank (A = largest), qualitative flags only (NDA)."""
    k = pd.read_parquet(PROC / "supplier_kpis.parquet").sort_values("revenue_keur", ascending=False, na_position="last")
    rows = []
    for i, r in enumerate(k.itertuples()):
        flags = []
        if pd.notna(r.debt_to_ebitda) and r.debt_to_ebitda > 10 or (pd.notna(r.interest_coverage) and 0 <= r.ebitda_margin and r.interest_coverage < 0):
            flags.append('<span class="tag bad">highly leveraged, interest not covered</span>')
        if pd.notna(r.ebitda_margin) and r.ebitda_margin < 0:
            flags.append('<span class="tag bad">negative EBITDA</span>')
        if pd.notna(r.current_ratio) and r.current_ratio < 1:
            flags.append('<span class="tag warn">current ratio below 1</span>')
        if pd.notna(r.revenue_year) and r.revenue_year <= r.last_year - 3:
            flags.append('<span class="tag warn">P&amp;L out of date</span>')
        if pd.isna(r.revenue_keur):
            flags.append('<span class="tag warn">balance sheet only</span>')
        margin = ("—" if pd.isna(r.ebitda_margin) else "negative" if r.ebitda_margin < 0 else "thin (&lt; 5%)" if r.ebitda_margin < 0.05
                  else "moderate (5–12%)" if r.ebitda_margin < 0.12 else "healthy (≥ 12%)")
        rows.append([f"Supplier {chr(65 + i)}", margin, " ".join(flags)])
    return pd.DataFrame(rows, columns=["Supplier (by revenue rank)", "EBITDA margin", "Flags"])


def extract_nb_figures() -> None:
    """Notebook 00 shows its plots inline only: copy the PNG outputs of the chosen cells to reports/figures/."""
    cells = json.loads(NB00.read_text(encoding="utf-8"))["cells"]
    for idx, name in NB00_FIGS.items():
        pngs = [o["data"]["image/png"] for o in cells[idx].get("outputs", []) if "image/png" in o.get("data", {})]
        if not pngs:
            raise RuntimeError(f"notebook 00 cell {idx} has no image output: re-run the notebook")
        (FIGS / f"{name}.png").write_bytes(base64.b64decode(pngs[0]))


def report_profiling() -> str:  # public: figures come from make_public_figures.py
    raw = pd.read_parquet(PROC / "panel_monthly.parquet")
    method = pd.read_parquet(PROC / "panel_impute_method.parquet")
    cat = pd.read_parquet(PROC / "series_catalog.parquet")

    imp = method.stack()
    imp = imp[imp != ""]
    n_cells, n_imp = method.size, len(imp)
    method_label = {
        "ffill": "Forward-fill of the recent edge (only if the whole trailing gap is ≤ 3 months)",
        "interp_internal": "Linear interpolation of a single-month hole",
        "interp_data_error": "Interpolation of a cell set to NaN because it was a typo",
        "lme_proxy": "Copper and aluminium Oct-2026 from LME cash × 12-month median ratio",
        "identity": "Recomputed from components (D01 = S01/S06, D02 = S02/S06)",
    }
    imputed = (imp.rename("method").reset_index().rename(columns={"level_1": "series"})
               .groupby("method").agg(cells=("series", "size"),
                                      months=("month", lambda m: ", ".join(d.strftime("%b-%Y") for d in sorted(set(m)))))
               .sort_values("cells", ascending=False).reset_index())
    imputed.insert(1, "What it does", imputed.method.map(method_label).fillna(""))
    imputed = imputed.rename(columns={"method": "Method", "cells": "Cells", "months": "Months"})

    det = profiling_detail(raw)
    imp_by = (imp.rename("method").reset_index().rename(columns={"level_1": "series"})
              .groupby(["method", "month"]).series.apply(lambda s: ", ".join(s) if len(s) <= 6 else f"{len(s)} series")
              .reset_index().sort_values(["month", "method"]))
    imp_by = pd.DataFrame({"Month": [f"{m:%b %Y}" for m in imp_by.month], "Method": "<span class='mono'>" + imp_by.method + "</span>",
                           "Series": imp_by.series})
    tg = cat[cat.role == "target"]
    tgt = pd.DataFrame({"Series": [f"<span class='mono'>{s}</span>" for s in tg.index], "Unit": tg.unit.values,
                        "BOM material": tg.bom_material.values, "First": [f"{d:%b %Y}" for d in pd.to_datetime(tg["first"])],
                        "Last": [f"{d:%b %Y}" for d in pd.to_datetime(tg["last"])], "Publication lag (months)": tg.pub_lag_months.values,
                        "Imputed cells": tg.n_imputed.values})
    n_lagging = int((cat.status == "lagging").sum()); n_disc = int((cat.status == "discontinued").sum())

    fam = cat.groupby(["family", "status"]).size().unstack(fill_value=0)
    fam = fam.reindex(columns=["live", "lagging", "discontinued"], fill_value=0)
    fam["total"] = fam.sum(axis=1)
    fam = fam.sort_values("total", ascending=False).reset_index().rename(columns={"family": "Family"})
    fam.columns = [str(c).capitalize() if c != "Family" else c for c in fam.columns]
    n_target = int((cat.role == "target").sum())

    outputs = []
    for f, use in [
        ("panel_monthly", "Wide panel, NaN kept. Use for backtests"),
        ("panel_monthly_filled", "Same grid, recent edge nowcast. Use for the production forecast"),
        ("panel_impute_method", "Per cell: empty = observed, else the imputation method"),
        ("panel_monthly_long", "Tidy version for global / panel models"),
        ("series_catalog", "Source, unit, family, role, publication lag, BOM material, status"),
        ("beyond_panel_end", "Nov-2026 quotes (LME month-to-date, CRU FR/ES/UK), provisional"),
        ("bom", "Bill of materials with model price series and mass assumptions"),
        ("steel_index_mapping", "Part references → CRU/MEPS index and panel series"),
        ("competitor_benchmark", "Teardown should-cost per product"),
        ("teardown_bom", "Teardown parts list"),
        ("price_benchmark", "Synthetic yearly P10/P50/P90 RMU prices"),
        ("supplier_financials", "PitchBook statements, long format"),
        ("supplier_kpis", "Latest supplier snapshot with risk flags"),
    ]:
        df = pd.read_parquet(PROC / f"{f}.parquet")
        outputs.append([f"<span class='mono'>{f}</span>", use])  # NDA: no row counts (they reveal internal sizes)
    outputs = pd.DataFrame(outputs, columns=["File (.parquet)", "Use"])

    files = pd.DataFrame([
        ["Core RM panel (<span class='mono'>rm_prices_monthly_wide.csv</span>)", "15 free RM series (LME, Eurostat, BLS, FX) + 8 licensed placeholders", "Nov-2016 → Sep/Oct-2026", '<span class="tag ok">primary</span>'],
        ["Energy panel (<span class='mono'>energy_prices_monthly_wide.csv</span>)", "Gas, power (DE/FR/HU), oil, Japan, Russia crude share", "Nov-2016 → Jul/Oct-2026", '<span class="tag ok">primary</span>'],
        ["Long CSVs, Excel masters in data.zip", "Same values as the wide CSVs", "—", '<span class="tag warn">redundant</span>'],
        ["BOM (<span class='mono'>bom_materials.csv</span>)", "Material masses for AirSeT and RM6", "Snapshot", '<span class="tag ok">primary</span>'],
        ["Root <span class='mono'>RM_Prices_XHEC_Challenge.xlsx</span>", "A second BOM only, with different masses", "Snapshot", '<span class="tag bad">conflict</span>'],
        ["<span class='mono'>RM_Prices_XHEC_Challenge_v1.xlsx</span> (22 Nov)", "Latest BOM: both products complete (adds RM6 carbon steel, copper and epoxy in AirSeT, brass)", "Snapshot", '<span class="tag ok">reference BOM</span>'],
        ["Europe Steel INDEXES (MEPS & CRU)", "Licensed flat, stainless and long steel prices; part → index mapping", "2012 → Oct-2026", '<span class="tag ok">high value</span>'],
        ["LME copper & aluminium", "Cash, 3-month, spreads, warehouse stocks", "2016 → Nov-2026 MTD", '<span class="tag ok">useful</span>'],
        ["Epoxy cost build-up", "Benzene, acetone, ECH, resin (EUR/t)", "Jan-2018 → Oct-2026", '<span class="tag ok">useful</span>'],
        ["Teardown should-cost (xlsx + pdf)", "RM6 vs 5 competitors: material, process, labour, overhead", "Snapshot", '<span class="tag warn">provisional</span>'],
        ["RMU IQI EU price benchmark", "Yearly P10/P50/P90 selling prices by brand", "2017 → 2026", '<span class="tag warn">synthetic</span>'],
        ["Supplier balance sheets (PitchBook)", "9 suppliers: P&amp;L, balance sheet, ratios", "2005 → 2025, uneven", '<span class="tag warn">patchy</span>'],
        ["Kick-off deck (pdf)", "Business context and evaluation criteria", "Nov-2026", '<span class="tag ok">context</span>'],
    ], columns=["File", "Content", "Coverage", "Verdict"])

    issues = pd.DataFrame([
        ["Licensed series L01–L08", "100 % empty, on purpose", "Dropped; the steel workbook fills most of the gap"],
        ["Recent edge", "Series end Jul–Oct 2026 (publication lag). No holes inside the core history", "Treated as nowcasting: LME proxy, identities, then forward-fill ≤ 3 months, all flagged"],
        ["Two BOMs", "AirSeT: root file has more carbon steel, less stainless than data.zip", "data.zip version used (documented, newer). To confirm with organisers"],
        ["RM6 carbon-steel mass", "Largest slice labelled “to confirm”, no mass", "Assumed share × implied total, flagged <span class='mono'>is_assumption</span>; now given by the latest BOM (v1)"],
        ["Epoxy typo", "May-2026 ECH about 10× below the neighbouring months; resin computed from it", "Both cells set to NaN, interpolated in the filled panel"],
        ["MEPS Belgium", "One internal missing month (Nov-2014)", "Linear interpolation, flagged"],
        ["▲ sheets (steel workbook)", "Months after the last price are 0, not blank", "Not exported; deltas recomputed from levels when needed"],
        ["Stainless ES/IT/UK, 316/430", "Stop in Feb-2026", "Marked lagging; not forward-filled for 8 months"],
        ["SGC should-cost", "Total two orders of magnitude below every other unit", "<span class='mono'>is_valid = False</span>, excluded"],
        ["PitchBook Debt/Equity", "0 for 8 of 9 suppliers, incl. one with material debt", "Recomputed as total debt / total equity"],
        ["Teardown masses", "Inferred; Schneider parts sum to more than twice the sheet's unit mass", "Order-of-magnitude use only"],
    ], columns=["Issue", "What we found", "Fix"])

    dups = pd.DataFrame([
        ["D01, D02", "S01/S06, S02/S06 (copper, aluminium in EUR)", "exact identity"],
        ["E23", "S06 (EUR/USD)", "exact copy, dropped"],
        ["ED01–ED06", "E series in harmonised units", "transformations"],
        ["*_long.csv, Excel masters", "Wide CSVs", "same values"],
        ["MEPS flat prices", "CRU flat prices", "level correlation ≈ 1.00"],
        ["LME cash (workbook)", "S01/S02", "within ≈ 0.1 %"],
        ["▲ sheets", "Differences of the price sheets", "100 % match"],
        ["Brent, WTI, Dubai", "Each other", "Δ correlation 0.83–0.93: use one"],
    ], columns=["Series", "Duplicates", "How close"])

    body = f"""
<header class="cover">
  <span class="eyebrow">Schneider Electric × X-HEC Data Challenge 2026 · Report 0 · Data audit</span>
  <h1>Data Profiling &amp; Preprocessing</h1>
  <p class="lede">Every file we received, what it is worth, what was wrong with it, and how it became the model-ready panel used by Reports 1 and 2.</p>
  <div class="meta"><span>Source <b>public/notebooks/00_data_profiling_preprocessing.ipynb</b></span><span>Panel <b>{raw.index.min():%b %Y} → {raw.index.max():%b %Y}</b></span><span>Report <b>{pd.Timestamp.today():%d %b %Y}</b></span></div>
</header>

<section>
  <div class="stats" style="grid-template-columns: repeat(auto-fit, minmax(140px, 1fr))">
    <div class="stat"><span class="v">{len(files)}</span><span class="l">sources profiled, column by column</span></div>
    <div class="stat"><span class="v">{raw.shape[1]}</span><span class="l">series in the panel, {n_target} of them forecast targets</span></div>
    <div class="stat"><span class="v">{raw.shape[0]}</span><span class="l">months on one month-start grid</span></div>
    <div class="stat"><span class="v">{n_imp}</span><span class="l">imputed cells out of {n_cells:,} ({100 * n_imp / n_cells:.2f} %), every one flagged</span></div>
  </div>
</section>

<section>
  <h2>Key takeaways</h2>
  <ol class="keys">
    <li><span><b>The free data is clean; the gap is at the edge.</b> No internal holes in the core panel. The only missing values are the last 1–3 months not yet published, so “imputation” is really nowcasting.</span></li>
    <li><span><b>The steel workbook is the most valuable file.</b> It has the licensed CRU/MEPS prices Schneider's buyers use, and maps Schneider part references to their contract index. Four indices (HDG Benelux, CR France, CR Spain, HDG Germany) cover most references, with a 1-month lead time.</span></li>
    <li><span><b>Much of the data is redundant.</b> Identities, copies and near-duplicates must not enter a model as independent signals.</span></li>
    <li><span><b>Some inputs are wrong or incomplete</b>: two conflicting BOMs, no RM6 carbon-steel mass, an epoxy typo, a broken competitor cost and a misleading PitchBook ratio. Each is fixed or flagged.</span></li>
    <li><span><b>Two price regimes.</b> Producer-price indices are smooth and persistent (AC1 ≈ 0.7–0.8); LME metals are close to a random walk (AC1 ≈ 0.1–0.3). Some prices are almost deterministic: the stainless alloy surcharge follows nickel one month later, and epoxy resin is a linear recipe of its feedstocks (R² ≈ 0.98).</span></li>
  </ol>
</section>

<section>
  <h2>Business context</h2>
  <p>We forecast raw-material prices 3, 6 and 9 months ahead for two medium-voltage ring main units (RMUs), then turn the forecasts into sourcing and pricing advice. RMUs are sold as engineer-to-order contracts with the price fixed for 3, 6 or 9 months. If material prices rise between the quote and the purchase, the margin absorbs the difference.</p>
  <div class="tbl"><table><thead><tr><th>Product</th><th>Main materials</th></tr></thead><tbody>
    <tr><td><b>RM AirSeT IIQ 24 kV</b> (SF6-free)</td><td>Carbon steel, stainless steel, plastics, aluminium</td></tr>
    <tr><td><b>RM6 NE-IQI</b> (SF6)</td><td>Carbon steel, stainless steel, copper, galvanised steel, PET, aluminium</td></tr>
  </tbody></table></div>
  <p>A good forecast with a reliable range lets procurement price a quote correctly, decide when to buy ahead or hedge, and design the index clauses in supplier contracts. Every file was audited in the same six steps: business meaning, every column, missing values, duplicates, anything unusual, and the decision taken.</p>
</section>

<section>
  <h2>What we received</h2>
  {table(files)}
</section>

<section class="pb">
  <h2>Source by source</h2>

  <h3>1 · Core raw-material panel</h3>
  <p>Prepared by the organisers from free public sources (IMF and BLS via FRED, ECB, Eurostat). Three kinds of series: <b>S</b> are downloaded prices and indices, <b>D</b> are derived in Excel (copper and aluminium converted to EUR), and <b>L</b> are licensed CRU / MEPS / ICIS prices left empty on purpose. Producer-price indices are smooth and are often what supply contracts are indexed on; exchange prices move daily.</p>
  {det["prices"]}
  <p class="note">Last observed month, change over 12 months, peak since Nov-2016, annualised volatility of monthly log changes, and the lag-1 autocorrelation of monthly changes. The producer-price indices have strong momentum (AC1 {det["ac1"][0]}); the LME metals are close to a random walk (AC1 {det["ac1"][1]}).</p>
  {fig("00_rm_rebased", "<b>Main raw-material series, rebased to November 2016 = 100.</b> Copper is at its peak in the last month; nickel's peak is the March-2022 LME squeeze.")}

  <h3>2 · Energy panel</h3>
  <p>Energy is the largest variable cost behind the materials and moves before them: aluminium smelting uses large amounts of power, steel uses gas and electric-arc power, and plastics are made from oil and gas. The file covers Europe (with power for Germany, France and Hungary), the US and Japan as competing cost regions, and Russia's share of EU crude imports. There are no holes inside the history; the edge ends in three steps: Oct-2026 for power and FX, Sep-2026 for IMF and Eurostat, Jul-2026 for crude-import data.</p>
  <p><b>Power price, EUR/MWh, last three months</b></p>
  {det["power"]}
  <p><b>Russia's share of EU crude imports, %, yearly mean</b></p>
  {det["russia"]}
  <p class="note">Hungarian power rose {det["hu_jump"]:.0f}% in two months. For EU production this is a conversion-cost risk, not only a leading indicator. At the 2022 peak, EU gas and German power reached {det["e_peak"]["ED02"]:.0f}× and {det["e_peak"]["E09"]:.0f}× their 2019 average, against {det["e_peak"]["ED04"]:.0f}× for US gas.</p>

  <h3>3 · Bill of materials</h3>
  <p>The only Schneider-internal file in <span class="mono">data.zip</span>, read off pie charts, so some slices are unlabelled. It turns material prices into a product material cost: cost(t) = Σ mass × price(t). In this version less than half of the RM6 mass is labelled, and the AirSeT “Other” slice cannot be priced. Masses are confidential and not shown. The organisers have since sent a <b>latest BOM (v1)</b>, now the reference, in which both products are complete.</p>
  {det["bom"]}
  <p class="note">The root workbook disagrees most on AirSeT carbon steel (higher), stainless and “Other” (both lower). The RM6 carbon-steel mass is an assumed share of the implied total, flagged as an assumption.</p>

  <h3>4 · Steel index workbook (MEPS &amp; CRU)</h3>
  <p>The most valuable file: the licensed market prices Schneider's steel buyers use, by country and product (hot-rolled, cold-rolled, galvanised coil; stainless 304 / 316 / 430 as base price plus alloy surcharge; long products). Its mapping sheet links Schneider part references across several plants to the index its supplier contract follows, with a 1-month lead time wherever it is filled. The most-used indices are, in order:</p>
  {det["mapping"]}
  <p>CRU and MEPS report almost the same prices. Their levels correlate at 0.99 or more and their monthly changes at {det["cm_dlog"]}, so one is kept as the target and the other only as a check.</p>
  {det["cru_meps"]}
  <p>The stainless alloy surcharge is computed from the previous month's nickel price. Its monthly change correlates {det["ss_lags"][0]:.2f} with nickel in the same month, {det["ss_lags"][1]:.2f} one month later and {det["ss_lags"][2]:.2f} two months later. The surcharge is on average {det["ss_share"][0]:.0f}% of the German 304 price (range {det["ss_share"][1]:.0f}–{det["ss_share"][2]:.0f}%). It is close to known one month ahead, so base and surcharge should be forecast separately.</p>
  <p class="note">Other sheets: the ▲ sheets are exact month-on-month differences of the price sheets (100% match) and fill future months with 0, not blank. Weekly CRU quotes average exactly to the monthly column. The 2017–18 snapshot sheets are negotiation records, and one contains employee names, which are not exported.</p>

  <h3>5 · LME copper &amp; aluminium</h3>
  <p>Cash and 3-month prices, the spread between them (above 0 = contango, a well-supplied market; below 0 = backwardation, a tight one) and LME warehouse stocks. The cash prices agree with the core panel's copper and aluminium to within {det["lme_gap"]:.1f}%, so the sources are consistent.</p>
  {det["lme"]}

  <h3>6 · Epoxy resin cost build-up</h3>
  <p>Epoxy is the insulating material in RMUs (cast-resin bushings, insulators). The chemistry runs from benzene and acetone to bisphenol-A, then with epichlorohydrin to epoxy. A linear recipe of the feedstock prices fits the resin almost exactly (May-2026 typo excluded):</p>
  {det["epoxy"]}

  <h3>7 · Teardown should-cost</h3>
  <p>A teardown estimate of what each competitor's 3-function unit should cost to make, marked provisional, with every part mass inferred. Materials are about 80% of should-cost for every valid unit, so a raw-material move passes almost one-for-one into competitive cost. € values and masses are confidential and not shown.</p>
  {det["teardown"]}

  <h3>8 · Synthetic selling-price benchmark</h3>
  <p>Yearly synthetic P10 / P50 / P90 selling prices in the EU, by brand. Change in the median price, %:</p>
  {det["price"]}
  <p class="note">The 2026 jump in every SF6 brand, against +8% for Eaton's SF6-free unit, matches the EU F-gas Regulation 2024/573 ban on SF6 in new switchgear up to 24 kV. Yearly and synthetic: use it for business framing, not as a forecasting target.</p>

  <h3>9 · Supplier financials (PitchBook)</h3>
  {det["suppliers"]}
  <p class="note">Suppliers are anonymised and ordered by revenue (A = largest); names and financial figures are confidential. Flags use the latest available year; debt / equity is recomputed from total debt and equity because PitchBook's own ratio is 0 for 8 of 9 suppliers. Coverage is too uneven for statistics, so this is a qualitative risk overlay.</p>
</section>

<section>
  <h2>Data-quality issues and fixes</h2>
  {table(issues)}
  {fig("00_rm_missingness", "<b>Core RM panel, observed (blue) vs missing.</b> L01–L08 (licensed) are empty by design; everything else is complete except the ragged right edge.")}
  {fig("00_epoxy_typo_pct", "<b>Epoxy chain, monthly % change.</b> The May-2026 epichlorohydrin value is a data-entry error that creates a spurious −90 % / +900 % month.")}
</section>

<section>
  <h2>Redundancy: do not double-count</h2>
  <p>No two files are byte-identical, but many carry the same information. Keep one of each pair as a model input.</p>
  {table(dups)}
</section>

<section>
  <h2>Findings that matter for the business</h2>
  <ul>
    <li><b>Copper (EUR) is at an all-time high</b> in the last observed month. It is the main risk for RM6; AirSeT has none in the data.zip BOM, and a small share in the latest BOM (v1).</li>
    <li><b>Copper is in backwardation</b> (cash &gt; 3-month): the market is tight now, consistent with the record price.</li>
    <li><b>The 2022 energy shock</b> lifted EU gas and power about 10× while US gas barely moved: the structural cost handicap of EU plants. <b>Hungarian power is rising fast again.</b></li>
    <li><b>Stainless = base + alloy surcharge.</b> The surcharge is computed from last month's nickel, so it is close to known one month ahead. Forecast the two parts separately.</li>
    <li><b>Materials are ≈ 80 % of should-cost</b> in every valid teardown unit, so a raw-material move passes almost one-for-one into competitive cost.</li>
    <li><b>SF6 regulation drives selling prices</b>: in the synthetic benchmark, SF6 brands rise 32–40 % in 2026 vs +8 % for the only SF6-free unit (EU F-gas Regulation 2024/573).</li>
    <li><b>Supplier risk:</b> the largest supplier is highly leveraged (debt far above EBITDA); one supplier has negative EBITDA; one has a P&amp;L several years out of date.</li>
  </ul>
  {fig("00_energy_shock", "<b>EU gas and power vs US gas (EUR/MWh).</b> Energy moves before steel, aluminium and plastics and is used as a leading indicator.")}
  <div class="two">
    {fig("00_stainless_alloy_share", "<b>Stainless 304 Germany</b>: alloy surcharge as a share of the transaction price (base + surcharge).")}
    {fig("00_lme_spread", "<b>LME 3-month vs cash.</b> Above 0 = contango, below 0 = backwardation.")}
  </div>
</section>

<section>
  <h2>Preprocessing pipeline</h2>
  <p>All in <span class="mono">src/preprocessing.py</span> → <span class="mono">run_pipeline()</span>.</p>
  <ol>
    <li><b>One grid.</b> All blocks (RM, energy, CRU, MEPS, stainless, long steel, LME, epoxy) on month-start, {raw.index.min():%Y-%m} → {raw.index.max():%Y-%m}. Never back-filled before a series' first observation.</li>
    <li><b>Cleaning.</b> Drop empty, duplicate and fragmentary series; set the epoxy typo to NaN; move Nov-2026 quotes to <span class="mono">beyond_panel_end</span>.</li>
    <li><b>Flagged imputation</b> (filled panel only), in order: interpolation of single holes → LME proxy → identities → forward-fill ≤ 3 months. Lagging and discontinued series stay NaN.</li>
    <li><b>Catalog.</b> Source, unit, family, role (target / driver), publication lag, BOM material, coverage, status.</li>
  </ol>
  <h3>Imputation methods</h3>
  {table(imputed, num_cols=("Cells",))}
  <h3>Which cells were imputed, month by month</h3>
  {table(imp_by)}
  <h3>Series by family and status</h3>
  <p>{n_lagging} series are <b>lagging</b> (mostly stainless grades and countries that stop in Feb-2026) and {n_disc} are <b>discontinued</b> (long products that stopped in 2017). These stay NaN and are used as history only.</p>
  {table(fam, num_cols=tuple(c for c in fam.columns if c != "Family"), fmt="{:.0f}")}
  <h3>The {n_target} forecast targets</h3>
  {table(tgt, num_cols=("Publication lag (months)", "Imputed cells"), fmt="{:.0f}")}
  <p class="note">The publication lag is how many months after the reference month a value is published. Features are shifted by it so a backtest never uses a value before it was actually available.</p>
  <p class="note">Sanity checks pass: complete grid, raw panel equals the loaders, imputation only touches NaN cells, identities hold, long-table keys are unique, every mapped index exists in the panel, and the targets of <span class="mono">features.build_panel()</span> are reproduced exactly.</p>
</section>

<section>
  <h2>Outputs in <span class="mono">data/processed/</span></h2>
  {table(outputs)}
  <div class="asks">
    <h3>Modelling rules this data implies</h3>
    <ul>
      <li><b>Model Δlog, not levels.</b> Every series is I(1) and indices have different bases.</li>
      <li><b>Shift drivers by <span class="mono">pub_lag_months</span></b> to avoid look-ahead: the September PPI is not known in early October.</li>
      <li><b>Backtest on <span class="mono">panel_monthly</span>, not the filled panel.</b> Imputed cells are estimates and are never scored as truth.</li>
      <li><b>No duplicate signals</b> (see the redundancy table).</li>
      <li><b>UK series are in GBP</b>; discontinued series are history only.</li>
    </ul>
  </div>
</section>
<footer>Public version (NDA-safe). Prepared for the Schneider Electric × X-HEC Data Challenge 2026 jury. Built by <span class="mono">public/reports/build_public_reports.py</span> from <span class="mono">data/processed/</span>. Licensed prices and Schneider-internal figures are shown only as indices, % changes or statistics.</footer>
"""
    return page("Data Profiling Report", body)


# =========================================================================== REPORT 3
TGT_LABEL = {"S09": "EU steel index (S09)", "S10": "EU galvanised index (S10)", "S11": "EU plastics index (S11)",
             "SS304_ALLOY_DE": "Stainless 304 alloy surcharge", "CRU_SS304_DE": "Stainless 304, Germany",
             "CRU_CR_DE": "CR coil, Germany", "CRU_CR_FR": "CR coil, France", "CRU_HDG_DE": "HDG coil, Germany",
             "D01": "Copper (EUR)", "D02": "Aluminium (EUR)", "S03": "Nickel", "S04": "Iron ore"}
DRV_LABEL = {"gap_steel": "CRU coil vs steel index gap", "cu_spread": "LME copper futures spread",
             "al_spread": "LME aluminium futures spread", "cu_stocks": "LME copper stocks", "al_stocks": "LME aluminium stocks",
             "S06": "EUR/USD", "D01": "copper", "D02": "aluminium", "S03": "nickel", "S04": "iron ore", "S05": "zinc",
             "S09": "steel index", "S10": "galvanised index", "S11": "plastics index", "ED01": "Brent", "ED02": "gas",
             "E09": "German power", "E13": "EU energy index", "E22": "coal", "CRU_CR_DE": "CRU CR coil",
             "EPOXY_BENZENE": "benzene", "EPOXY_ECH": "epichlorohydrin"}
OWN_LABEL = {"own_dlog_l0": "own change, last month", "own_dlog_l1": "own change, 2 months ago",
             "own_mom3": "own 3-month momentum", "own_mom6": "own 6-month momentum", "own_mom12": "own 12-month momentum",
             "own_vol6": "own 6-month volatility", "own_vol12": "own 12-month volatility",
             "own_dev_ma12": "own distance from 12-month average", "own_dev_ma24": "own distance from 24-month average"}


def feat_label(f: str) -> str:
    if f in OWN_LABEL:
        return OWN_LABEL[f]
    b = f[2:]
    for suf, txt in (("_dev24", " vs its 24-month mean"), ("_d1", ", 1-month change"), ("_d3", ", 3-month change")):
        if b.endswith(suf):
            return DRV_LABEL.get(b[: -len(suf)], b[: -len(suf)]) + txt
    return DRV_LABEL.get(b, b) + (" (level)" if b in ("gap_steel", "cu_spread", "al_spread") else "")


FAMILY_LABEL = {"rm_free_proxy": "EU indices &amp; LME (free data)", "steel_market": "Steel coil (CRU)", "stainless": "Stainless (CRU)"}


def feature_detail(sc: pd.DataFrame) -> dict[str, str]:
    """Extra tables for report 3: stationarity, Granger checks, selection funnel, stability, full core list."""
    import io, contextlib
    from statsmodels.tsa.stattools import adfuller, grangercausalitytests
    out = {}
    P = pd.read_parquet(PROC / "panel_monthly.parquet").rename(columns={"CRU_SS304_DE_ALLOY": "SS304_ALLOY_DE"})
    lab = {**TGT_LABEL, "EPOXY_RESIN": "Epoxy resin", "S05": "Zinc", "S06": "EUR/USD", "ED01": "Brent (EUR)", "ED02": "TTF gas (EUR)",
           "E09": "German power", "E13": "EU energy index", "E22": "Coal", "EPOXY_BENZENE": "Benzene", "EPOXY_ECH": "Epichlorohydrin",
           "LME_CU_SPREAD_REL": "LME copper spread", "LME_AL_SPREAD_REL": "LME aluminium spread",
           "LME_CU_STOCKS": "LME copper stocks", "LME_AL_STOCKS": "LME aluminium stocks"}

    def dl(s):
        return np.log(s).diff() if (s.dropna() > 0).all() else s.diff()

    rows = []
    for s in list(TGT_LABEL) + ["EPOXY_RESIN", "S05", "S06", "ED01", "ED02", "E09", "E22",
                                "LME_CU_SPREAD_REL", "LME_AL_SPREAD_REL", "LME_CU_STOCKS", "LME_AL_STOCKS"]:
        x = P[s].dropna()
        lv = np.log(x) if (x > 0).all() else x
        pl, pd_ = adfuller(lv, autolag="AIC")[1], adfuller(dl(x).dropna(), autolag="AIC")[1]
        verdict = ('<span class="tag ok">use monthly changes</span>' if pl > 0.05 and pd_ < 0.05 else
                   '<span class="tag warn">stationary in levels</span>' if pl <= 0.05 else '<span class="tag warn">borderline</span>')
        rows.append([lab[s], "target" if s in TGT_LABEL or s == "EPOXY_RESIN" else "driver", len(x), pl, pd_, verdict])
    adf = pd.DataFrame(rows, columns=["Series", "Role", "Months", "ADF p, level", "ADF p, monthly change", "Verdict"])
    out["cu_adf"] = adf.set_index("Series").loc["Copper (EUR)", "ADF p, monthly change"]
    out["adf"] = table(adf, num_cols=("Months", "ADF p, level", "ADF p, monthly change"), fmt="{:.3f}")

    pairs = [("S09", "CRU_CR_DE"), ("S10", "CRU_HDG_DE"), ("SS304_ALLOY_DE", "S03"), ("S11", "ED01"), ("S11", "E22"),
             ("CRU_CR_DE", "S04"), ("CRU_HDG_DE", "S05"), ("D02", "E13"), ("EPOXY_RESIN", "EPOXY_BENZENE"), ("EPOXY_RESIN", "EPOXY_ECH")]
    rows = []
    for t, d in pairs:
        xy = pd.concat([dl(P[t]), dl(P[d])], axis=1).dropna()
        r = {k: xy.iloc[:, 0].corr(xy.iloc[:, 1].shift(k)) for k in range(0, 4)}
        with contextlib.redirect_stdout(io.StringIO()):
            g = grangercausalitytests(xy, maxlag=3)
        p = {k: v[0]["ssr_ftest"][1] for k, v in g.items()}
        kb = min(p, key=p.get)
        rows.append([lab[d], lab[t], r[0], r[1], p[kb], f"{kb}", len(xy),
                     '<span class="tag ok">useful</span>' if p[kb] < 0.05 else '<span class="tag warn">not significant</span>'])
    gr = pd.DataFrame(rows, columns=["Driver", "Target", "r, same month", "r, 1 month later", "Granger p", "Best lag", "Months", "Verdict"])
    out["granger"] = table(gr, num_cols=("r, same month", "r, 1 month later", "Granger p", "Best lag", "Months"), fmt="{:.3f}")

    funnel = sc.groupby("horizon").agg(scored=("feature", "size"), stable=("stable_regimes", lambda s: int((s >= 2).sum())),
                                       candidate=("candidate", "sum"), selected=("selected", "sum"), core=("core", "sum"))
    funnel = funnel.reset_index().rename(columns={"horizon": "Horizon (months)", "scored": "Signal × price pairs scored",
                                                  "stable": "Same direction in ≥ 2 of 3 periods", "candidate": "Top-10 candidates",
                                                  "selected": "After VIF ≤ 5", "core": "Core"})
    out["funnel"] = table(funnel.astype(str), num_cols=tuple(funnel.columns))

    fam = sc.assign(fam=sc.target.map(lambda t: "EU indices" if t in ("S09", "S10", "S11") else
                                      ("Steel coil &amp; stainless" if t.startswith(("CRU", "SS304")) else "LME metals &amp; iron ore")))
    st = fam.groupby("fam").apply(lambda g: pd.Series({
        "Mean |ρ|, all signals": g.abs_rho.mean(), "Mean |ρ|, core": g[g.core].abs_rho.mean(),
        "Core signals stable in all 3 periods %": 100 * (g[g.core].stable_regimes == 3).mean(),
        "Core per price & horizon": g[g.core].groupby(["target", "horizon"]).size().mean()}))
    st = st.loc[["EU indices", "Steel coil &amp; stainless", "LME metals &amp; iron ore"]].reset_index().rename(columns={"fam": "Group"})
    out["stability"] = table(st, num_cols=tuple(st.columns[1:]), fmt="{:.2f}")
    out["stab_ppi"] = st.iloc[0, 3]; out["stab_mkt"] = st.iloc[1:, 3].mean()

    core = sc[sc.core].sort_values("rank_score", ascending=False)
    rows = []
    for t in TGT_LABEL:
        row = [TGT_LABEL[t]]
        for h in (3, 6, 9):
            g = core[(core.target == t) & (core.horizon == h)]
            row.append("<br>".join(f"{feat_label(f)} <span class='mono'>({r:+.2f})</span>" for f, r in zip(g.feature, g.rho)) or "—")
        rows.append(row)
    out["core_full"] = table(pd.DataFrame(rows, columns=["Price", "3 months", "6 months", "9 months"]))

    gl = pd.DataFrame([
        ["<span class='mono'>own_dlog_l0 / l1 / l2</span>", "The price's own log change in the last month, 2 and 3 months ago"],
        ["<span class='mono'>own_mom3 / 6 / 12</span>", "Own log change over the last 3, 6 or 12 months (momentum)"],
        ["<span class='mono'>own_vol6 / 12</span>", "Standard deviation of own monthly changes over 6 or 12 months"],
        ["<span class='mono'>own_dev_ma12 / 24</span>", "Log distance of the price from its own 12- or 24-month average"],
        ["<span class='mono'>x_&lt;driver&gt;_d1 / d3</span>", "Log change of a driver over 1 or 3 months, delayed by its extra publication lag"],
        ["<span class='mono'>x_gap_steel</span>", "log(CRU CR coil Germany) − log(EU steel index S09): how far the index lags the market"],
        ["<span class='mono'>x_&lt;spread / gap&gt;_dev24</span>", "A level signal (LME futures spread, steel gap) minus its 24-month average"],
        ["<span class='mono'>x_cu_spread, x_al_spread</span>", "(LME 3-month − cash) / cash: below 0 = tight market (backwardation)"],
        ["<span class='mono'>x_cu_stocks, x_al_stocks</span>", "LME warehouse stocks, 1- or 3-month change"],
    ], columns=["Signal", "Definition"])
    out["glossary"] = table(gl)
    return out


def report_features() -> str:
    sc = pd.read_parquet(PROC / "feature_selection.parquet")
    det = feature_detail(sc)
    core = sc[sc.core]
    n_cand = sc.feature.nunique()
    n_core = core.feature.nunique()
    core_per = core.groupby(["target", "horizon"]).size()

    # signal strength per target: strongest |rho| of a core feature, per horizon
    best = core.loc[core.groupby(["target", "horizon"]).abs_rho.idxmax()]
    strength = best.pivot(index="target", columns="horizon", values="rho")
    top_feat = best[best.horizon == 3].set_index("target").feature.map(feat_label)
    order = strength.abs().mean(axis=1).sort_values(ascending=False).index
    sig = pd.DataFrame({"Price": [TGT_LABEL[t] for t in order],
                        "Strongest feature at 3 months": [top_feat.get(t, "—") for t in order],
                        "ρ, 3m": strength.loc[order, 3].values, "ρ, 6m": strength.loc[order, 6].values,
                        "ρ, 9m": strength.loc[order, 9].values,
                        "Core features (3/6/9m)": [f"{core_per.get((t, 3), 0)} / {core_per.get((t, 6), 0)} / {core_per.get((t, 9), 0)}" for t in order]})
    ppi_rho = strength.loc[["S09", "S10", "S11"]].abs()
    mkt_rho = strength.loc[["CRU_CR_DE", "CRU_CR_FR", "CRU_HDG_DE", "CRU_SS304_DE", "D01", "S03", "S04"]].abs()

    common = core.groupby("feature").agg(n=("target", "size"), targets=("target", "nunique"), rho=("abs_rho", "mean"))
    common = common.sort_values("n", ascending=False).head(10).reset_index()
    common = pd.DataFrame({"Feature": [f"{feat_label(f)} <span class='mono'>{f}</span>" for f in common.feature],
                           "Kept as core (target × horizon)": common.n.values,
                           "Targets": common.targets.values, "Mean |ρ|": common.rho.values})
    gap_n = int((core.feature == "x_gap_steel_dev24").sum())
    own_share = 100 * core.feature.str.startswith("own_").mean()

    leads = pd.DataFrame([
        ["CRU CR coil, Germany", "EU steel index (S09)", "1 month", 0.715, 0.526],
        ["Nickel", "Stainless 304 alloy surcharge", "1 month", 0.707, 0.024],
        ["Brent crude (EUR)", "EU plastics index (S11)", "1 month", 0.632, 0.240],
        ["Iron ore", "CR coil, Germany", "1 month", 0.454, 0.317],
    ], columns=["Leading series", "Follows", "Best lag", "Correlation at best lag", "Correlation same month"])

    dedup = pd.DataFrame([
        ["CRU CR coil, Germany", "8", "Hot-rolled Germany, Italian CR / HDG / HR coil, MEPS copies"],
        ["MEPS CR coil, Germany", "13", "MEPS flat steel in Spain, France, Italy and other products"],
        ["CRU stainless 304 (base, alloy, total), Germany", "13", "French, Spanish, Italian and UK variants"],
        ["Copper and aluminium in EUR (D01, D02)", "5", "LME cash and 3-month prices in EUR; aluminium in USD (S02)"],
        ["Brent in EUR (ED01)", "2", "Brent and WTI in USD (E01, E02)"],
        ["Other pairs", "39", "Country variants of stainless 316/430 and long steel; MEPS UK products; LME USD cash vs 3-month; unit conversions of energy series"],
    ], columns=["Kept", "Dropped", "What was dropped"])

    recos = pd.DataFrame([
        ["EU steel, galvanised, plastics indices", "Gap between market coil and the index; own momentum; 3-month change in aluminium, coal and gas",
         "ρ 0.6–0.7 at every horizon, stable in all three periods", '<span class="tag ok">features help</span>'],
        ["Stainless alloy surcharge", "Nickel, last month (the surcharge formula)", "0.71 on monthly changes at a 1-month lag; ρ 0.42 at 3 months",
         '<span class="tag ok">3 months only</span>'],
        ["Steel coil (CRU CR, HDG, stainless)", "Gap to the index (mean reversion); EUR/USD and German power over 3 months; iron ore and zinc",
         "|ρ| 0.2–0.4; sign changes between periods", '<span class="tag warn">near random walk</span>'],
        ["Copper, aluminium, nickel, iron ore", "Own volatility; LME futures spread vs its 24-month mean; stocks; EUR/USD over 3 months",
         "|ρ| 0.15–0.35", '<span class="tag warn">focus on ranges</span>'],
        ["Epoxy resin", "Benzene (0.66 on monthly changes), then epichlorohydrin (0.38)", "The level recipe fits far better (R² 0.98) than monthly changes",
         '<span class="tag ok">forecast feedstocks</span>'],
    ], columns=["Price", "Features to keep", "Evidence", "Verdict"])

    rows = {h: int(pd.read_parquet(PROC / f"features_h{h}.parquet", columns=["y"]).y.notna().sum()) for h in (3, 6, 9)}
    nsel = {h: sc.query("horizon == @h and selected").feature.nunique() for h in (3, 6, 9)}

    body = f"""
<header class="cover">
  <span class="eyebrow">Schneider Electric × X-HEC Data Challenge 2026 · Report 3 · Feature study</span>
  <h1>Which Signals Predict Raw-Material Prices</h1>
  <p class="lede">We tested {n_cand} candidate signals against twelve raw-material prices at 3, 6 and 9 months ahead: how prices move together, which ones move first, which inputs repeat each other, and which signals were known in time to be used. The result is a short, checked list of inputs for each price.</p>
  <div class="meta"><span>Source <b>public/notebooks/03_feature_study.ipynb</b></span><span>History <b>2012 → Oct 2026</b></span><span>Report <b>{pd.Timestamp.today():%d %b %Y}</b></span></div>
</header>

<section>
  <div class="stats" style="grid-template-columns: repeat(auto-fit, minmax(140px, 1fr))">
    <div class="stat"><span class="v">172 → 92</span><span class="l">series after removing near-copies (monthly changes correlated above 0.95)</span></div>
    <div class="stat"><span class="v">{n_cand} → {n_core}</span><span class="l">candidate signals → signals kept as “core” for at least one price</span></div>
    <div class="stat"><span class="v">{ppi_rho.values.min():.2f}–{ppi_rho.values.max():.2f}</span><span class="l">strongest correlation with future returns on the EU indices, against {mkt_rho.values.min():.2f}–{mkt_rho.values.max():.2f} on market prices</span></div>
    <div class="stat"><span class="v">0</span><span class="l">signals that use information not yet published at the forecast date (automated check, 3 dates × 3 horizons)</span></div>
  </div>
</section>

<section>
  <h2>Summary</h2>
  <ol class="keys">
    <li><span><b>The signal is in the EU price indices, not in market prices.</b> For the steel, galvanised and plastics indices, the best signals correlate {ppi_rho.values.min():.2f}–{ppi_rho.values.max():.2f} with the return over the next 3 to 9 months, and the relation holds before 2021, during the 2021–22 shock and since 2023. For steel coil, copper, nickel and iron ore, the best signals reach only {mkt_rho.values.min():.2f}–{mkt_rho.values.max():.2f} and often change sign from one period to the next.</span></li>
    <li><span><b>One signal does most of the work: the gap between the market price and the index.</b> When CRU coil sits well above the EU steel index, the index rises in the following months to catch up. This gap is kept as a core signal {gap_n} times out of 36 price × horizon pairs, more than any other.</span></li>
    <li><span><b>Three pass-through links are stable and one month long.</b> Coil prices lead the EU steel index, nickel leads the stainless surcharge, and Brent leads the plastics index, each by one month with a correlation of 0.63–0.72. These links held in every 24-month window since 2018.</span></li>
    <li><span><b>Almost half the data repeats itself.</b> 80 of 172 series are near-copies of another (country variants, MEPS against CRU, prices quoted in two currencies). Leaving them in would make a model count the same information several times.</span></li>
    <li><span><b>The kept signals are few and checked.</b> Each price keeps 1 to 8 core signals per horizon. A signal is core only if it holds in at least two of three periods, does not repeat another signal, and either has a business reason or holds in all three periods. No signal uses data that was not yet published.</span></li>
  </ol>
  <div class="asks">
    <h3>What this means</h3>
    <ul>
      <li>Index-linked purchases (EU steel, galvanised, plastics) can be forecast from data Schneider already has: the gap to market coil prices and the recent trend.</li>
      <li>For market prices, more signals will not beat “same as today” by much. Effort should go into the risk range and hedging, as Report 2 found.</li>
      <li>EUR/USD should be used to convert prices, not to predict them: its correlation with copper flips between about −0.4 and +0.4 over time.</li>
    </ul>
  </div>
</section>

<section class="pb">
  <h2>Prices or price changes?</h2>
  <p>Two prices that both trend upward always look correlated, even if they are unrelated. So before measuring any link, we tested each series for a unit root (augmented Dickey–Fuller). A p-value above 0.05 on the level and below 0.05 on the monthly change means the series wanders in level but its changes are stable. Every later step therefore uses monthly log changes or future returns, never price levels.</p>
  {det["adf"]}
  <p class="note">Copper has the weakest result on monthly changes (p = {det["cu_adf"]:.3f}), because of its steady 2025–26 rise. The LME copper spread is borderline in levels. The LME aluminium spread and copper stocks are stationary in levels: they swing around a stable average, so they enter as their distance from the 24-month mean rather than as changes.</p>
  {fig("17_corr_levels", "<b>Correlation of price levels, for reference only.</b> Almost everything looks strongly correlated because everything rose in 2021–22. This picture is misleading and was not used for selection.")}
</section>

<section class="pb">
  <h2>Who moves first</h2>
  <p>We correlated each price's monthly change with the earlier monthly changes of every candidate driver, at delays of 0 to 6 months. Four links stand out and match how the prices are set: the EU indices are built from contract prices that follow the market, and the stainless surcharge is computed from the previous month's nickel price.</p>
  {table(leads, num_cols=("Correlation at best lag", "Correlation same month"))}
  <h3>Do the drivers add information? Granger tests</h3>
  <p>A Granger test asks whether a driver's past monthly changes improve a forecast of the target that already uses the target's own past (up to 3 lags). With 100–170 months, we read p &lt; 0.05 as “useful for forecasting”, not as proof of cause.</p>
  {det["granger"]}
  <p class="note">The pass-through links are confirmed: their correlation jumps from the same month to one month later, and the test is significant. Epoxy resin moves with benzene in the same month, so its feedstocks help to explain it but not to forecast it ahead.</p>
  {fig("19_corr_family_blocks", "<b>Same-month correlation of monthly changes, by family.</b> Left: CRU coil prices move together (0.91–0.99), and the steel and galvanised indices follow each other (0.95). Middle: the stainless surcharge tracks the total 304 price (0.75). Right: the plastics index moves with acetone (0.77) and epoxy resin (0.68); gas, power and coal move together.")}
  {fig("21_leadlag_ccf", "<b>Strongest correlation between a driver's past monthly change and each price's monthly change</b>, delay 0–6 months in brackets; values of 0.3 or more are labelled. The red block at the bottom right is the EU indices following market prices and each other.")}
  {fig("20_rolling_corr", "<b>Are the links stable?</b> Correlation over rolling 24-month windows. Coil → steel index, nickel → surcharge and Brent → plastics stay positive throughout. Iron ore → HDG coil fades in 2020 and 2026; copper vs EUR/USD changes sign.")}
</section>

<section>
  <h2>Removing duplicates</h2>
  <p>We walked through the series in priority order (forecast targets first, then known drivers, then the CRU German and French prices that match the part mapping) and dropped any series whose monthly changes correlate above 0.95 with one already kept. Two targets are themselves near-duplicates and are kept but flagged: HDG coil follows CR coil (0.99), and the galvanised index follows the steel index (0.95). Their forecasts should agree.</p>
  {table(dedup)}
  {fig("18_corr_dlog_pearson", "<b>Correlation of monthly changes after removing duplicates</b>, grouped by similarity. Clear blocks appear for flat steel, stainless, non-ferrous metals and energy.")}
  <h3>Are a few extreme months driving the links?</h3>
  <p>The usual (Pearson) correlation can be pulled up by a handful of extreme months, such as March 2022. The rank (Spearman) correlation is not. The biggest differences are CRU coil against nickel (Pearson higher by 0.35) and against zinc (by 0.22): those links mostly come from the 2021–22 shock. Stainless 304 and its surcharge against gas and German power go the other way (Pearson lower by 0.26–0.29). The pass-through links in the previous section are similar under both measures.</p>
  {fig("18b_corr_dlog_spearman", "<b>Rank correlation of monthly changes</b>, same series. Less affected by the 2021–22 outliers; the block structure is unchanged.")}
</section>

<section class="pb">
  <h2>Which signals predict the next 3 to 9 months</h2>
  <p>For each price and horizon we built {n_cand} candidate signals from the same code as the forecasting benchmark: the price's own recent changes, momentum and volatility, and the 1- and 3-month changes of each driver. Each signal is delayed by its real publication lag. We ranked them by rank correlation and mutual information with the future return, and checked that the direction held in three periods: before 2021, 2021–22, and since 2023.</p>
  {table(sig, num_cols=("ρ, 3m", "ρ, 6m", "ρ, 9m"))}
  <p class="note">ρ is the rank correlation of the strongest core signal with the return over the next 3, 6 or 9 months. Returns over several months overlap, so with about 110–170 dates per price a ρ of 0.2–0.3 can be chance. Read it as a ranking, not a test.</p>
  {fig("22_feature_target_rho", "<b>Rank correlation of each signal with the future return</b>, by price (columns) and horizon (panels). The three columns that are red almost everywhere are the EU steel, galvanised and plastics indices; market prices stay pale.")}
</section>

<section>
  <h2>The final selection</h2>
  <p>Per price and horizon we took the ten best-ranked signals that keep their direction in at least two of three periods, have |ρ| ≥ 0.10 and at least 60 observations. We then removed signals that repeat each other until every variance inflation factor was 5 or less. Finally, a signal is marked <b>core</b> only if it has a business reason (the price's own history, or a driver documented for that price) or holds in all three periods with |ρ| ≥ 0.25. This last step removes chance links such as the plastics index “predicting” steel coil.</p>
  <h3>How many signals survive each step</h3>
  {det["funnel"]}
  <h3>Strength and stability by group of prices</h3>
  {det["stability"]}
  <p class="note">Mean |ρ| over every signal scored and over core signals only, the share of core signals that keep the same direction in all three periods, and the average number of core signals per price and horizon. On the EU indices, {det["stab_ppi"]:.0f}% of core signals hold in every period, against {det["stab_mkt"]:.0f}% on average for market prices.</p>
  <h3>Signals kept most often</h3>
  {table(common, num_cols=("Kept as core (target × horizon)", "Targets", "Mean |ρ|"))}
  <p class="note">{own_share:.0f}% of core signals are the price's own history (momentum, volatility); the rest are drivers. Training files keep {nsel[3]}, {nsel[6]} and {nsel[9]} signals at 3, 6 and 9 months, with {rows[3]:,}, {rows[6]:,} and {rows[9]:,} training rows.</p>
  {fig("23_feature_selection", "<b>How many prices keep each signal as core</b>, stacked by horizon. The market-to-index gap measured against its 24-month average is kept 20 times; EUR/USD and own volatility follow.")}
</section>

<section class="pb">
  <h2>Recommended inputs by price</h2>
  {table(recos)}
  <h3>No look-ahead</h3>
  <p>The kick-off brief asks us to prove that every signal was known at forecast time. Each driver is delayed by the difference between its publication lag and the target's (for example, the EU indices are published two months late). An automated check then rebuilds all signals after scrambling every value after the forecast date, at three dates (June 2020, March 2022, January 2025) and three horizons. No signal changed.</p>
</section>

<section>
  <h2>Appendix</h2>
  <h3>Methods</h3>
  <ul>
    <li><b>Levels or changes.</b> Augmented Dickey–Fuller tests: every price is non-stationary in levels and stationary in monthly log changes, so correlations of levels are inflated by shared trends and are not used. LME futures spreads and copper stocks are stationary in levels and enter as their distance from the 24-month mean.</li>
    <li><b>Lead–lag.</b> Pearson correlation of monthly log changes, driver at month t−k against target at month t, k = 0–6, minimum 36 months. Granger tests (up to 3 lags) were run on pairs with |r| ≥ 0.3 as a supporting check.</li>
    <li><b>Signal ranking.</b> Average percentile rank of |Spearman ρ| and mutual information (5 neighbours) with log(y<sub>t+h</sub>/y<sub>t</sub>).</li>
    <li><b>Redundancy.</b> Variance inflation factor ≤ 5 on standardised signals; the lowest-ranked signal above 5 is dropped first.</li>
  </ul>
  <h3>Signal glossary</h3>
  {det["glossary"]}
  <h3>Core signals per price and horizon</h3>
  <p>Ordered by rank score; the number is ρ with the future return. A positive ρ means a higher signal value has come before a higher future price.</p>
  {det["core_full"]}
  <h3>Outputs</h3>
  <p><span class="mono">data/processed/feature_selection.parquet</span>: one row per price, horizon and signal, with ρ, mutual information, period stability, VIF and the <span class="mono">selected</span> and <span class="mono">core</span> flags. <span class="mono">data/processed/features_h3/h6/h9.parquet</span>: the training tables, indexed by month and price.</p>
  <h3>Next step</h3>
  <p>Retrain the forecasting benchmark with the core signals per price (and the wider “selected” set as a comparison), in the same rolling backtest as Report 2. That backtest, not these correlations, decides which signals stay.</p>
</section>
<footer>Public version (NDA-safe). Prepared for the Schneider Electric × X-HEC Data Challenge 2026 jury. Built by <span class="mono">public/reports/build_public_reports.py</span> from <span class="mono">data/processed/feature_selection.parquet</span>. Licensed prices are shown only as statistics.</footer>
"""
    return page("Feature Study Report", body)


def to_pdf(html: Path) -> Path:
    pdf = html.with_suffix(".pdf")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=8000",
                    f"--print-to-pdf={pdf}", html.as_uri()], check=True, capture_output=True)
    return pdf


if __name__ == "__main__":
    only = set(sys.argv[1:])
    for name, fn in [("00_profiling_report", report_profiling), ("01_eda_report", report_eda),
                     ("02_forecasting_report", report_forecast), ("03_feature_report", report_features)]:
        if only and name not in only:
            continue
        out = REPORTS / f"{name}.html"
        out.write_text(fn(), encoding="utf-8")
        pdf = to_pdf(out)
        print(f"{out.relative_to(ROOT)}  {out.stat().st_size/1e6:.1f} MB  |  {pdf.relative_to(ROOT)}  {pdf.stat().st_size/1e6:.1f} MB")
