"""Figures of the AI track: scores, backtest and live forecast.

    uv run python -m src.ai.plots        # or: make ai-plots (run make ai first)

Only reads what ``make ai`` wrote in data/outputs/ai/ (no model is run, nothing is selected or scored again) and
writes PNG files to data/outputs/ai/figures/ (git-ignored). Everything here is PRIVATE: licensed series are drawn in
separate files (``LICENSED_internal``), and no figure may be copied to public/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src import features as fe
from src.testbench.protocol import NOT_PUBLISHABLE_PREFIXES

from . import config as C

BLUE, ORANGE = "#2a78d6", "#eb6834"  # same chart system as public/reports/make_public_figures.py
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.frameon": False, "legend.fontsize": 8, "font.size": 9,
    "font.family": "sans-serif",
}

# name and unit of the 12 targets. S03 and S04 are in USD (source catalogue), not converted to EUR.
SERIES = {
    "D01": ("Copper", "EUR/t"), "D02": ("Aluminium", "EUR/t"), "S03": ("Nickel", "USD/t"),
    "S04": ("Iron ore", "USD/dmt"), "S09": ("Steel PPI, EU", "index 2021=100"),
    "S10": ("Steel PPI, Germany", "index 2021=100"), "S11": ("Plastics PPI, EU", "index 2021=100"),
    "CRU_CR_DE": ("Cold-rolled steel, DE", "EUR/t"), "CRU_CR_FR": ("Cold-rolled steel, FR", "EUR/t"),
    "CRU_HDG_DE": ("Galvanised steel, DE", "EUR/t"), "CRU_SS304_DE": ("Stainless 304, DE", "EUR/t"),
    "SS304_ALLOY_DE": ("Stainless surcharge, DE", "EUR/t"),
}
FAMILIES = {"lme": "LME metals", "ore": "Iron ore", "ppi": "Producer prices", "stainless": "Stainless",
            "steel_market": "Steel market"}
MODELS = {  # plotting order and readable names
    "naive_vol": "naive_vol (team benchmark)", "ensemble": "Classical ensemble", "dynreg": "Dynamic regression",
    "chronos-bolt-small": "Chronos-Bolt small", "chronos-bolt-base": "Chronos-Bolt base",
    "chronos-2+cov": "Chronos-2 + drivers", "chronos2+classical": "Chronos-2 + classical",
    "chronos2_p50+nvband": "Chronos-2 P50 + naive_vol range", "chronos2~naive_vol": "Chronos-2 ~ naive_vol average",
    "ai_selected": "Automatic pick (ai_selected)",
}
SHORT = {"naive": "naive", "naive_vol": "naive_vol", "ensemble": "ensemble", "dynreg": "dynreg",
         "chronos-bolt-small": "bolt-small", "chronos-bolt-base": "bolt-base", "chronos-2+cov": "chronos-2",
         "chronos2+classical": "c2+classical", "chronos2_p50+nvband": "c2+nv range", "chronos2~naive_vol": "c2~naive_vol"}
PICK_NOTE = "Automatic picks under review (selection compares unequal row sets): do not submit as is."
INPUTS = ("ai_backtest.parquet", "ai_live_forecast.parquet", "ai_scores_clean_window.csv",
          "ai_scores_clean_window_vs_naive_vol.csv", "ai_scores_from_2024.csv",
          "ai_scores_from_2024_vs_naive_vol.csv", "ai_scores_by_family_from_2024.csv")


# --------------------------------------------------------------------------------------------- inputs
def load(out: Path) -> dict[str, pd.DataFrame]:
    missing = [f for f in INPUTS if not (out / f).exists()]
    if missing:
        raise SystemExit(f"missing in {out}: {', '.join(missing)}. Run `make ai` first.")
    d = {f.rsplit(".", 1)[0]: pd.read_parquet(out / f) if f.endswith(".parquet") else pd.read_csv(out / f)
         for f in INPUTS}
    origins = set(d["ai_live_forecast"].origin)
    if origins != {pd.Timestamp(C.LIVE_ORIGIN)}:
        raise SystemExit(f"forecast origin {sorted(origins)} is not {C.LIVE_ORIGIN}. Run `make ai` again.")
    return d


def groups(series) -> dict[str, list[str]]:
    """Series allowed in the submission CSV, and licensed ones (never drawn in the same file)."""
    known = [s for s in SERIES if s in set(series)] + sorted(set(series) - set(SERIES))
    lic = [s for s in known if s.startswith(NOT_PUBLISHABLE_PREFIXES)]
    return {g: ss for g, ss in {"csv_series": [s for s in known if s not in lic], "LICENSED_internal": lic}.items()
            if ss}


def history(T: pd.DataFrame, s: str, months: int = 36) -> pd.Series:
    """Prices up to the official origin only: later months are not known to the forecast."""
    return T[s].loc[:pd.Timestamp(C.LIVE_ORIGIN)].dropna().iloc[-months:]


def family_tables(fam: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Error vs naive, and range quality vs naive_vol (ratio of the two ``relQL``, exact on identical rows)."""
    p = fam.pivot(index="model", columns="family")
    n = p["n"].dropna(how="all")
    if not (n.eq(n.loc["naive_vol"]) | n.isna()).all().all():
        raise ValueError("models are not scored on the same rows: the ratio of relQL is not valid")
    return p["relMAE"], p["relQL"] / p["relQL"].loc["naive_vol"]


def name(s: str) -> str:
    return f"{SERIES[s][0]} ({s})" if s in SERIES else s


def _footer(fig, text: str) -> None:
    fig.text(0.01, 0.005, f"PRIVATE, do not publish. {text}", fontsize=7.5, color=INK2, ha="left", va="bottom")


def _private(group: str) -> str:
    return "Licensed series: internal use only. " if group == "LICENSED_internal" else ""


def _save(fig, out: Path, stem: str) -> Path:
    path = out / f"{stem}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path


def _price_axes(ax, every: int) -> None:
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=range(1, 13, every)))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))


def _margins(fig, left: float, top: float = 0.95, bottom: float = 0.85, wspace: float = 0.06) -> None:
    """Top and bottom margins in inches, so the title and the notes keep the same distance for any number of rows."""
    h = fig.get_size_inches()[1]
    fig.subplots_adjust(left=left, right=0.985, top=1 - top / h, bottom=bottom / h, wspace=wspace)


def _panels(n: int, ncol: int = 3, h: float = 2.5):
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, h * nrow + 1.1), squeeze=False, layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.97), hspace=0.08, wspace=0.05)
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, list(axes.flat[:n])


# --------------------------------------------------------------------------------------------- figures
def fig_scores(d: dict, out: Path) -> Path:
    clean, late = d["ai_scores_clean_window"].set_index("model"), d["ai_scores_from_2024"].set_index("model")
    cleanv = d["ai_scores_clean_window_vs_naive_vol"].set_index("model")
    latev = d["ai_scores_from_2024_vs_naive_vol"].set_index("model")
    models = [m for m in MODELS if m in late.index]
    cols = [("Error vs naive", "below 1 is better", clean.relMAE, late.relMAE, 1.0),
            ("Range quality vs naive_vol", "below 1 is better", cleanv.relQL, latev.relQL, 1.0),
            ("Coverage of P10–P90", "target 0.80", clean.coverage, late.coverage, 0.8)]
    y = np.arange(len(models))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.44 * len(models) + 2.0), sharey=True)
    _margins(fig, 0.27, wspace=0.08)
    for ax, (title, sub, c, f, ref) in zip(axes, cols):
        ax.axvline(ref, color=INK2, lw=1)
        for yi, m in zip(y, models):
            a, b = c.get(m, np.nan), f.get(m, np.nan)
            ax.plot([a, b], [yi, yi], color=AXIS, lw=1.2, zorder=1)
            ax.plot(b, yi, "o", ms=8, mfc=SURFACE, mec=BLUE, mew=1.6, zorder=2)
            ax.plot(a, yi, "o", ms=8, mfc=BLUE, mec=SURFACE, mew=1.2, zorder=3)
            if np.isfinite(a):
                ax.annotate(f"{a:.2f}", (a, yi), (0, 8), textcoords="offset points", ha="center", fontsize=7.5,
                            color=INK, bbox=dict(boxstyle="square,pad=0.05", fc=SURFACE, ec="none"), zorder=2)
        ax.set_title(f"{title}\n", fontsize=10.5)
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=INK2)
        ax.margins(x=0.12)
        ax.set_ylim(-0.6, len(models) - 0.3)
        ax.tick_params(length=0)
        ax.grid(axis="y", visible=False)

    def label(m):
        k = f"n = {int(clean.n[m])} clean, " if m in clean.index else "no clean window, n = "
        return f"{MODELS[m]}\n{k}{int(late.n[m])} from 2024"
    axes[0].set_yticks(y, [label(m) for m in models], color=INK, fontsize=8)
    fig.suptitle("AI track scores", x=0.01, y=0.98, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=8, mfc=BLUE, mec=SURFACE,
                               label="Clean window: months after the model's release (value shown)"),
                        Line2D([], [], marker="o", ls="", ms=8, mfc=SURFACE, mec=BLUE, mew=1.6,
                               label="Forecasts made from 2024 on")],
               loc="upper right", bbox_to_anchor=(0.99, 1.0), ncol=2)
    _footer(fig, "Forecasts made from 2024 on may cover months the pretrained models saw in training. Clean windows "
                 "differ between models (n), so compare filled dots only at equal n.\nThe classical models have no "
                 "clean-window score. Forecasts overlap, so there are fewer independent ones than n. " + PICK_NOTE)
    return _save(fig, out, "01_scores")


def fig_family(d: dict, out: Path) -> Path:
    err, rng = family_tables(d["ai_scores_by_family_from_2024"])
    models = [m for m in MODELS if m in err.index]
    fams = [f for f in FAMILIES if f in err.columns]
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", ORANGE])
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.4 * len(models) + 2.0), sharey=True)
    _margins(fig, 0.21, top=0.75, wspace=0.05)
    for ax, (title, t) in zip(axes, [("Error vs naive", err), ("Range quality vs naive_vol", rng)]):
        v = t.reindex(index=models, columns=fams).to_numpy(float)
        ax.imshow(v, cmap=cmap, norm=TwoSlopeNorm(1.0, vmin=0.5, vmax=1.5), aspect="auto")
        for (i, j), x in np.ndenumerate(v):
            if np.isfinite(x):
                ax.text(j, i, f"{x:.2f}", ha="center", va="center", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(fams)), [FAMILIES[f] for f in fams], color=INK2)
        ax.set_yticks(range(len(models)), [MODELS[m] for m in models], color=INK)
        ax.set_xticks(np.arange(len(fams) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(models) + 1) - 0.5, minor=True)
        ax.grid(False)
        ax.grid(which="minor", color=SURFACE, lw=2)
        ax.tick_params(which="both", length=0)
        ax.spines[:].set_visible(False)
        ax.set_title(title)
    fig.suptitle("Scores by material family, forecasts made from 2024 on", x=0.01, y=0.98, ha="left", fontsize=13,
                 fontweight="bold", color=INK)
    fig.legend(handles=[Patch(fc=BLUE, label="below 1: better"), Patch(fc="#f0efec", ec=AXIS, label="1: same"),
                        Patch(fc=ORANGE, label="above 1: worse")], loc="upper right", bbox_to_anchor=(0.99, 1.0), ncol=3)
    _footer(fig, "2024–26 may include months the pretrained models saw in training. Full colour at 0.5 and 1.5.\n"
            + PICK_NOTE)
    return _save(fig, out, "02_scores_by_family")


def fig_backtest(d: dict, group: str, series: list[str], out: Path, h: int = 6) -> Path:
    r = d["ai_backtest"]
    r = r[(r.model == "ai_selected") & (r.h == h)]
    fig, axes = _panels(len(series))
    for ax, s in zip(axes, series):
        g = r[r.series == s].sort_values("target_month")
        lo, mid, hi = (g.y_origin * np.exp(g[c]) for c in ("q10", "pred", "q90"))
        ax.fill_between(g.target_month, lo, hi, color=BLUE, alpha=0.18, lw=0)
        ax.plot(g.target_month, mid, color=BLUE, lw=2)
        ax.plot(g.target_month, g.y_true, color=INK, lw=2)
        inside = ((g.y_true >= lo) & (g.y_true <= hi)).mean()
        ax.set_title(name(s))
        ax.set_xlabel(f"inside range {inside:.0%}, n = {len(g)}", loc="left", fontsize=8)
        ax.set_ylabel(SERIES.get(s, ("", ""))[1], fontsize=8)
        _price_axes(ax, 6)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Backtest: actual price and forecast made {h} months before (automatic pick, origins from 2024)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.legend(handles=[Line2D([], [], color=INK, lw=2, label="Actual"),
                        Line2D([], [], color=BLUE, lw=2, label="Forecast P50"),
                        Patch(fc=BLUE, alpha=0.18, label="Forecast P10–P90")], loc="outside lower right", ncol=3)
    _footer(fig, _private(group) + "Target inside range: 80%. May include months the models saw in training.\n"
            + PICK_NOTE)
    return _save(fig, out, f"03_backtest_{group}")


def fig_forecast(d: dict, T: pd.DataFrame, group: str, series: list[str], out: Path) -> Path:
    live = d["ai_live_forecast"]
    fig, axes = _panels(len(series), h=2.9)
    for ax, s in zip(axes, series):
        y, g = history(T, s), live[live.series == s].sort_values("h")
        ax.plot(y.index, y, color=INK, lw=2)
        ax.axvline(pd.Timestamp(C.LIVE_ORIGIN), color=MUTED, lw=0.8)
        ax.vlines(g.target_month, g.p10, g.p90, color=BLUE, lw=4, alpha=0.35)
        ax.plot(g.target_month, g.p50, "o", ms=8, mfc=BLUE, mec=SURFACE, mew=1.2)
        ax.set_title(name(s))
        for k, f in enumerate(g.itertuples()):
            ax.text(k * 0.345, -0.16, f"{f.h} months: {f.p50:,.0f}", transform=ax.transAxes, fontsize=8, color=INK,
                    va="top")
            ax.text(k * 0.345, -0.235, SHORT.get(f.model, f.model), transform=ax.transAxes, fontsize=7.5, color=INK2,
                    va="top")
        ax.set_ylabel(SERIES.get(s, ("", ""))[1], fontsize=8)
        ax.set_xlim(y.index[0], g.target_month.max() + pd.DateOffset(months=2))
        _price_axes(ax, 6)
        ax.grid(axis="x", visible=False)
    o = pd.Timestamp(C.LIVE_ORIGIN).strftime("%b %Y")
    fig.suptitle(f"Live forecast from {o}: 3, 6 and 9 months ahead", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=INK)
    fig.legend(handles=[Line2D([], [], color=INK, lw=2, label="Actual, up to the forecast date"),
                        Line2D([], [], marker="o", ls="", ms=8, mfc=BLUE, mec=SURFACE, label="P50 (value under the panel)"),
                        Line2D([], [], color=BLUE, lw=4, alpha=0.35, label="P10–P90")],
               loc="outside lower right", ncol=3)
    _footer(fig, _private(group) + "Dots are not joined: each horizon may use a different model (under each panel; c2 = Chronos-2, nv = naive_vol).\n"
            + PICK_NOTE)
    return _save(fig, out, f"04_forecast_{group}")


def fig_change(d: dict, group: str, series: list[str], out: Path) -> Path:
    live = d["ai_live_forecast"]
    live = live[live.series.isin(series)]
    pct = {c: 100 * (live[c] / live.y_origin - 1) for c in ("p10", "p50", "p90")}
    live = live.assign(**pct)
    hs = sorted(live.h.unique())
    y = dict(zip(series, np.arange(len(series))[::-1]))
    fig, axes = plt.subplots(1, len(hs), figsize=(13, 0.48 * len(series) + 2.0), sharey=True, sharex=True, squeeze=False)
    _margins(fig, 0.2, top=0.75, bottom=0.7)
    for ax, h in zip(axes[0], hs):
        g = live[live.h == h]
        yy = g.series.map(y)
        ax.axvline(0, color=INK2, lw=1)
        ax.hlines(yy, g.p10, g.p90, color=BLUE, lw=4, alpha=0.35)
        ax.plot(g.p50, yy, "o", ms=8, mfc=BLUE, mec=SURFACE, mew=1.2)
        for f, yi in zip(g.itertuples(), yy):
            ax.annotate(f"{f.p50:+.1f}%", (f.p50, yi), (0, 8), textcoords="offset points", ha="center", fontsize=7.5,
                        color=INK, bbox=dict(boxstyle="square,pad=0.05", fc=SURFACE, ec="none"), zorder=2)
        ax.set_title(f"{h} months ({g.target_month.iloc[0]:%b %Y})")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0%"))
        ax.set_ylim(-0.6, len(series) - 0.3)
        ax.tick_params(length=0)
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(list(y.values()), [name(s) for s in y], color=INK)
    fig.suptitle("Forecast change from the last known price", x=0.01, y=0.98, ha="left", fontsize=13,
                 fontweight="bold", color=INK)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=8, mfc=BLUE, mec=SURFACE, label="P50 (value shown)"),
                        Line2D([], [], color=BLUE, lw=4, alpha=0.35, label="P10–P90")],
               loc="upper right", bbox_to_anchor=(0.99, 1.0), ncol=2)
    _footer(fig, _private(group) + PICK_NOTE)
    return _save(fig, out, f"05_forecast_change_{group}")


def build(d: dict, T: pd.DataFrame, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        paths = [fig_scores(d, out), fig_family(d, out)]
        for g, ss in groups(d["ai_live_forecast"].series.unique()).items():
            paths += [fig_backtest(d, g, ss, out), fig_forecast(d, T, g, ss, out), fig_change(d, g, ss, out)]
    return sorted(paths)


def main() -> None:
    d = load(C.OUT)
    T, _ = fe.build_panel()
    for p in build(d, T, C.OUT / "figures"):
        print(p.relative_to(C.ROOT))


if __name__ == "__main__":
    main()
