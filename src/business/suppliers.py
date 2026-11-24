"""Task 8 — supplier emissions (Scope 1+2 and Scope 3 cat. 1) estimated from size, sector and country.

    uv run python -m src.business.suppliers

Revenue is parsed from the raw PitchBook files in data/raw/suppliers/ (never typed by hand). The peer pool is
SYNTHETIC (demo of the pipeline). Rules: peer S1+S2 and spend-based figures are never added; S3 stacks estimates on
estimates; all figures are for the whole supplier, not Schneider's share.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import openpyxl  # noqa: E402
import pandas as pd  # noqa: E402

from src import data as d  # noqa: E402

from .config import load_business_config  # noqa: E402
from .paths import fig_dir, out_dir  # noqa: E402

FIRST_COL = 12  # column M
OVERVIEW_HDR, STATEMENT_HDR = 4, 5  # row 5 / row 6 (0-based)


# ---------------------------------------------------------------------------------------------- parser
def supplier_dir():
    return d.RAW / "suppliers"


def _norm(s: str) -> str:
    """Lower-case ASCII letters and digits only (brackets, underscores, accents ignored) for file matching."""
    s = unicodedata.normalize("NFKD", s)
    return re.sub(r"[^0-9a-z]", "", s.encode("ascii", "ignore").decode().lower())


def find_file(keyword: str, folder=None):
    folder = supplier_dir() if folder is None else folder
    hits = [p for p in sorted(folder.glob("*.xlsx")) if _norm(keyword) in _norm(p.name)]
    if len(hits) != 1:
        raise FileNotFoundError(f"file_keyword {keyword!r}: expected exactly one PitchBook file in {folder}, "
                                f"found {len(hits)}: {[p.name for p in hits]}")
    return hits[0]


def _rows(wb, sheet):
    return list(wb[sheet].iter_rows(values_only=True)) if sheet in wb.sheetnames else None


def _check_units(rows, sheet, fname):
    text = str(rows[6][2]) if len(rows) > 6 and len(rows[6]) > 2 else ""
    if "thousands" not in text or "EUR" not in text:
        raise ValueError(f"{fname} [{sheet}] C7 = {text!r}: expected amounts in thousands of EUR")


def _row_values(rows, label, hdr):
    """(periods, values) of the row whose column C stripped equals ``label``, from column M on (None if absent)."""
    periods = list(rows[hdr][FIRST_COL:]) if len(rows) > hdr else []
    for r in rows:
        if len(r) > 2 and isinstance(r[2], str) and r[2].strip() == label:
            vals = [v if isinstance(v, (int, float)) and not isinstance(v, bool) else None for v in r[FIRST_COL:]]
            return periods, vals
    return None


def _newest(periods, vals, newest_last: bool):
    """Newest non-empty (value, period): right-most if columns run oldest→newest, else left-most."""
    pairs = [(v, p) for v, p in zip(vals, periods) if v is not None]
    if not pairs:
        return None
    return pairs[-1] if newest_last else pairs[0]


def _overview(rows, label):
    rv = _row_values(rows, label, OVERVIEW_HDR) if rows else None
    return _newest(*rv, newest_last=True) if rv else None


def _statement(rows, label):
    rv = _row_values(rows, label, STATEMENT_HDR) if rows else None
    return _newest(*rv, newest_last=False) if rv else None


def parse_pitchbook(path, turnover: float) -> dict:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ov, inc, bs = _rows(wb, "Overview"), _rows(wb, "Income Statement"), _rows(wb, "Balance Sheet")
    finally:
        wb.close()
    for name, rows in [("Overview", ov), ("Income Statement", inc), ("Balance Sheet", bs)]:
        if rows is not None:
            _check_units(rows, name, path.name)

    assets = _overview(ov, "Total Assets") or _statement(bs, "Total Assets")
    ebitda = _overview(ov, "EBITDA") or _statement(inc, "EBITDA")
    rev = _overview(ov, "Total Revenue")
    if rev:
        rule, source = "a", f"PitchBook Overview: Total Revenue {rev[1]}"
    else:
        rev = _statement(inc, "Revenue")
        if not rev and inc is not None:
            b = _row_values(inc, "Business Revenue", STATEMENT_HDR)
            o = _row_values(inc, "Other Revenue", STATEMENT_HDR)
            if b or o:
                periods = (b or o)[0]
                bv = b[1] if b else [None] * len(periods)
                ovv = o[1] if o else [None] * len(periods)
                summed = [None if (x is None and y is None) else (x or 0.0) + (y or 0.0) for x, y in zip(bv, ovv)]
                rev = _newest(periods, summed, newest_last=False)
        if rev:
            rule, source = "b", f"PitchBook Income Statement: Revenue {rev[1]}"
        elif assets:
            rev = (assets[0] * turnover, assets[1])
            rule, source = "c", f"PROXY: total assets x {turnover} (assumed)"
        else:
            raise ValueError(f"{path.name}: no revenue and no total assets")
    return {"revenue_keur": float(rev[0]), "period": rev[1], "revenue_rule": rule, "revenue_source": source,
            "total_assets_keur": float(assets[0]) if assets else np.nan,
            "ebitda_keur": float(ebitda[0]) if ebitda else np.nan, "file": path.name}


def load_supplier_financials(cfg: dict) -> pd.DataFrame:
    sc = cfg["suppliers"]
    rows = []
    if not sc.get("companies"):
        raise FileNotFoundError("no supplier list: copy config/suppliers.example.yaml to config/suppliers.local.yaml "
                                "(git-ignored, NDA) and fill it")
    for c in sc["companies"]:
        info = parse_pitchbook(find_file(c["file_keyword"]), float(sc["revenue_proxy_asset_turnover"]))
        rows.append({"name": c["name"], "country": c["country"], "country_status": c["country_status"],
                     "sector": c["sector"], "sector_status": c["sector_status"], **info})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- peers
def generate_peers(cfg: dict) -> pd.DataFrame:
    sc = cfg["suppliers"]
    rng = np.random.default_rng(int(sc["peer_seed"]))
    n = int(sc["n_peers"])
    sectors, countries = list(sc["sectors"]), list(sc["grid_kg_per_kwh"])
    sector = rng.choice(sectors, size=n)
    country = rng.choice(countries, size=n)
    rev_meur = np.exp(rng.uniform(np.log(5), np.log(500), size=n))
    rev_eur = rev_meur * 1e6

    def u(key):
        lo = np.array([sc["sectors"][s][key][0] for s in sector], dtype=float)
        hi = np.array([sc["sectors"][s][key][1] for s in sector], dtype=float)
        return rng.uniform(lo, hi)

    elec = rev_eur * u("elec")
    gas = rev_eur * u("gas")
    noise12 = rng.lognormal(0.0, 0.15, size=n)  # shared by S1 and S2
    grid = np.array([sc["grid_kg_per_kwh"][c] for c in country], dtype=float)
    scope1 = gas * float(sc["gas_kg_per_kwh"]) / 1000 * noise12
    scope2 = elec * grid / 1000 * noise12
    scope3 = rev_eur * u("purchased_share") * u("upstream_kg_per_eur") / 1000 * rng.lognormal(0.0, 0.25, size=n)
    reports = rng.uniform(size=n) < float(sc["share_peers_reporting_scope3"])
    scope3 = np.where(reports, scope3, np.nan)
    df = pd.DataFrame({"peer_id": np.arange(n), "sector": sector, "country": country, "revenue_meur": rev_meur,
                       "elec_kwh": elec, "gas_kwh": gas, "scope1_t": scope1, "scope2_t": scope2,
                       "s12_t": scope1 + scope2, "reports_s3": reports, "s3_t": scope3})
    df["s12_int"] = df["s12_t"] / df["revenue_meur"]  # t per MEUR
    df["s3_int"] = df["s3_t"] / df["revenue_meur"]
    df["source"] = "SYNTHETIC (demo)"
    return df


def predict(country: str, sector: str, revenue_keur: float, pool: pd.DataFrame, target: str, k: int) -> dict:
    """P10/P50/P90 (t) of the target intensity of the k nearest peers by size × the company's revenue."""
    col = f"{target}_int"
    ok = pool[pool[col].notna()]
    cand = ok[(ok["sector"] == sector) & (ok["country"] == country)]
    fallback = len(cand) < 5
    if fallback:
        cand = ok[ok["sector"] == sector]
    rev_meur = revenue_keur / 1000
    gap = (np.log(cand["revenue_meur"]) - np.log(rev_meur)).abs()
    near = cand.loc[gap.sort_values(kind="stable").index[:k]]
    q = np.percentile(near[col].to_numpy(), [10, 50, 90]) * rev_meur
    return {"n": int(len(near)), "sector_only": bool(fallback), "p10": float(q[0]), "p50": float(q[1]),
            "p90": float(q[2])}


def validate(peers: pd.DataFrame, k: int) -> pd.DataFrame:
    rows = []
    for target in ["s12", "s3"]:
        pool = peers[peers[f"{target}_int"].notna()]
        errs, inside = [], []
        for i, r in pool.iterrows():
            p = predict(r["country"], r["sector"], r["revenue_meur"] * 1000, pool.drop(index=i), target, k)
            actual = r[f"{target}_t"]
            errs.append(abs(p["p50"] - actual) / actual)
            inside.append(p["p10"] <= actual <= p["p90"])
        rows.append({"target": target, "n_peers": len(pool), "median_abs_pct_error": float(np.median(errs)),
                     "band_coverage_p10_p90": float(np.mean(inside))})
    return pd.DataFrame(rows)


def run(cfg: dict | None = None) -> dict:
    cfg = load_business_config() if cfg is None else cfg
    sc = cfg["suppliers"]
    k = int(sc["k_neighbours"])
    out = out_dir()
    fin = load_supplier_financials(cfg)
    for r in fin.itertuples():  # check against expected values (warn, never fail)
        exp, rule = sc.get("expected_keur", {}).get(r.name), sc.get("expected_rule", {}).get(r.name)
        if exp is not None and round(r.revenue_keur) != exp:
            print(f"WARNING {r.name}: revenue {r.revenue_keur:,.0f} kEUR, expected {exp:,}")
        if rule is not None and r.revenue_rule != rule:
            print(f"WARNING {r.name}: revenue rule {r.revenue_rule}, expected {rule}")
    fin.to_csv(out / "supplier_financials.csv", index=False)

    peers = generate_peers(cfg)
    peers.to_csv(out / "peers_synthetic.csv", index=False)
    val = validate(peers, k)
    val.to_csv(out / "supplier_validation.csv", index=False)

    rows = []
    for r in fin.itertuples():
        s12 = predict(r.country, r.sector, r.revenue_keur, peers, "s12", k)
        s3 = predict(r.country, r.sector, r.revenue_keur, peers, "s3", k)
        low = "to confirm" in (r.country_status, r.sector_status) or r.revenue_source.startswith("PROXY")
        rows.append({"name": r.name, "country": r.country, "country_status": r.country_status, "sector": r.sector,
                     "sector_status": r.sector_status, "revenue_keur": r.revenue_keur,
                     "revenue_source": r.revenue_source,
                     **{f"s12_{q}_t": s12[q] for q in ["p10", "p50", "p90"]}, "s12_n_peers": s12["n"],
                     "s12_sector_only": s12["sector_only"],
                     **{f"s3_{q}_t": s3[q] for q in ["p10", "p50", "p90"]}, "s3_n_peers": s3["n"],
                     "s3_sector_only": s3["sector_only"],
                     "spend_based_total_t": r.revenue_keur * float(sc["sectors"][r.sector]["spend_factor"]),
                     "confidence": "low" if low else "medium"})
    est = pd.DataFrame(rows)
    est["s123_p50_t"] = est["s12_p50_t"] + est["s3_p50_t"]
    est["priority_rank"] = est["s123_p50_t"].rank(ascending=False, method="first").astype(int)
    est = est.sort_values("priority_rank")
    est.to_csv(out / "supplier_estimates.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(est))
    for off, t, col in [(-0.2, "s12", "tab:blue"), (0.2, "s3", "tab:orange")]:
        p50 = est[f"{t}_p50_t"].to_numpy()
        err = [p50 - est[f"{t}_p10_t"].to_numpy(), est[f"{t}_p90_t"].to_numpy() - p50]
        ax.bar(x + off, p50, 0.4, yerr=err, capsize=3, color=col,
               label="Scope 1+2 (P50, P10-P90)" if t == "s12" else "Scope 3 cat. 1 (P50, P10-P90)")
    ax.set_xticks(x, est["name"], rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("t CO2e per year (whole supplier, log scale)")
    ax.set_title("Estimated supplier emissions from synthetic peers (demo)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir() / "supplier_estimates.png", dpi=150)
    plt.close(fig)

    v = val.set_index("target")
    return {"s12_loo_median_abs_error_pct": round(100 * v.at["s12", "median_abs_pct_error"], 1),
            "s12_loo_band_coverage_pct": round(100 * v.at["s12", "band_coverage_p10_p90"], 1),
            "s3_loo_median_abs_error_pct": round(100 * v.at["s3", "median_abs_pct_error"], 1),
            "s3_loo_band_coverage_pct": round(100 * v.at["s3", "band_coverage_p10_p90"], 1),
            "top3_suppliers_by_priority": ", ".join(est["name"].head(3)),
            "n_suppliers": int(len(est)), "n_low_confidence": int((est["confidence"] == "low").sum())}


if __name__ == "__main__":
    print(run())
