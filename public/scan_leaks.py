"""NDA leak scanner for everything that may be pushed.

Usage:  uv run python public/scan_leaks.py [paths ...]      (default: public/ README.md src/)
Lines containing "nda-ok" are skipped (reviewed false positives).
Exit code 1 if any private term, private number or structural problem is found.

The deny-lists are NOT written in this file (that would leak them). They are built at run time from
the git-ignored NDA data:
- data/processed/*.parquet  supplier names and financials, plant names, ERP systems, part references,
                            BOM masses, competitor should-costs, price benchmark, CRU / MEPS / epoxy levels;
- data/RM_Prices_XHEC_Challenge_v1.xlsx  latest BOM masses;
- data/nda_denylist.txt     extra terms and numbers (presenters, € figures quoted in private reports).
So the scanner only works on a machine that has the NDA data, which is where pushes are made from.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROC = DATA / "processed"
PUBLIC = ROOT / "public"
FIG_DIR = PUBLIC / "reports" / "figures"

# Figures allowed in the public reports (names only, no data).
ALLOWED_FIGS = {
    "00_rm_rebased", "00_rm_missingness", "00_energy_shock", "00_lme_spread", "00_stainless_alloy_share",
    "00_epoxy_typo_pct", "01_missingness", "02_rm_indexed", "03_rolling_vol", "04_momentum", "05_corr_returns",
    "06_energy_ccf", "07_steel_market_vs_proxy", "08_bom_cost_index", "11_baseline_mape", "12_leaderboard_heatmap",
    "13_interval_coverage", "14_dynreg_s09_coefs", "15_forecast_fans_pct", "16_price_vs_cost_index",
    "17_corr_levels", "18_corr_dlog_pearson", "18b_corr_dlog_spearman", "19_corr_family_blocks",
    "20_rolling_corr", "21_leadlag_ccf", "22_feature_target_rho", "23_feature_selection",
}
TEXT_EXT = {".md", ".py", ".html", ".ipynb", ".txt", ".yaml", ".yml", ".toml", ".csv", ".json"}


# --------------------------------------------------------------------------------------------- deny-lists
def _num_variants(v: float) -> set[str]:
    """Spellings of a private number as it may appear in text."""
    out: set[str] = set()
    if v != v or abs(v) < 10:  # NaN or too small to be distinctive
        return out
    r = round(v)
    if abs(v - r) < 1e-9 or abs(v) >= 500:
        # years and round numbers are too common in normal text to be evidence of a leak
        if abs(r) >= 100 and not (1990 <= r <= 2035) and r % 100 != 0:
            out |= {f"{r}", f"{r:,}", f"{r:,}".replace(",", " "), f"{r:,}".replace(",", " ")}
    else:
        for d in (1, 2):
            s = f"{v:.{d}f}"
            if float(s) == round(v, d) and len(s.replace(".", "").lstrip("0")) >= 3:
                out.add(s)
    return out


def build_denylists() -> tuple[set[str], set[str], set[str], set[str]]:
    """Returns (terms, numbers, kg_numbers, eur_m_numbers). Small masses and €m amounts are too common as plain
    numbers, so they are only flagged next to "kg" or next to "€" / "m" respectively."""
    terms: set[str] = set()
    nums: set[str] = set()
    kg: set[str] = set()
    eur_m: set[str] = set()

    def add_mass(v: float) -> None:
        (nums if abs(v) >= 100 else kg).update(_num_variants(v))

    k = pd.read_parquet(PROC / "supplier_kpis.parquet")
    for name in k.supplier.dropna():
        terms.add(name)
        terms.add(name.split(" (")[0])
    for col in k.select_dtypes("number").columns:
        if col.endswith("_keur"):
            for v in k[col].dropna():
                nums |= _num_variants(v)
                if abs(v) >= 10_000:  # also as €m with 1 decimal (>= 3 significant digits)
                    eur_m.add(f"{v / 1000:.1f}")

    bom = pd.read_parquet(PROC / "bom.parquet")
    terms |= set(bom.plant_name.dropna()) | set(bom.plant_code.dropna())
    for v in bom.mass_kg.dropna():
        add_mass(v)
    v1 = DATA / "RM_Prices_XHEC_Challenge_v1.xlsx"
    if v1.exists():
        b1 = pd.read_excel(v1, sheet_name="BOM_Materials")
        for v in pd.to_numeric(b1.mass_kg, errors="coerce").dropna():
            add_mass(v)

    mp = pd.read_parquet(PROC / "steel_index_mapping.parquet")
    terms |= {s for s in mp.erp_code.dropna().unique() if s.startswith("SAP")}
    terms |= {s for s in mp.plant_name.dropna().unique() if len(s) > 6}
    terms |= {str(s) for s in mp.reference.dropna().unique() if len(str(s)) >= 6}

    cb = pd.read_parquet(PROC / "competitor_benchmark.parquet")
    for col in ("total", "material", "process", "labour", "overhead", "list_price"):
        for v in cb[col].dropna():
            (nums if abs(v) >= 100 else eur_m).update(_num_variants(v))  # small € amounts: only next to €

    pb = pd.read_parquet(PROC / "price_benchmark.parquet")
    for col in ("p10", "p50", "p90"):
        for v in pb[col].dropna():
            nums |= _num_variants(v)

    panel = pd.read_parquet(PROC / "panel_monthly.parquet")
    cat = pd.read_parquet(PROC / "series_catalog.parquet")
    lic = cat.index[cat.block.isin(["cru_flat", "meps", "cru_stainless", "cru_long", "epoxy"])]
    recent = panel[[c for c in lic if c in panel]].iloc[-24:]
    for v in recent.stack().dropna():
        if abs(v) >= 500:
            nums |= _num_variants(v)

    extra = DATA / "nda_denylist.txt"
    if extra.exists():
        for line in extra.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("num:"):
                nums |= _num_variants(float(line[4:]))
            else:
                terms.add(line)
    return {t for t in terms if t and len(t) >= 3}, nums - kg - eur_m, kg, eur_m


# --------------------------------------------------------------------------------------------- extraction
def file_text(path: Path) -> str:
    if path.suffix == ".pdf":
        from pypdf import PdfReader  # optional dependency, checked in scan()
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix == ".ipynb":
        nb = json.loads(raw)
        return "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    if path.suffix == ".html":
        raw = re.sub(r"data:[a-z]+/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+", "", raw)
    return raw


def structural_problems(path: Path) -> list[str]:
    probs = []
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        n = sum(len(c.get("outputs", [])) for c in nb["cells"] if c["cell_type"] == "code")
        if n:
            probs.append(f"notebook has {n} outputs (clear them before pushing)")
    if path.suffix == ".html" and PUBLIC in path.parents:
        for src in re.findall(r'<img[^>]+src="([^"]+)"', path.read_text(encoding="utf-8", errors="ignore")):
            if src.startswith("data:"):
                continue
            name = Path(src).stem
            if name not in ALLOWED_FIGS:
                probs.append(f"image not on the allow-list: {src}")
    if path.suffix == ".png" and path.parent == FIG_DIR and path.stem not in ALLOWED_FIGS:
        probs.append("figure not on the allow-list")
    return probs


def scan(paths: list[Path]) -> int:
    terms, nums, kg, eur_m = build_denylists()
    em = "|".join(re.escape(n) for n in sorted(eur_m, key=len, reverse=True))
    em_re = re.compile(rf"€\s?({em})(?![\d])|(?<![\w.,])({em})\s?(?:€\s?m|m€|m\b|M\b|million)")
    term_re = re.compile("|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)), re.I)
    # PDF text extraction glues adjacent table cells ("2" + "118" -> "2 118"), so in PDFs the space-separated
    # spellings are skipped; the HTML the PDFs are rendered from is scanned with every spelling.
    pdf_nums = {n for n in nums if " " not in n and "\u202f" not in n}
    num_re_pdf = re.compile(r"(?<![\w.,])(" + "|".join(re.escape(n) for n in sorted(pdf_nums, key=len, reverse=True))
                            + r")(?![\w]|[.,]\d)")
    kg_re = re.compile(r"(?<![\w.,])(" + "|".join(re.escape(n) for n in sorted(kg, key=len, reverse=True))
                       + r")\s?kg\b", re.I)
    num_re = re.compile(r"(?<![\w.,])(" + "|".join(re.escape(n) for n in sorted(nums, key=len, reverse=True))
                        + r")(?![\w]|[.,]\d)")
    files = []
    for p in paths:
        p = p.resolve()
        files += [f for f in p.rglob("*") if f.is_file()] if p.is_dir() else [p]
    files = [f for f in files if "__pycache__" not in f.parts and ".ipynb_checkpoints" not in f.parts]

    try:
        import pypdf  # noqa: F401
        text_ext = TEXT_EXT | {".pdf"}
    except ImportError:
        text_ext = TEXT_EXT
        if any(f.suffix == ".pdf" for f in files):
            print("note: pypdf not installed, PDFs not scanned (they are rendered from the scanned HTML). "
                  "Full check: uv run --with pypdf python public/scan_leaks.py")
    n_hits = 0
    for f in sorted(files):
        rel = f.relative_to(ROOT) if ROOT in f.parents else f
        if f.name == "scan_leaks.py":
            continue
        for prob in structural_problems(f):
            print(f"{rel}: {prob}")
            n_hits += 1
        if f.suffix not in text_ext:
            continue
        for i, line in enumerate(file_text(f).splitlines(), 1):
            if "nda-ok" in line:  # reviewed false positive (e.g. a MAPE that equals a private number)
                continue
            for m in term_re.finditer(line):
                print(f"{rel}:{i}: private term '{m.group(0)}'")
                n_hits += 1
            nre = num_re_pdf if f.suffix == ".pdf" else num_re
            for m in [*nre.finditer(line), *kg_re.finditer(line), *em_re.finditer(line)]:
                print(f"{rel}:{i}: private number '{m.group(0)}'  …{line[max(0, m.start() - 40):m.end() + 40].strip()}…")
                n_hits += 1
    print(f"\n{len(files)} files scanned, {len(terms)} terms + {len(nums) + len(kg) + len(eur_m)} numbers on the deny-list: "
          f"{'NO LEAK FOUND' if n_hits == 0 else f'{n_hits} PROBLEM(S)'}")
    return 1 if n_hits else 0


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]] or [PUBLIC, ROOT / "README.md", ROOT / "src"]
    sys.exit(scan(args))
