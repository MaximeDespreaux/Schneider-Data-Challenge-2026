"""Check that the local data folder and config are laid out as the code expects.

    uv run python -m src.checks.check_data_layout        (or: make data-check)

Only file NAMES are checked, with two exceptions: the sheet list of RM_Prices_XHEC_Challenge_v1.xlsx (must contain
BOM_Materials) and config/assumptions.yaml (no eur_per_kg_at_base value may still be null). No data value is read
or printed. Exit code 1 if any REQUIRED item is missing, with the exact fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from src import data as d

ROOT = d.ROOT
RAW = d.RAW
PANEL_FILES = ["rm_prices_monthly_wide.csv", "rm_prices_monthly_long.csv", "energy_prices_monthly_wide.csv",
               "energy_prices_monthly_long.csv", "source_catalog.csv", "energy_source_catalog.csv",
               "data_dictionary.csv", "bom_materials.csv"]


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def check() -> tuple[list[dict], list[str]]:
    rows, fixes = [], []

    def add(path: str, required: bool, found: bool, what: str, fix: str = ""):
        rows.append({"path": path, "required": "required" if required else "optional",
                     "found": "yes" if found else "no", "what": what})
        if required and not found:
            fixes.append(f"{path}: {fix}")

    # challenge panel (data.zip)
    for f in PANEL_FILES:
        p = d.PANEL / f
        add(rel(p), True, p.exists(), "challenge panel (data.zip)",
            f"unzip the challenge data.zip and put {f} in {rel(d.PANEL)}/")

    # official BOM v2 (+ sheet name)
    bom = RAW / "RM_Prices_XHEC_Challenge_v1.xlsx"
    ok_sheet = False
    if bom.exists():
        import openpyxl
        wb = openpyxl.load_workbook(bom, read_only=True)
        ok_sheet = "BOM_Materials" in wb.sheetnames
        wb.close()
    add(rel(bom), True, bom.exists() and ok_sheet,
        "official BOM v2 (Schneider, ~20 Nov), sheet BOM_Materials" + ("" if ok_sheet or not bom.exists()
                                                                        else " — SHEET MISSING"),
        f"put the BOM workbook sent by Schneider (~20 Nov) in {rel(RAW)}/ named RM_Prices_XHEC_Challenge_v1.xlsx; "
        "it must contain a sheet named BOM_Materials")

    # epoxy (two accepted names)
    ep = [RAW / "Epoxy for student.xlsx", RAW / "Epoxy_for_student.xlsx"]
    found = next((p for p in ep if p.exists()), None)
    add(rel(found) if found else "data/raw/Epoxy for student.xlsx", True, found is not None,
        "epoxy resin price (Schneider follow-up file)",
        f"put the epoxy file in {rel(RAW)}/ as 'Epoxy for student.xlsx' (or 'Epoxy_for_student.xlsx')")

    for name, what in [("LME_Copper_Aluminium_EU_Monthly_2016_2026.xlsx", "LME term structure and stocks (challenge)"),
                       ("RMU_IQI_EU_Price_Benchmark_2017-2026_Synthetic.xlsx", "synthetic RMU price benchmark (challenge)")]:
        p = RAW / name
        add(rel(p), True, p.exists(), what, f"put the file in {rel(RAW)}/ named exactly {name}")
    from src.licensed import FILE_NAMES
    st = next((RAW / n for n in FILE_NAMES if (RAW / n).exists()), None)
    add(rel(st) if st else f"data/raw/{FILE_NAMES[0]}", True, st is not None,
        "LICENSED MEPS/CRU steel prices — never commit", f"put the licensed steel workbook in {rel(RAW)}/ as {FILE_NAMES[0]!r}")

    # competitor teardowns (renamed or original name)
    td = RAW / "competitor_teardown_rm6_class.xlsx"
    alt = sorted(RAW.glob("benchmark-modules-fonctionnels-rm6*.xlsx"))
    found = td if td.exists() else (alt[0] if alt else None)
    add(rel(found) if found else rel(td), True, found is not None, "competitor teardowns (outside agency)",
        f"put the teardown workbook in {rel(RAW)}/ as competitor_teardown_rm6_class.xlsx "
        "(the original benchmark-modules-fonctionnels-rm6-...-sf6.xlsx name is also accepted)")

    # supplier PitchBook files, one per company in config/business.yaml
    from src.business.suppliers import _norm  # same matching rule as the parser (punctuation / accents ignored)
    sup_dir = RAW / "suppliers"
    files = sorted(sup_dir.glob("PitchBook_-_*.xlsx")) if sup_dir.exists() else []
    with open(ROOT / "config" / "business.yaml", encoding="utf-8") as f:
        companies = yaml.safe_load(f)["suppliers"]["companies"]
    for c in companies:
        hits = [p for p in files if _norm(c["file_keyword"]) in _norm(p.name)]
        add(f"data/raw/suppliers/PitchBook_-_*{c['file_keyword']}*.xlsx", True, len(hits) >= 1,
            f"PitchBook financials: {c['name']}" + (f" ({len(hits)} files)" if len(hits) > 1 else ""),
            f"export '{c['name']}' from PitchBook and put the PitchBook_-_..._Company_Financials_....xlsx file in "
            f"{rel(sup_dir)}/ (its name must contain '{c['file_keyword']}')")

    # local assumptions
    cfg = ROOT / "config" / "assumptions.yaml"
    nulls = []
    if cfg.exists():
        with open(cfg, encoding="utf-8") as f:
            a = yaml.safe_load(f) or {}
        nulls = [k for k, v in (a.get("eur_per_kg_at_base") or {}).items() if (v or {}).get("value") is None]
    add(rel(cfg), True, cfg.exists() and not nulls,
        "local assumptions (gitignored)" + (f" — still null: {', '.join(nulls)}" if nulls else ""),
        ("copy config/assumptions.example.yaml to config/assumptions.yaml and fill every FILL_LOCALLY value"
         if not cfg.exists() else f"fill eur_per_kg_at_base values still null: {', '.join(nulls)} (see docs/SETUP.md)"))

    # optional
    p = RAW / "RM_Prices_XHEC_Challenge.xlsx"
    add(rel(p), False, p.exists(), "previous BOM (sensitivity only)")
    for f in ["fred_monthly.csv", "eurostat_ppi_long.csv", "gscpi.csv"]:
        p = RAW / "external" / f
        add(rel(p), False, p.exists(), "public data, created by `make fetch`")
    return rows, fixes


def main() -> None:
    rows, fixes = check()
    w = max(len(r["path"]) for r in rows)
    print(f"{'path':<{w}} | required/optional | found | what it is")
    print("-" * (w + 60))
    for r in rows:
        print(f"{r['path']:<{w}} | {r['required']:<17} | {r['found']:<5} | {r['what']}")
    missing_opt = [r["path"] for r in rows if r["required"] == "optional" and r["found"] == "no"]
    if missing_opt:
        print(f"\nOptional items missing ({len(missing_opt)}): run `make fetch` for data/raw/external/.")
    if fixes:
        print(f"\nMISSING REQUIRED ITEMS ({len(fixes)}):")
        for f in fixes:
            print(f"  - {f}")
        sys.exit(1)
    print(f"\nDATA LAYOUT OK: all {sum(r['required'] == 'required' for r in rows)} required items found.")


if __name__ == "__main__":
    main()
