"""One-off: copy the challenge files from the delivered layouts into the layout the code reads (data/raw/).

    uv run python -m src.setup_data

Copies only files missing at the destination; never overwrites, never deletes.
- Workbooks: from the TOP LEVEL of data/Data/ or data/. The RM_Prices workbook inside data.zip is the deprecated BOM
  (see data/Data/BOM_SOURCE_OF_TRUTH.md), so it is never copied (data/raw/data/ is not searched for workbooks).
- Panel CSVs: from data/Data/data_zip/**/ or data/raw/data/ (data.zip extracted there) into data/raw/panel/.
- Supplier exports: from data/raw/Supplier_balancesheet/ into data/raw/suppliers/.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from src.data import PANEL, RAW, ROOT

SRC = ROOT / "data" / "Data"
DATA = ROOT / "data"
WORKBOOKS = ["RM_Prices_XHEC_Challenge_v1.xlsx", "RM_Prices_XHEC_Challenge.xlsx", "Copia de Europe Steel INDEXES.xlsx", "Epoxy for student.xlsx",
             "LME_Copper_Aluminium_EU_Monthly_2016_2026.xlsx", "RMU_IQI_EU_Price_Benchmark_2017-2026_Synthetic.xlsx", "competitor_teardown_rm6_class.xlsx"]
PATTERNS = ["benchmark-modules-fonctionnels-rm6*.xlsx"]  # teardown workbook under its original name


def plan_copies(src: Path = SRC, raw: Path = RAW, panel: Path = PANEL, data: Path = DATA) -> list[tuple[Path, Path]]:
    """(source, destination) of every file missing at its destination; the first source found wins."""
    pairs: list[tuple[Path, Path]] = []
    for folder in (src, data):
        pairs += [(folder / f, raw / f) for f in WORKBOOKS if (folder / f).exists()]
        pairs += [(p, raw / p.name) for pat in PATTERNS for p in sorted(folder.glob(pat))]
    pairs += [(p, panel / p.name) for p in sorted(src.glob("data_zip/**/*.csv")) + sorted((raw / "data").glob("*.csv"))]
    pairs += [(p, raw / "suppliers" / p.name) for p in sorted((raw / "Supplier_balancesheet").glob("PitchBook_*.xlsx"))]
    out: dict[Path, Path] = {}
    for a, b in pairs:
        if not b.exists():
            out.setdefault(b, a)
    return [(a, b) for b, a in out.items()]


def main() -> None:
    todo = plan_copies()
    if not todo:
        print("nothing to copy")
    for a, b in todo:
        b.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a, b)
        print(f"copied {a.relative_to(ROOT)} -> {b.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
