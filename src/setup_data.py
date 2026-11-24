"""One-off: copy the challenge files from the delivered layout (data/Data/) into the layout the code reads (data/raw/).

    uv run python -m src.setup_data

Copies only files missing at the destination; never overwrites, never deletes.
- Workbooks: from the TOP LEVEL of data/Data/ only. The RM_Prices workbook inside data.zip is the deprecated BOM
  (see data/Data/BOM_SOURCE_OF_TRUTH.md), so it is never copied.
- Panel CSVs: from data/Data/data_zip/**/ into data/raw/panel/.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from src.data import PANEL, RAW, ROOT

SRC = ROOT / "data" / "Data"
WORKBOOKS = ["RM_Prices_XHEC_Challenge.xlsx", "Copia de Europe Steel INDEXES.xlsx", "Epoxy for student.xlsx",
             "LME_Copper_Aluminium_EU_Monthly_2016_2026.xlsx", "RMU_IQI_EU_Price_Benchmark_2017-2026_Synthetic.xlsx"]


def plan_copies(src: Path = SRC, raw: Path = RAW, panel: Path = PANEL) -> list[tuple[Path, Path]]:
    pairs = [(src / f, raw / f) for f in WORKBOOKS if (src / f).exists()]
    pairs += [(p, panel / p.name) for p in sorted(src.glob("data_zip/**/*.csv"))]
    return [(a, b) for a, b in pairs if not b.exists()]


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
