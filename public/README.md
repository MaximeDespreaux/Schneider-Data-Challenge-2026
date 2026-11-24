# `public/`: NDA-safe copies

The challenge data is under an **NDA with Schneider Electric** and includes licensed CRU / MEPS prices. The original notebooks, reports and overview contain private values, so they are **git-ignored** (`/notebooks/`, `/reports/`, `/DATA_OVERVIEW.md`, `data/*`). This folder holds the copies that **may be pushed**.

| Path | What it is |
|---|---|
| `DATA_OVERVIEW.md` | Sanitised data overview and first results |
| `notebooks/` | The four analysis notebooks: **outputs cleared**, private values removed from the text, figure saving disabled |
| `reports/*.html`, `*.pdf` | Public versions of the four jury reports |
| `reports/figures/` | Allow-listed figures only (public data, rebased indices, % changes, statistics) |
| `reports/make_public_figures.py` | Copies the safe figures and redraws the sensitive ones as indices / % changes |
| `reports/build_public_reports.py` | Builds the public reports (sanitised copy of the private builder) |
| `scan_leaks.py` | Leak scanner: run it before every push |

## What is never shown

- CRU / MEPS / epoxy prices in EUR/t, or anything they can be rebuilt from.
- BOM masses and shares, € product costs, teardown should-costs, price-benchmark € levels.
- Supplier names and financials (suppliers become *Supplier A–I* with qualitative flags), plant codes, ERP systems, part references and mapping counts.
- Personal data (kick-off deck presenters).

Kept: public-source values (IMF, ECB, Eurostat, FRED, BLS, energy), rebased indices, % changes, correlations, error metrics and feature rankings.

## Workflow

```bash
# 1. work privately in notebooks/ and reports/ (ignored), with the NDA data in data/
# 2. rebuild the public copies
uv run python public/reports/make_public_figures.py
uv run python public/reports/build_public_reports.py
# 3. check: must print "NO LEAK FOUND"
uv run python public/scan_leaks.py
```

Editing a public notebook: run it locally if you like, but **clear its outputs before committing** (`uv run jupyter nbconvert --clear-output --inplace public/notebooks/*.ipynb`). The scanner fails on any notebook output.

The scanner builds its deny-list at run time from the NDA data (`data/processed/`, the latest BOM, and the git-ignored `data/nda_denylist.txt`), so no private value is written in the scanner itself. It only works on a machine that has the data. Mark a reviewed false positive with `nda-ok` on the same line.
