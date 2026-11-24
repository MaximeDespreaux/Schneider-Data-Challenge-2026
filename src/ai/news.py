"""Phase 3: monthly news-tension signal per material, from public GDELT headlines scored by an LLM.

1. ``fetch``: English headlines per material and month from the GDELT DOC 2.0 API (public, free; one request every
   6 s as GDELT asks). Only generic public search terms are sent; nothing about the project.
2. ``tone``: baseline without an LLM, GDELT's own average tone per material and month.
3. ``score``: the LLM reads one month's headlines for every material at once and gives a supply/demand tension score
   from -2 (strong downward price pressure) to +2 (strong upward pressure). The prompt has no date, so the model must
   read the headlines rather than recall what prices did next.
4. Evaluation (``evaluate``): both signals go through the same pre-2024 screen as the agent's features and into the
   same dynreg comparison on 2024-26; results are also split at the LLM's release date (training contamination).

    uv run --group ai python -m src.ai.news fetch      # ~70 min, network only
    uv run --group ai python -m src.ai.news score      # LLM, ~1 call per month
    uv run --group ai python -m src.ai.news evaluate
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from . import config as C
from . import llm

DIR = C.ROOT / "data" / "external" / "gdelt"
API = "https://api.gdeltproject.org/api/v2/doc/doc"
START, PAUSE = "2017-01-01", 6.0
QUERIES = {
    "steel": '("steel prices" OR "steel price" OR steelmakers OR "steel demand" OR "steel output")',
    "stainless_nickel": '(nickel OR "stainless steel")',
    "copper": '(copper price OR copper prices OR copper mine OR copper demand)',
    "aluminium": '(aluminium OR aluminum)',
    "iron_ore": '"iron ore"',
    "plastics": '(polyethylene OR polypropylene OR petrochemicals OR "plastics prices")',
}
# Target series each material signal is tested on (public and licensed targets alike: the signal is computed
# from public text, the targets stay local).
MATERIAL_TARGETS = {
    "steel": ["CRU_CR_DE", "CRU_CR_FR", "CRU_HDG_DE", "S09", "S10"],
    "stainless_nickel": ["CRU_SS304_DE", "SS304_ALLOY_DE", "S03"],
    "copper": ["D01"], "aluminium": ["D02"], "iron_ore": ["S04"], "plastics": ["S11"],
}


def _get(params: dict) -> dict | None:
    url = f"{API}?{urllib.parse.urlencode(params)}"
    for attempt in range(4):
        time.sleep(PAUSE * (1 + attempt))
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-client"}),
                                        timeout=90) as r:
                body = r.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001  (network hiccup: retry with a longer pause)
            continue
        if body.startswith("Please limit"):
            continue
        try:
            return json.loads(body) if body.strip() else {}
        except json.JSONDecodeError:
            return {}
    return None


def _months(start: str = START) -> pd.DatetimeIndex:
    return pd.date_range(start, pd.Timestamp.today().normalize().replace(day=1), freq="MS")


def fetch(max_records: int = 75) -> None:
    """Headlines per material and month (resumable: months already on disk are skipped)."""
    DIR.mkdir(parents=True, exist_ok=True)
    for mat, q in QUERIES.items():
        path = DIR / f"{mat}.parquet"
        have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["month", "title", "domain", "seendate"])
        done = set(pd.to_datetime(have.month))
        rows = []
        for m in _months():
            if m in done:
                continue
            end = m + pd.DateOffset(months=1) - pd.Timedelta(seconds=1)
            data = _get({"query": f"{q} sourcelang:english", "mode": "artlist", "format": "json",
                         "maxrecords": max_records, "startdatetime": m.strftime("%Y%m%d%H%M%S"),
                         "enddatetime": end.strftime("%Y%m%d%H%M%S")})
            if data is None:
                print(f"  {mat} {m:%Y-%m}: failed, will retry on the next run", flush=True)
                continue
            arts = data.get("articles", [])
            rows += [{"month": m, "title": a.get("title", "").strip(), "domain": a.get("domain"),
                      "seendate": a.get("seendate")} for a in arts] or [{"month": m, "title": None}]
            if len(rows) and m.month == 12:
                print(f"  {mat} {m:%Y}: {sum(1 for r in rows if r['title'])} headlines so far", flush=True)
        out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        out["month"] = pd.to_datetime(out.month)
        out.to_parquet(path)
        print(f"{mat}: {out.title.notna().sum()} headlines, {out.month.nunique()} months", flush=True)


def tone() -> pd.DataFrame:
    """GDELT average tone per material and month (the no-LLM baseline)."""
    path = DIR / "tone.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frames = {}
    for mat, q in QUERIES.items():
        parts = []
        for y in range(int(START[:4]), pd.Timestamp.today().year + 1):
            data = _get({"query": f"{q} sourcelang:english", "mode": "timelinetone", "format": "json",
                         "startdatetime": f"{y}0101000000", "enddatetime": f"{y}1231235959"})
            for s in (data or {}).get("timeline", []):
                parts += [(pd.Timestamp(d["date"][:8]), d["value"]) for d in s.get("data", [])]
        d = pd.Series(dict(parts)).sort_index() if parts else pd.Series(dtype=float)
        frames[mat] = d.resample("MS").mean()
        print(f"tone {mat}: {frames[mat].notna().sum()} months", flush=True)
    out = pd.DataFrame(frames)
    out.to_parquet(path)
    return out


# --------------------------------------------------------------------------------------------- LLM scoring
SCHEMA = {"type": "object", "properties": {m: {"type": "object", "properties": {
    "score": {"type": "integer", "enum": [-2, -1, 0, 1, 2]}, "reason": {"type": "string"}},
    "required": ["score", "reason"], "additionalProperties": False} for m in QUERIES},
    "required": list(QUERIES), "additionalProperties": False}
SYSTEM = ("You are a commodity market analyst. From news headlines only, judge the direction of price pressure over "
          "the next few months for each material: -2 strong downward, -1 mild downward, 0 neutral or mixed, "
          "+1 mild upward, +2 strong upward. Consider supply disruptions, demand, inventories, trade measures, energy "
          "costs and capacity changes. Use only the headlines given; if they are irrelevant or few, answer 0.")


def _prompt(month_titles: dict[str, list[str]], k: int = 25) -> str:
    parts = []
    for mat, titles in month_titles.items():
        lines = "\n".join(f"- {t}" for t in titles[:k]) or "- (no headlines)"
        parts.append(f"## {mat.replace('_', ' / ')}\n{lines}")
    return "Headlines from one month (date withheld):\n\n" + "\n\n".join(parts)


def score(limit: int | None = None) -> pd.DataFrame:
    path = DIR / "llm_scores.parquet"
    heads = {m: pd.read_parquet(DIR / f"{m}.parquet") for m in QUERIES}
    months = sorted(set().union(*[set(h.month) for h in heads.values()]))[:limit]
    rows = []
    for i, m in enumerate(months):
        titles = {mat: h[(h.month == m) & h.title.notna()].title.drop_duplicates().tolist() for mat, h in heads.items()}
        prompt = _prompt(titles)
        # GDELT titles are public, but they still go through the gate: a match would be a coincidence to review.
        try:
            out = llm.ask(prompt, system=SYSTEM, schema=SCHEMA, tag=f"news/{m:%Y-%m}", max_tokens=4000)
        except Exception as e:  # noqa: BLE001
            print(f"  {m:%Y-%m}: skipped ({type(e).__name__}: {str(e)[:120]})", flush=True)
            continue
        rows.append({"month": m, **{mat: out[mat]["score"] for mat in QUERIES},
                     **{f"n_{mat}": len(titles[mat]) for mat in QUERIES}})
        if i % 12 == 0:
            print(f"  scored {m:%Y-%m} ({i + 1}/{len(months)})", flush=True)
    res = pd.DataFrame(rows).set_index("month")
    res.to_parquet(path)
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if cmd == "fetch":
        fetch()
        tone()
    elif cmd == "score":
        print(score().describe().round(2))
    elif cmd == "evaluate":
        from .news_eval import main
        main()
