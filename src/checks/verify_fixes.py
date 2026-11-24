"""Vérification locale des correctifs de données, sur les VRAIES données (data/raw/, config/assumptions.yaml).

    uv run python -m src.checks.verify_fixes

Ne tourne pas en CI (data/ vide). Lance un seul backtest court (2 modèles naïfs, origines 2025-01 → 2026-09).
Sorties (confidentielles, gitignorées) : data/outputs/business/ et data/outputs/checks/.
"""

from __future__ import annotations

import copy

import pandas as pd

from src import data as d
from src.testbench import metrics
from src.testbench.backtest import compute_footprint, run_backtest
from src.testbench.models import Naive, NaiveVolScaled
from src.testbench.run import bom_deriver, build_panel
from src.testbench.targets import TARGET_IDS, add_bom_indices, bom_weights, coverage_table, load_assumptions

BASE = pd.Timestamp("2026-09-01")
FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAILURES.append(msg)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def nonnull_between(s: pd.Series, start: str, end: str) -> bool:
    return bool(s.loc[start:end].notna().all()) and len(s.loc[start:end]) == len(pd.date_range(start, end, freq="MS"))


EXPECTED = d.ROOT / "data" / "verify_expected.yaml"  # gitignored: licensed / challenge values never in git


def _expected() -> dict:
    import yaml
    if not EXPECTED.exists():
        raise SystemExit(f"{EXPECTED} missing: this check compares against local (licensed) values kept out of git")
    return yaml.safe_load(EXPECTED.read_text(encoding="utf-8"))


def main() -> None:
    E = _expected()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    # 1. panel -------------------------------------------------------------------------------------------------
    section("1. Panel : nouvelles séries")
    panel = build_panel()
    for c in ["L01", "L03"]:
        check(c in panel and nonnull_between(panel[c], "2016-11", "2026-09"), f"{c} non vide de 2016-11 à 2026-09")
    check("L02" in panel and nonnull_between(panel["L02"], "2016-11", "2026-08"), "L02 non vide de 2016-11 à 2026-08")
    check(panel.at[BASE, "L01"] == E["L01_base"], "L01[2026-09] == expected value (local file)")
    check(panel.at[BASE, "L03"] == E["L03_base"], "L03[2026-09] == expected value (local file)")
    l02 = panel.at[pd.Timestamp("2026-08-01"), "L02"]
    check(l02 == E["L02_2026_08"], "L02[2026-08] == expected value (local file)")
    check("EP01" in panel and nonnull_between(panel["EP01"], "2018-01", "2026-09"), "EP01 non vide de 2018-01 à 2026-09")
    ep = panel.at[BASE, "EP01"]
    check(abs(ep - E["EP01_base"]) < 1, "|EP01[2026-09] - expected| < 1 (local file)")
    if "EL01" in panel and panel["EL01"].notna().any():
        el = panel["EL01"]
        check(int(el.notna().sum()) >= 100, f"EL01 : au moins 100 mois (lu : {int(el.notna().sum())})")
        recent = panel.index[-4:]
        check(bool(el.loc[recent].notna().any()),
              f"EL01 a une valeur dans les 4 derniers mois du panel (dernière : {el.last_valid_index():%Y-%m})")
    else:
        print("  [WARN] EL01 absent : lancer `uv run python -m src.external.fetch_public` pour télécharger l'indice "
              "des prix de l'électronique (Eurostat C26.1).")

    # 2. poids BOM ---------------------------------------------------------------------------------------------
    section("2. Poids en valeur de la BOM (config/assumptions.yaml)")
    a = load_assumptions()
    check(a["bom"]["version"] == "xlsx_v2", f"version de BOM = xlsx_v2 (lu : {a['bom']['version']})")
    w = bom_weights(a)
    cols = ["material", "series", "kg", "eur_per_kg", "value_eur", "weight"]
    for prod in ["RM6", "AirSeT"]:
        print(f"\n{prod}")
        print(w[w["product"] == prod][cols].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    for prod in ["RM6", "AirSeT"]:
        has_el = bool(((w["product"] == prod) & (w["material"] == "electronics")).any())
        check(not has_el, f"{prod} : pas de ligne électronique dans l'indice BOM (absente de la BOM Schneider)")
    print("\nPour information : part de l'électronique dans le poids de l'indice si on l'ajoutait "
          "(EUR/boîte ÷ (valeur matières pricée + EUR/boîte))")
    boxes = {"RM6": [20, 50, 150], "AirSeT": [100, 300, 800]}
    value = w.groupby("product")["value_eur"].sum()
    print(pd.DataFrame([{"product": p, "eur_per_box": b, "priced_material_value_eur": value[p],
                         "electronics_share": b / (value[p] + b)} for p, bs in boxes.items() for b in bs])
          .to_string(index=False, formatters={"priced_material_value_eur": "{:,.1f}".format,
                                              "electronics_share": "{:.1%}".format}))
    for prod, s in w.groupby("product")["weight"].sum().items():
        check(abs(s - 1) < 1e-9, f"{prod} : somme des poids = 1 (lu : {s:.12f})")
    kg = w.set_index(["product", "material"])["kg"]
    expected = {tuple(k.split("|")): v for k, v in E["bom_kg"].items()}
    for (prod, mat), v in expected.items():
        got = kg.get((prod, mat), float("nan"))
        check(abs(got - v) <= 1e-3 + 1e-9, f"{prod} {mat} kg == expected (local file)")
    ser = w.set_index("material")["series"]
    check(set(ser[ser.index == "epoxy"]) == {"EP01"}, "lignes epoxy -> série EP01")
    check(set(ser[ser.index == "brass"]) == {"BR01"}, "lignes laiton -> série BR01")
    dup = w.groupby(["product", "material"]).size()
    check(int(dup.max()) == 1, "aucune ligne matière dupliquée (epoxy…) par produit")
    rm6_total = d.load_bom("v2")
    rm6_total = float(rm6_total.loc[rm6_total["product"] == "RM6", "implied_total_kg"].iloc[0])
    check(abs(rm6_total - E["rm6_total_kg"]) <= 2, "RM6 implied total mass == expected (local file)")

    a1 = copy.deepcopy(a)
    a1["bom"]["version"] = "xlsx_v1"
    w1 = bom_weights(a1)
    side = w.merge(w1, on=["product", "material", "series"], how="outer", suffixes=("_v2", "_v1"))
    print("\nPour information : xlsx_v2 (nouvelle BOM) vs xlsx_v1 (ancienne)")
    print(side[["product", "material", "series", "kg_v2", "kg_v1", "weight_v2", "weight_v1"]]
          .sort_values(["product", "material"]).to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    # 3. EUR/kg acier du yaml vs panel ---------------------------------------------------------------------------
    section("3. EUR/kg acier du yaml vs panel au mois de base")
    base = a["eur_per_kg_at_base"]
    pairs = [("L01", panel.at[BASE, "L01"], base["carbon_steel"]["value"]),
             ("L03", panel.at[BASE, "L03"], base["galvanised_steel"]["value"]),
             ("L02 (dernière valeur)", panel["L02"].loc[:BASE].dropna().iloc[-1], base["stainless_steel"]["value"])]
    for name, px, yv in pairs:
        check(abs(px / 1000 / yv - 1) < 0.01, f"{name}/1000 = {px / 1000:.3f} vs yaml {yv} (écart < 1 %)")

    # 4. indices BOM ---------------------------------------------------------------------------------------------
    section("4. Indices BOM (base 2026-09 = 100)")
    pb, _ = add_bom_indices(build_panel())
    for prod, col in TARGET_IDS.items():
        v0, v1 = pb.at[pd.Timestamp("2025-09-01"), col], pb.at[BASE, col]
        check(abs(v1 - 100) < 1e-6, f"{col}[2026-09] = 100 (lu : {v1:.6f})")
        print(f"  {prod}: 2025-09 = {v0:.2f}, 2026-09 = {v1:.2f}, variation 12 mois = {100 * (v1 / v0 - 1):+.2f} %")

    # 5. couverture ----------------------------------------------------------------------------------------------
    section("5. Couverture de la BOM pricée")
    print(coverage_table(a).to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    # 6. teardowns concurrents ----------------------------------------------------------------------------------
    section("6. Teardowns concurrents")
    td = d.load_competitor_teardowns()
    pick = lambda b, col, v: td.loc[(td.brand == b) & (td[col] == v), "mass_kg"].sum()  # noqa: E731
    abb_cu, eaton_st = pick("ABB", "material", "Copper"), pick("Eaton", "group", "carbon_steel")
    check(abs(abb_cu - E["abb_copper_kg"]) < 1e-9, "ABB copper kg == expected (local file)")
    check(abs(eaton_st - E["eaton_steel_kg"]) < 1e-9, "Eaton steel kg == expected (local file)")
    check(bool(td.loc[td.brand == "SGC", "broken"].all()), "SGC marqué broken")
    out = d.OUT / "business"
    out.mkdir(parents=True, exist_ok=True)
    td.to_csv(out / "competitor_material_mass.csv", index=False)
    ok = td[~td["broken"]]
    shares = ok.pivot_table(index="brand", columns="group", values="mass_kg", aggfunc="sum").fillna(0.0)
    shares = shares.div(ok.groupby("brand")["total_kg"].first(), axis=0)
    groups = ["copper", "carbon_steel", "aluminium", "brass", "epoxy", "plastics"]
    print("Part de la masse totale par marque (teardowns non cassés) :")
    print(shares.reindex(columns=groups, fill_value=0.0).to_string(float_format=lambda x: f"{100 * x:.1f}%"))
    print(f"  -> {out / 'competitor_material_mass.csv'}")

    # 7. backtest court ------------------------------------------------------------------------------------------
    section("7. Backtest court (naive, naive_vol ; origines 2025-01 → 2026-09)")
    targets = [s for s in ["L01", "L02", "L03", "EP01", "EL01"] if s in panel and panel[s].notna().any()] + \
        ["RM6_NE_IQI_433520", "RMAirSeT_IIQ_24kV_IEC"]
    fc = run_backtest(panel, targets, [Naive(), NaiveVolScaled()],
                      origins=pd.date_range("2025-01-01", "2026-09-01", freq="MS"), derive=bom_deriver(panel),
                      verbose=False)
    last = fc[fc["origin"] == BASE]
    for s in targets:
        g = last[last["series"] == s]
        hs = set(g["horizon"])
        ordered = bool(((g.p10 <= g.p50) & (g.p50 <= g.p90)).all()) and not g.empty
        check({3, 6, 9} <= hs and ordered, f"{s} : prévisions h=3/6/9 à l'origine 2026-09, p10 ≤ p50 ≤ p90")
    scored = metrics.score_rows(fc)
    tab = metrics.summarize(scored)[["n", "MAE_pct", "coverage"]].unstack("horizon")
    print(tab.to_string(float_format=lambda x: f"{x:.3f}"))
    print(compute_footprint(fc.attrs["runtime_s"]).to_string())
    outc = d.OUT / "checks"
    outc.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(outc / "verify_fixes_forecasts.parquet")
    print(f"  -> {outc / 'verify_fixes_forecasts.parquet'}")

    # 8. bilan ---------------------------------------------------------------------------------------------------
    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for m in FAILURES:
            print(f"  - {m}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
