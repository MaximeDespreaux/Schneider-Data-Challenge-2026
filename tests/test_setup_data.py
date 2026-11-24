"""setup_data: plans the copies from every delivered layout, never overwrites, never takes the deprecated BOM."""

from src import setup_data as sd


def test_plan_covers_the_flat_layout_and_never_overwrites(tmp_path):
    data, raw = tmp_path / "data", tmp_path / "data" / "raw"
    panel = raw / "panel"
    for p in [data / "RM_Prices_XHEC_Challenge_v1.xlsx", data / "Epoxy for student.xlsx",
              data / "benchmark-modules-fonctionnels-rm6-x.xlsx", raw / "data" / "bom_materials.csv",
              raw / "data" / "RM_Prices_XHEC_Challenge.xlsx",                       # deprecated copy inside data.zip
              raw / "Supplier_balancesheet" / "PitchBook_-_A_Company_Financials_1.xlsx"]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    plan = sd.plan_copies(data / "Data", raw, panel, data)
    names = {b.relative_to(raw).as_posix() for _, b in plan}
    assert names == {"RM_Prices_XHEC_Challenge_v1.xlsx", "Epoxy for student.xlsx", "panel/bom_materials.csv",
                     "benchmark-modules-fonctionnels-rm6-x.xlsx", "suppliers/PitchBook_-_A_Company_Financials_1.xlsx"}
    (raw / "Epoxy for student.xlsx").write_text("already there")
    assert "Epoxy for student.xlsx" not in {b.name for _, b in sd.plan_copies(data / "Data", raw, panel, data)}
