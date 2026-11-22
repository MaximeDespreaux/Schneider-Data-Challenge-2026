import json

import pandas as pd

from src.testbench.candidates import SUBMITTED
from src.testbench.report import verdicts, write_report


def _tests(p_sub, p_rt):
    mk = lambda p: pd.DataFrame({"model": [SUBMITTED, SUBMITTED], "group": ["steel", "metals"], "horizon": [3, 3],  # noqa: E731
                                 "n_origins": [50, 50], "rel_QL": [0.8, 1.0], "rel_QL_lo": [0.7, 0.95],
                                 "rel_QL_hi": [0.9, 1.05], "p_better": [p, 0.5], "p_holm": [p, 1.0],
                                 "coverage": [0.8, 0.79], "kupiec_p": [0.5, 0.5]})
    return {"submission": mk(p_sub), "realtime": mk(p_rt)}


def test_pass_needs_both_modes():
    mde = pd.DataFrame({"group": ["steel", "metals"], "horizon": [3, 3], "mde_rel_ql": [0.05, 0.04]})
    v = verdicts(_tests(0.01, 0.02), mde).set_index(["group", "horizon"])
    assert v.loc[("steel", 3), "verdict"] == "beats naive"
    assert v.loc[("metals", 3), "verdict"].startswith("not detectable")
    v2 = verdicts(_tests(0.01, 0.20), mde).set_index(["group", "horizon"])
    assert v2.loc[("steel", 3), "verdict"].startswith("not detectable")


def test_report_written(tmp_path):
    for m, t in _tests(0.01, 0.02).items():
        (tmp_path / m).mkdir()
        t.to_csv(tmp_path / m / "pooled_tests.csv", index=False)
    can = tmp_path / "canary.json"
    can.write_text(json.dumps({"head": "abc", "dirty": False, "PASS": True, "results": {}}))
    mde = pd.DataFrame({"group": ["steel", "metals"], "horizon": [3, 3], "mde_rel_ql": [0.05, 0.04]})
    path = write_report(tmp_path, can, mde)
    txt = path.read_text(encoding="utf-8")
    assert "beats naive" in txt and "PASS" in txt
