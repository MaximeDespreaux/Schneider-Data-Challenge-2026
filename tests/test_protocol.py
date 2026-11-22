"""Dev/graded protocol, grade guard and manifest (temporary git repos, synthetic data)."""

import json
import subprocess

import pandas as pd
import pytest

from src.testbench import protocol as pr


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def _scored(origin, h):
    o = pd.Timestamp(origin)
    return pd.DataFrame({"origin": [o], "horizon": [h], "target_month": [o + pd.DateOffset(months=h)], "ql": [0.1]})


def test_dev_view_purges():
    rows = pd.concat([_scored("2021-11-01", 3), _scored("2021-06-01", 3)])
    out = pr.dev_view(rows)
    assert list(out.origin) == [pd.Timestamp("2021-06-01")]
    with pytest.raises(AssertionError):
        pr.dev_view(_scored("2022-01-01", 3))


def _canary(tmp, head, dirty=False, ok=True):
    d = tmp / "canaries"
    d.mkdir(exist_ok=True)
    (d / f"{head}.json").write_text(json.dumps({"head": head, "dirty": dirty, "PASS": ok, "results": {}}))
    return d


def _head(repo):
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()


def test_grade_refused_without_tag(repo):
    with pytest.raises(RuntimeError, match="tag"):
        pr.assert_can_grade(repo, _canary(repo, _head(repo)), repo / "graded")


def test_grade_refused_dirty(repo):
    _git(repo, "tag", "freeze-v1")
    (repo / "src" / "a.py").write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="clean"):
        pr.assert_can_grade(repo, _canary(repo, _head(repo)), repo / "graded")


def test_grade_refused_without_passing_canaries(repo):
    _git(repo, "tag", "freeze-v1")
    with pytest.raises(RuntimeError, match="canar"):
        pr.assert_can_grade(repo, _canary(repo, _head(repo), ok=False), repo / "graded")


def test_grade_ok_then_refused_second_time(repo):
    _git(repo, "tag", "freeze-v1")
    can = _canary(repo, _head(repo))
    assert pr.assert_can_grade(repo, can, repo / "graded") == "freeze-v1"
    (repo / "graded" / "freeze-v1").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="once"):
        pr.assert_can_grade(repo, can, repo / "graded")


def test_manifest_fields_and_hashes(tmp_path):
    f = tmp_path / "in.csv"
    f.write_text("a\n1\n")
    path = pr.write_manifest(tmp_path / "out", mode="submission", origins=pd.date_range("2020-01-01", periods=3, freq="MS"),
                             runtime_s=1.5, inputs=[f], configs=[f], use_licensed=True)
    m = json.loads(path.read_text())
    assert {"git_head", "dirty", "tag", "config_sha256", "input_sha256", "mode", "origins", "runtime_s", "python",
            "packages", "use_licensed"} <= set(m)
    assert m["origins"] == ["2020-01", "2020-03", 3]
    assert len(next(iter(m["input_sha256"].values()))) == 64


def test_submit_rejects_licensed_ids():
    with pytest.raises(ValueError, match="L01"):
        pr.assert_publishable_ids(["S09", "L01"])
    pr.assert_publishable_ids(["S09", "D01", "RM6_NE_IQI_433520"])


def test_graded_ignores_mode_and_report_needs_both_modes():
    import pandas as pd
    from src.testbench.pipeline import graded_modes
    from src.testbench.report import verdicts
    assert graded_modes("submission") == ["submission", "realtime"]
    t = pd.DataFrame({"model": ["wf_blend"], "group": ["steel"], "horizon": [3], "rel_QL": [.8], "rel_QL_lo": [.7],
                      "rel_QL_hi": [.9], "p_holm": [.01], "coverage": [.8]})
    with pytest.raises(ValueError, match="both modes"):
        verdicts({"submission": t})


def test_submission_lags_mirror_live_edge_of_internal_inputs():
    import numpy as np
    import pandas as pd
    from src.testbench.pipeline import lags_for
    idx = pd.date_range("2020-01-01", "2026-09-01", freq="MS")
    p = pd.DataFrame({"S09": 1.0, "L02": 1.0}, index=idx)
    p.loc[idx[-1], "L02"] = np.nan                      # licensed stainless one month behind at the live origin
    lags = lags_for("submission", p, ["S09", "L02"], "2026-09-01")
    assert lags.get("L02") == 1 and lags.get("S09", 0) == 0
