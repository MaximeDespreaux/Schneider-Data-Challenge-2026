"""Agent loop with a fake LLM, on the local data (skips without it)."""

import re

import pandas as pd
import pytest

from src.ai import config as C

pytestmark = pytest.mark.skipif(not (C.ROOT / "data" / "processed" / "features_h3.parquet").exists(),
                                reason="needs the local NDA data")


@pytest.fixture(scope="module")
def data():
    from src import features as fe
    T, X = fe.build_panel()
    ex = {h: pd.read_parquet(C.ROOT / "data" / "processed" / f"features_h{h}.parquet") for h in C.HORIZONS}
    return T, X, ex


def test_every_prompt_passes_the_nda_gate():
    from src.ai import agent_features as af
    from src.ai import guard
    history = [{"formula": "dlog(E09, 3)", "summary": "passed 2 of 9 target-horizon checks, best signal weak"}]
    for fam in af.FAMILY_DESC:
        guard.check(af.SYSTEM + "\n" + af.prompt(fam, history))


def test_loop_with_fake_llm(data):
    from src.ai import agent_features as af
    T, X, ex = data
    sent = []

    def fake_ask(prompt, system="", schema=None, tag="", effort=""):
        from src.ai import guard
        guard.check(system + "\n" + prompt)
        sent.append(prompt)
        return {"features": [{"formula": "dlog(ED02, 6)", "rationale": "energy cost pass-through"},
                             {"formula": "dlog(CRU_CR_DE, 3)", "rationale": "licensed series, must be refused"},
                             {"formula": "__import__('os')", "rationale": "code injection, must be refused"}]}

    rows = af.run_family("ppi", T, X, ex, rounds=2, ask=fake_ask)
    res = pd.DataFrame(rows)
    assert len(sent) == 2
    assert res[res.formula == "dlog(CRU_CR_DE, 3)"].reason.str.contains("non-public").all()
    assert res[res.formula.str.startswith("__import__")].reason.str.contains("parser").all()
    assert "dlog(ED02, 6): passed" in sent[1]          # round 2 sees round 1's feedback...
    fb = sent[1].split("Already tried")[1].split("Propose")[0]
    assert not re.search(r"\d\.\d|rho|20\d\d", fb)      # ...but no correlation values or dates
    acc = af.accepted(res)
    assert set(acc.formula) <= {"dlog(ED02, 6)"} and set(acc.target) <= {"S09", "S10", "S11"}
