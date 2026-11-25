"""NDA gate tests. They need the local (git-ignored) data to build the deny-list, so they skip without it.
Private values are read from the deny-list at runtime; none is written in this file."""

import pandas as pd
import pytest

from src.ai import config as C

pytestmark = pytest.mark.skipif(not (C.ROOT / "data" / "processed" / "series_catalog.parquet").exists(),
                                reason="needs the local NDA data")


def test_only_public_series_pass():
    from src.ai import guard
    ok = pd.DataFrame(columns=["S09", "E09", "D01", "ED02"])
    assert guard.public_frame(ok) is ok
    for bad in ["CRU_CR_DE", "MEPS_CR_DE", "EPOXY_RESIN"]:
        with pytest.raises(guard.NDAViolation):
            guard.public_frame(pd.DataFrame(columns=["S09", bad]))
    assert not any(s.startswith(("CRU", "MEPS", "LME", "EPOXY")) for s in guard.public_series())


@pytest.mark.parametrize("text", ["Forecast for Schneider", "the CRU cold-rolled price", "our BOM masses",
                                  "Supplier margins", "AirSeT demand"])
def test_blocked_words(text):
    from src.ai import guard
    with pytest.raises(guard.NDAViolation):
        guard.check(text)


def test_private_numbers_and_terms_are_refused():
    from src.ai import guard
    path = C.ROOT / "data" / "nda_denylist.txt"
    if not path.exists():
        pytest.skip("no local deny-list (data/nda_denylist.txt)")
    lines = path.read_text(encoding="utf-8").splitlines()
    num = next((line[4:].strip() for line in lines if line.startswith("num:")), None)
    term = next((line.strip() for line in lines if line.strip() and not line.startswith(("#", "num:"))), None)
    if num is None or term is None:
        pytest.skip("the local deny-list has no 'num:' line or no term")
    with pytest.raises(guard.NDAViolation):
        guard.check(f"the value was {num} last month")
    with pytest.raises(guard.NDAViolation):
        guard.check(f"note from {term}")


def test_public_text_passes():
    from src.ai import guard
    text = "EU steel producer prices (Eurostat) rose 3.2% year on year; TTF gas averaged 35 EUR/MWh in 2025."
    assert guard.check(text) == text


def test_ask_refuses_before_any_network_call(monkeypatch):
    from src.ai import guard, llm
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    with pytest.raises(guard.NDAViolation):
        llm.ask("Forecast the CRU coil price")


def test_ollama_provider_goes_through_the_gate_and_logs(monkeypatch, tmp_path):
    from src.ai import guard, llm
    monkeypatch.setattr(llm, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(llm, "LOG", tmp_path / "log.jsonl")
    calls = []
    monkeypatch.setattr(llm, "_ollama", lambda *a: calls.append(a) or ('{"x": 1}', {"input_tokens": 3, "output_tokens": 2}))
    with pytest.raises(guard.NDAViolation):
        llm.ask("Forecast the CRU coil price", provider="ollama")
    assert calls == []
    out = llm.ask("public question", schema={"type": "object"}, provider="ollama")
    assert out == {"x": 1} and len(calls) == 1
    assert llm.ask("public question", schema={"type": "object"}, provider="ollama") == {"x": 1} and len(calls) == 1  # cached
    assert (tmp_path / "log.jsonl").read_text().count("\n") == 1
