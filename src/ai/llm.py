"""Single entry point for LLM calls in the AI track: a local model through Ollama, or the Claude API.

Every call: NDA gate (``guard.check``) → response cache → budget check → model → JSONL log.

* Provider: ``AI_PROVIDER=ollama`` (default, local ``qwen3:14b``: nothing leaves the machine, no cost) or
  ``anthropic`` (Claude API). The NDA gate runs for both, so the two providers see exactly the same prompts and
  results stay comparable.
* Settings come from ``.env`` (see ``.env.example``): AI_PROVIDER, OLLAMA_URL, OLLAMA_MODEL, ANTHROPIC_API_KEY,
  AI_MODEL, AI_BUDGET_USD.
* Responses are cached by a hash of (model, system, prompt, schema), so re-runs cost nothing and reproduce.
* The log (``data/outputs/ai/llm_log.jsonl``) keeps every prompt, response, token count and cost. It is the
  disclosure record and the source of the compute / CO₂ figures.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config as C
from . import guard

try:
    from dotenv import load_dotenv
except ImportError:  # only in the ``ai`` dependency group; without it, plain environment variables are used
    pass
else:
    load_dotenv(C.ROOT / ".env")

PROVIDER = os.getenv("AI_PROVIDER", "ollama")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "true").lower() == "true"
MODEL = os.getenv("AI_MODEL", "claude-opus-5-5")
BUDGET_USD = float(os.getenv("AI_BUDGET_USD", "5"))
LOG = C.OUT / "llm_log.jsonl"
CACHE = C.OUT / "llm_cache"
PRICE_PER_MTOK = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


def model_in_use() -> str:
    return OLLAMA_MODEL if PROVIDER == "ollama" else MODEL


def _same_model(answered: str, requested: str) -> None:
    """The clean window depends on which model answered: refuse (and do not cache) an answer from another one."""
    if answered != requested:
        raise RuntimeError(f"asked {requested}, answered by {answered}: not used (its release date would be wrong)")


def spent_usd() -> float:
    if not LOG.exists():
        return 0.0
    return sum(json.loads(line).get("cost_usd", 0.0) for line in LOG.read_text().splitlines() if line.strip())


def _cost(model: str, usage) -> float:
    pin, pout = PRICE_PER_MTOK.get(model, PRICE_PER_MTOK["claude-opus-5-5"])
    return (usage.input_tokens * pin + usage.output_tokens * pout) / 1e6


def _ollama(prompt: str, system: str, schema: dict | None, max_tokens: int, model: str) -> tuple[str, dict]:
    """Ollama native chat API (same call shape as the Kairos client). Returns (text, usage)."""
    import urllib.error
    import urllib.request

    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": msgs, "stream": False, "think": OLLAMA_THINK,
            # Ollama's default context is a few thousand tokens: the feature prompts need more.
            "options": {"num_predict": max_tokens, "num_ctx": 16384, "temperature": 0.7}}
    if schema:
        body["format"] = schema
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            data = json.loads(r.read())
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama unreachable at {OLLAMA_URL} ({e}); start it with `ollama serve`") from e
    if data.get("done_reason") == "length":
        raise RuntimeError(f"Ollama response truncated at num_predict={max_tokens}")
    return data["message"]["content"], {"input_tokens": data.get("prompt_eval_count", 0),
                                        "output_tokens": data.get("eval_count", 0)}


def ask(prompt: str, system: str = "", schema: dict | None = None, tag: str = "", effort: str = "medium",
        max_tokens: int = 16000, model: str | None = None, provider: str | None = None) -> dict | str:
    """Send one prompt. Returns parsed JSON if ``schema`` is given, else the text.

    Raises ``guard.NDAViolation`` before anything is sent if the payload is not public,
    and ``RuntimeError`` if the budget in ``.env`` would be exceeded (Claude API only).
    """
    provider = provider or PROVIDER
    model = model or (OLLAMA_MODEL if provider == "ollama" else MODEL)
    guard.check(system + "\n" + prompt)
    key = hashlib.sha256(json.dumps([provider, model, system, prompt, schema, effort, OLLAMA_THINK if provider == "ollama"
                                     else None], sort_keys=True).encode()).hexdigest()[:24]
    cached = CACHE / f"{key}.json"
    if cached.exists():
        return json.loads(cached.read_text())["output"]
    if provider == "ollama":
        t0 = time.perf_counter()
        text, usage = _ollama(prompt, system, schema, max_tokens, model)
        runtime = time.perf_counter() - t0
        output = json.loads(text) if schema else text
        _record(cached, dict(provider=provider, tag=tag, model=model, request_id=None, system=system, prompt=prompt,
                             schema=schema, output=output, **usage, cost_usd=0.0, runtime_s=runtime))
        return output
    if spent_usd() >= BUDGET_USD:
        raise RuntimeError(f"AI budget reached ({spent_usd():.2f} / {BUDGET_USD:.2f} USD); raise AI_BUDGET_USD in .env")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is empty: fill it in .env (template: .env.example)")

    import anthropic

    client = anthropic.Anthropic()
    kw = dict(model=model, max_tokens=max_tokens, output_config={"effort": effort},
              messages=[{"role": "user", "content": prompt}])
    if system:
        kw["system"] = system
    if schema:
        kw["output_config"]["format"] = {"type": "json_schema", "schema": schema}
    t0 = time.perf_counter()
    # No server-side fallback on purpose: another model's answer would not match the registered release date.
    resp = client.messages.create(**kw)
    runtime = time.perf_counter() - t0
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"model refused ({getattr(resp.stop_details, 'category', None)}): {tag}")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError(f"response truncated at max_tokens={max_tokens}: {tag}")
    _same_model(resp.model, model)
    text = next(b.text for b in resp.content if b.type == "text")
    output = json.loads(text) if schema else text
    _record(cached, dict(provider=provider, tag=tag, model=resp.model, request_id=resp._request_id, system=system,
                         prompt=prompt, schema=schema, output=output, input_tokens=resp.usage.input_tokens,
                         output_tokens=resp.usage.output_tokens, cost_usd=_cost(resp.model, resp.usage),
                         runtime_s=runtime))
    return output


def _record(cached: Path, record: dict) -> None:
    record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(record, indent=1))
    with LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")


def ping() -> str:
    """Cheap connectivity check (a few tokens)."""
    return ask("Reply with the single word: ok", tag="ping", effort="low", max_tokens=200)


if __name__ == "__main__":
    who = f"{OLLAMA_MODEL} via Ollama (local)" if PROVIDER == "ollama" else f"{MODEL} via the Claude API"
    print(ping(), f"| {who} | spent so far: {spent_usd():.4f} USD of {BUDGET_USD} | log: {Path(LOG)}")
