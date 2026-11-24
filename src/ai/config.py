"""Disclosure registry and settings for the AI track.

Every pretrained model or API used anywhere in the track must be listed in ``MODELS``: it feeds the licence table,
the ``ai_tools`` column of the submission CSV and the clean-window evaluation.

``clean_after``: forecasts whose *target month* is after this date cannot have been seen by the model during
training (the release date is used as a conservative proxy for the training cutoff, which the model cards do
not state precisely). Scores on that window are reported separately as the contamination-free check.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "outputs" / "ai"          # private outputs (git-ignored, data/*)
PUBLIC_OUT = ROOT / "public" / "ai"             # NDA-safe outputs (scanned by public/scan_leaks.py)

HORIZONS = (3, 6, 9)
QUANTILES = (0.1, 0.5, 0.9)
BACKTEST_START = "2020-01-01"   # same origins as notebooks/02 and src/backtest.py
SELECTION_SPLIT = "2024-01-01"
LIVE_ORIGIN = "2026-09-01"  # official forecast origin (Schneider CSV origin_month); data after it is ignored  # pre-registered: choose on targets before, report on targets from

MODELS: dict[str, dict] = {
    "chronos-bolt-small": {
        "kind": "pretrained time-series model (local)", "hf_id": "amazon/chronos-bolt-small",
        "revision": "772f3d25d38aec6d914c8949dab4462e2d46f5d8", "params_m": 48, "licence": "Apache-2.0",
        "released": "2024-11-25", "clean_after": "2024-11-30", "covariates": False,
        "source": "https://huggingface.co/amazon/chronos-bolt-small",
    },
    "chronos-bolt-base": {
        "kind": "pretrained time-series model (local)", "hf_id": "amazon/chronos-bolt-base",
        "revision": "5d9f166d69f47aef3401367a7b842e78fe97b121", "params_m": 205, "licence": "Apache-2.0",
        "released": "2024-11-25", "clean_after": "2024-11-30", "covariates": False,
        "source": "https://huggingface.co/amazon/chronos-bolt-base",
    },
    "chronos-2": {
        "kind": "pretrained time-series model (local)", "hf_id": "amazon/chronos-2",
        "revision": "29ec3766d36d6f73f0696f85560a422f50e8498c", "params_m": 119, "licence": "Apache-2.0",
        "released": "2025-10-30", "clean_after": "2025-10-31", "covariates": True,
        "source": "https://huggingface.co/amazon/chronos-2",
    },
}
# LLMs used by phases 2-4 (separate from MODELS, which the forecasting backtest loops over).
LLMS: dict[str, dict] = {
    "qwen3:14b": {
        "kind": "local LLM via Ollama (no data leaves the machine)", "hf_id": "Qwen/Qwen3-14B (Q4_K_M GGUF)",
        "revision": "ollama digest bdbd181c33f2", "licence": "Apache-2.0", "released": "2025-04-29",
        "source": "https://huggingface.co/Qwen/Qwen3-14B",
    },
    "claude-opus-5-5": {
        "kind": "Claude API (public data only, through src/ai/guard.py)", "hf_id": "-", "revision": "claude-opus-5-5",
        "licence": "Anthropic commercial terms", "released": "2026-01-01", "source": "https://docs.anthropic.com",
    },
}

EXCLUDED = {
    "moirai": "Salesforce Moirai: CC-BY-NC-4.0 (non-commercial) - incompatible with a forecast Schneider may use.",
}

# Drivers given to Chronos-2 as past covariates (same economic drivers as the dynamic regression,
# src/features.py::DRIVERS), shifted by their extra publication lag so no unpublished value is used.
COVARIATES_FROM_DRIVERS = True


def licence_table() -> pd.DataFrame:
    rows = [{"tool": k, **{c: v[c] for c in ("kind", "hf_id", "revision", "licence", "released", "source")}}
            for k, v in {**MODELS, **LLMS}.items()]
    rows += [{"tool": k, "kind": "excluded", "licence": v} for k, v in EXCLUDED.items()]
    return pd.DataFrame(rows)
