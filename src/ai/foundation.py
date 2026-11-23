"""Pretrained time-series foundation models (Chronos-Bolt, Chronos-2), run locally and zero-shot.

Leakage rules
-------------
* The context of a forecast made at origin ``t`` is the target series up to and including ``t`` (``t`` is the
  latest observation, as in ``src/backtest.py``). Nothing after ``t`` is ever passed to the model.
* Chronos-2 past covariates are the economic drivers of ``src/features.py::DRIVERS``, shifted by
  ``features._shift_for`` (extra publication lag of the driver vs the target) and cut at ``t`` as well.
* No fine-tuning: the weights are frozen, so the only possible contamination is what the model saw during
  pre-training. It is measured on the clean window (see ``config.MODELS[...]['clean_after']``).
"""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
import pandas as pd

from src import features as fe

from . import config as C


# --------------------------------------------------------------------------------------------- inputs
def build_input(targets: pd.DataFrame, exog: pd.DataFrame, series: str, origin: pd.Timestamp,
                covariates: bool = False, X: pd.DataFrame | None = None) -> dict:
    """Model input for one (series, origin): target levels up to ``origin`` (+ shifted drivers, cut at origin).

    Pure function of data dated <= origin (tested in tests/test_ai_foundation.py by scrambling the future).
    """
    y = targets[series].loc[:origin]
    y = y.loc[y.first_valid_index():]
    out = {"target": y.to_numpy(dtype="float64")}
    if covariates:
        X = fe._driver_frame(exog, targets) if X is None else X
        cov = {}
        for d in fe.DRIVERS.get(series, []):
            if d not in X or d == series:
                continue
            xs = X[d].shift(fe._shift_for(d, series)).reindex(y.index)
            if xs.notna().sum() >= 12:
                cov[d] = xs.to_numpy(dtype="float64")
        if cov:
            out["past_covariates"] = cov
    return out


# --------------------------------------------------------------------------------------------- models
@lru_cache(maxsize=None)
def load_pipeline(key: str):
    """Load a pinned pretrained pipeline (Apple GPU if available, else CPU)."""
    import torch
    from chronos import BaseChronosPipeline

    spec = C.MODELS[key]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return BaseChronosPipeline.from_pretrained(spec["hf_id"], revision=spec["revision"], device_map=device,
                                               torch_dtype=torch.float32)


def predict_batch(key: str, inputs: list[dict], horizon: int = max(C.HORIZONS),
                  quantiles=C.QUANTILES, batch_size: int = 128) -> tuple[np.ndarray, float]:
    """Quantile forecasts in LEVELS, shape (n_inputs, horizon, n_quantiles), sorted along the quantile axis.

    Returns (forecasts, runtime_seconds).
    """
    import torch

    pipe = load_pipeline(key)
    uses_cov = C.MODELS[key]["covariates"]
    out = np.full((len(inputs), horizon, len(quantiles)), np.nan)
    t0 = time.perf_counter()
    # Chronos-2 needs the same covariate names within a batch: group inputs by covariate set, keep the order.
    groups: dict[tuple, list[int]] = {}
    for i, d in enumerate(inputs):
        groups.setdefault(tuple(sorted(d.get("past_covariates", {}))) if uses_cov else (), []).append(i)
    for idx in groups.values():
        for j in range(0, len(idx), batch_size):
            part = idx[j:j + batch_size]
            if uses_cov:
                q, _ = pipe.predict_quantiles([inputs[i] for i in part], prediction_length=horizon,
                                              quantile_levels=list(quantiles))
                arr = np.stack([t[0].float().cpu().numpy() for t in q])  # (batch, horizon, nq), univariate target
            else:
                ctx = [torch.tensor(inputs[i]["target"], dtype=torch.float32) for i in part]
                q, _ = pipe.predict_quantiles(ctx, prediction_length=horizon, quantile_levels=list(quantiles))
                arr = q.float().cpu().numpy()
            out[part] = arr
    runtime = time.perf_counter() - t0
    return np.sort(out, axis=-1), runtime


# --------------------------------------------------------------------------------------------- adapter
class FoundationModel:
    """Team test-bench contract (champion/v2 ``src/testbench``):

    ``forecast(y, steps, exog=None) -> DataFrame(index=steps, columns=[p10, p50, p90])`` in levels.
    Univariate here (the team panel uses different driver names); covariate runs go through ``predict_batch``.
    """

    def __init__(self, key: str = "chronos-bolt-base"):
        self.key = key
        self.name = key

    def forecast(self, y: pd.Series, steps: list[int], exog: pd.DataFrame | None = None) -> pd.DataFrame:
        y = y.dropna()
        arr, _ = predict_batch(self.key, [{"target": y.to_numpy(dtype="float64")}], horizon=max(steps))
        q = arr[0]
        return pd.DataFrame({"p10": q[[s - 1 for s in steps], 0], "p50": q[[s - 1 for s in steps], 1],
                             "p90": q[[s - 1 for s in steps], 2]}, index=list(steps))
