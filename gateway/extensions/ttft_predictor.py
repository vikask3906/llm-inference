from __future__ import annotations

"""Predictive TTFT load scoring.

The static model in router._est_ttft uses two hand-tuned constants:
    prefill_ms_per_token, service_ms_per_request
which can't capture per-backend differences (different GPU SKUs, different
batch sizes, different KV-cache headroom). This module replaces them with
a learned model trained on real (features, observed_ttft) pairs.

Lifecycle:
    1. Gateway logs every request's features + observed TTFT to JSONL.
    2. `scripts/train_ttft_model.py` fits an XGBoost regressor offline.
    3. Gateway loads the trained model on startup; router blends learned
       prediction with the static formula via `Config.ttft_predictor_weight`.

Graceful fallback: if the model file is missing or sklearn/xgboost isn't
installed, predict() returns None and the router uses the static formula
alone. This keeps the gateway working before any training data exists.
"""

import dataclasses
import json
import os
import threading
import time
from pathlib import Path

FEATURE_NAMES = [
    "prompt_tokens",
    "match_blocks",
    "uncached_tokens",
    "inflight",
    "kv_usage",
    "backend_idx",   # stable hash of backend_id mod 64 -- learns per-backend bias
]


@dataclasses.dataclass
class Observation:
    """One (features, label) pair written per completed request."""
    ts: float
    backend_id: str
    prompt_tokens: int
    match_blocks: int
    uncached_tokens: int
    inflight: int
    kv_usage: float
    observed_ttft_ms: float
    hash_mode: str = "char"  # "char" or "bpe", for downstream analysis


class ObservationLogger:
    """Append-only JSONL writer. Thread-safe, fail-soft: a logging error
    never fails the request -- it just drops the observation."""

    def __init__(self, path: str | None):
        self.path = path
        self._lock = threading.Lock()
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    def log(self, obs: Observation) -> None:
        if not self.path:
            return
        line = json.dumps(dataclasses.asdict(obs), separators=(",", ":"))
        try:
            with self._lock, open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


def _backend_idx(backend_id: str) -> int:
    h = 0
    for c in backend_id.encode("utf-8"):
        h = (h * 31 + c) & 0xFFFFFFFF
    return h % 64


def features_for(prompt_tokens: int, match_blocks: int, block_tokens: int,
                 inflight: int, kv_usage: float, backend_id: str) -> list[float]:
    uncached = max(0, prompt_tokens - match_blocks * block_tokens)
    return [
        float(prompt_tokens),
        float(match_blocks),
        float(uncached),
        float(inflight),
        float(kv_usage),
        float(_backend_idx(backend_id)),
    ]


class TTFTPredictor:
    """Loads a trained sklearn-compatible regressor (XGBoost, GBM, RF) from
    a joblib file. predict() returns predicted TTFT in ms or None on any
    failure."""

    def __init__(self, model_path: str | None):
        self.model_path = model_path
        self._model = None
        self._mtime: float | None = None
        self._lock = threading.Lock()
        self._reload_if_changed()

    def available(self) -> bool:
        return self._model is not None

    def _reload_if_changed(self) -> None:
        if not self.model_path or not os.path.exists(self.model_path):
            return
        try:
            mtime = os.path.getmtime(self.model_path)
        except OSError:
            return
        if self._mtime == mtime:
            return
        try:
            import joblib  # type: ignore
        except ImportError:
            return
        try:
            with self._lock:
                self._model = joblib.load(self.model_path)
                self._mtime = mtime
        except Exception:
            self._model = None

    def predict(self, features: list[float]) -> float | None:
        if self._model is None:
            self._reload_if_changed()
        if self._model is None:
            return None
        try:
            import numpy as np  # type: ignore
            x = np.array(features, dtype=float).reshape(1, -1)
            y = self._model.predict(x)
            v = float(y[0])
            if v < 0 or not (v == v):  # negative or NaN
                return None
            return v
        except Exception:
            return None


def blended_ttft(static_ms: float, predicted_ms: float | None,
                 weight: float) -> float:
    """Blend the static linear-prefill estimate with the learned model.
    weight=0 -> pure static, weight=1 -> pure learned. None predicted -> static.
    """
    if predicted_ms is None:
        return static_ms
    w = max(0.0, min(1.0, weight))
    return (1.0 - w) * static_ms + w * predicted_ms
