from __future__ import annotations

import json
import os
import tempfile
import time

import pytest

from gateway.extensions.ttft_predictor import (
    Observation,
    ObservationLogger,
    TTFTPredictor,
    blended_ttft,
    features_for,
)


def test_features_shape():
    f = features_for(prompt_tokens=200, match_blocks=4, block_tokens=16,
                     inflight=3, kv_usage=0.45, backend_id="b0")
    assert len(f) == 6
    assert f[0] == 200.0
    assert f[1] == 4.0
    assert f[2] == 200 - 64       # uncached = prompt - match*block_tokens
    assert f[3] == 3.0
    assert f[4] == 0.45


def test_backend_idx_stable():
    f1 = features_for(100, 0, 16, 0, 0.0, "b0")
    f2 = features_for(100, 0, 16, 0, 0.0, "b0")
    assert f1[-1] == f2[-1]
    assert features_for(100, 0, 16, 0, 0.0, "b0")[-1] != \
           features_for(100, 0, 16, 0, 0.0, "b1")[-1]


def test_logger_writes_jsonl(tmp_path):
    p = tmp_path / "obs.jsonl"
    log = ObservationLogger(str(p))
    log.log(Observation(ts=time.time(), backend_id="b0",
                        prompt_tokens=100, match_blocks=2,
                        uncached_tokens=68, inflight=1, kv_usage=0.3,
                        observed_ttft_ms=42.0, hash_mode="bpe"))
    log.log(Observation(ts=time.time(), backend_id="b1",
                        prompt_tokens=200, match_blocks=0,
                        uncached_tokens=200, inflight=4, kv_usage=0.7,
                        observed_ttft_ms=130.0))
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["backend_id"] == "b0"
    assert rec["hash_mode"] == "bpe"


def test_logger_none_path_no_op():
    log = ObservationLogger(None)
    # must not raise
    log.log(Observation(ts=0, backend_id="x", prompt_tokens=1,
                        match_blocks=0, uncached_tokens=1, inflight=0,
                        kv_usage=0.0, observed_ttft_ms=1.0))


def test_predictor_no_model_returns_none():
    pred = TTFTPredictor(None)
    assert not pred.available()
    assert pred.predict([1.0] * 6) is None


def test_predictor_missing_file_returns_none(tmp_path):
    pred = TTFTPredictor(str(tmp_path / "nope.joblib"))
    assert not pred.available()
    assert pred.predict([1.0] * 6) is None


def test_blended_ttft_no_prediction_returns_static():
    assert blended_ttft(100.0, None, weight=0.5) == 100.0


def test_blended_ttft_pure_static():
    assert blended_ttft(100.0, 50.0, weight=0.0) == 100.0


def test_blended_ttft_pure_learned():
    assert blended_ttft(100.0, 50.0, weight=1.0) == 50.0


def test_blended_ttft_50_50():
    assert blended_ttft(100.0, 50.0, weight=0.5) == 75.0


def test_blended_ttft_clamps_weight():
    assert blended_ttft(100.0, 50.0, weight=2.0) == 50.0
    assert blended_ttft(100.0, 50.0, weight=-1.0) == 100.0


@pytest.mark.skipif(
    not all(__import__(m, fromlist=["_"]) for m in []) and False,
    reason="sklearn round-trip test only runs when sklearn+joblib are installed",
)
def test_predictor_loads_real_model(tmp_path):
    pytest.importorskip("sklearn")
    joblib = pytest.importorskip("joblib")
    from sklearn.linear_model import LinearRegression
    import numpy as np

    X = np.array([[100, 0, 100, 0, 0.0, 1],
                  [100, 4, 36, 0, 0.0, 1],
                  [200, 0, 200, 2, 0.3, 2]], dtype=float)
    y = np.array([50.0, 20.0, 120.0])
    m = LinearRegression().fit(X, y)
    p = tmp_path / "m.joblib"
    joblib.dump(m, str(p))

    pred = TTFTPredictor(str(p))
    assert pred.available()
    v = pred.predict([150.0, 0.0, 150.0, 1.0, 0.1, 1.0])
    assert v is not None and v > 0
