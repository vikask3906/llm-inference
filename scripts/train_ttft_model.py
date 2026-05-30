#!/usr/bin/env python
"""Train a TTFT regressor from gateway observation logs.

Input: JSONL file written by gateway.extensions.ttft_predictor.ObservationLogger
       (one Observation per line, schema: see that module).
Output: joblib-pickled scikit-learn / XGBoost regressor at --out.

The gateway hot-reloads the model file on mtime change, so this can run on
a cron and the gateway picks up the new weights without restart.

Usage:
    python scripts/train_ttft_model.py \\
        --in logs/ttft.jsonl \\
        --out models/ttft.joblib \\
        --model xgboost
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gateway.extensions.ttft_predictor import FEATURE_NAMES


def load_observations(path: Path) -> tuple[list[list[float]], list[float]]:
    X: list[list[float]] = []
    y: list[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                row = [
                    float(o["prompt_tokens"]),
                    float(o["match_blocks"]),
                    float(o["uncached_tokens"]),
                    float(o["inflight"]),
                    float(o["kv_usage"]),
                    float(_backend_idx(o["backend_id"])),
                ]
                X.append(row)
                y.append(float(o["observed_ttft_ms"]))
            except (KeyError, ValueError, TypeError):
                continue
    return X, y


def _backend_idx(backend_id: str) -> int:
    # mirror gateway.extensions.ttft_predictor._backend_idx
    h = 0
    for c in backend_id.encode("utf-8"):
        h = (h * 31 + c) & 0xFFFFFFFF
    return h % 64


def train(X: list[list[float]], y: list[float], model_name: str):
    import numpy as np  # type: ignore

    Xa = np.array(X, dtype=float)
    ya = np.array(y, dtype=float)

    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor  # type: ignore
        except ImportError:
            print("xgboost not installed; falling back to sklearn GradientBoostingRegressor",
                  file=sys.stderr)
            model_name = "gbm"
    if model_name == "xgboost":
        model = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            tree_method="hist", n_jobs=1,
        )
    elif model_name == "gbm":
        from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
        model = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                          learning_rate=0.05)
    elif model_name == "rf":
        from sklearn.ensemble import RandomForestRegressor  # type: ignore
        model = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=1)
    else:
        raise SystemExit(f"unknown model: {model_name}")

    model.fit(Xa, ya)
    # Quick fit-quality readout; not a real eval (no train/test split here).
    yhat = model.predict(Xa)
    mae = float(np.mean(np.abs(yhat - ya)))
    print(f"fit complete: n={len(ya)} features={FEATURE_NAMES} train_mae_ms={mae:.2f}")
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="JSONL observations file")
    ap.add_argument("--out", required=True,
                    help="output joblib model path")
    ap.add_argument("--model", default="xgboost",
                    choices=["xgboost", "gbm", "rf"])
    ap.add_argument("--min-rows", type=int, default=200,
                    help="refuse to train below this many observations")
    args = ap.parse_args()

    X, y = load_observations(Path(args.inp))
    if len(y) < args.min_rows:
        print(f"only {len(y)} observations < min-rows={args.min_rows}; skipping",
              file=sys.stderr)
        return 2

    model = train(X, y, args.model)

    import joblib  # type: ignore
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
