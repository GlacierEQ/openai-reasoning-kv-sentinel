"""ONNX Runtime scorer for token keep-importance (reasoning KV).

Diversifies the stack: portable model artifact + ORT inference.
Falls back cleanly if onnxruntime or model file is missing.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

DEFAULT_MODEL = (
    Path(__file__).resolve().parents[1] / "models" / "token_keep_importance.onnx"
)


def shannon_entropy(probs: Sequence[float]) -> float:
    total = float(sum(probs)) if probs else 0.0
    if total <= 0:
        return 0.0
    ent = 0.0
    for p in probs:
        if p <= 0:
            continue
        q = p / total
        ent -= q * math.log2(q)
    return ent


def features_from_tokens(
    tokens: list[dict[str, Any]], keep_tail: int = 10
) -> np.ndarray:
    n = len(tokens)
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)
    tail_start = max(0, n - keep_tail)
    rows = []
    for i, t in enumerate(tokens):
        probs = t.get("probs") or [1.0]
        ent = shannon_entropy(probs)
        anchor = 1.0 if t.get("is_anchor") else 0.0
        pos = i / max(n - 1, 1)
        tail = 1.0 if i >= tail_start else 0.0
        rows.append([ent, anchor, pos, tail])
    return np.asarray(rows, dtype=np.float32)


class OnnxKeepScorer:
    """Score tokens for keep/drop under KV pressure via ONNX Runtime."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL
        self._session = None
        self.backend = "unavailable"
        self._init_session()

    def _init_session(self) -> None:
        if not self.model_path.is_file():
            self.backend = "missing_model"
            return
        try:
            import onnxruntime as ort
        except ImportError:
            self.backend = "missing_onnxruntime"
            return
        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self.backend = "onnxruntime"

    @property
    def available(self) -> bool:
        return self._session is not None

    def score(self, tokens: list[dict[str, Any]], keep_tail: int = 10) -> np.ndarray:
        """Return keep_score float32[N]. Higher = prefer retain."""
        feats = features_from_tokens(tokens, keep_tail=keep_tail)
        if feats.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        if not self.available:
            # pure-numpy linear fallback matching export weights
            w = np.array([0.55, 1.40, 0.15, 0.90], dtype=np.float32)
            b = -0.25
            logits = feats @ w + b
            return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
        out = self._session.run(None, {"features": feats})[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)
