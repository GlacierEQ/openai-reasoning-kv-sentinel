"""
OpenAI Reasoning KV-Sentinel — real trajectory pruning + tool dispatch.

Innovations (implemented, tested):
  1. Shannon-entropy trajectory pruner for test-time compute KV pressure
  2. Schema-validated tool dispatcher (required fields + basic types)
  3. Soft-cap eviction when retained cache exceeds max_cache_tokens
  4. Optional ONNX keep-importance scores (portable runtime; Python fallback)
"""
from __future__ import annotations

import math
import time
from typing import Any

# Shannon entropy of a discrete distribution (bits)
def shannon_entropy(probs: list[float]) -> float:
    if not probs:
        return 0.0
    total = sum(probs)
    if total <= 0:
        return 0.0
    ent = 0.0
    for p in probs:
        if p <= 0:
            continue
        q = p / total
        ent -= q * math.log2(q)
    return ent


class ReasoningKVSentinel:
    """KV-cache pressure manager for long reasoning trajectories."""

    def __init__(
        self,
        max_cache_tokens: int = 32768,
        entropy_threshold: float = 0.35,
        keep_tail: int = 10,
        use_onnx: bool = False,
        onnx_threshold: float = 0.45,
    ) -> None:
        if max_cache_tokens < 1:
            raise ValueError("max_cache_tokens must be >= 1")
        if not 0.0 <= entropy_threshold <= 32.0:
            raise ValueError("entropy_threshold out of range")
        if keep_tail < 0:
            raise ValueError("keep_tail must be >= 0")
        self.max_cache_tokens = max_cache_tokens
        self.entropy_threshold = entropy_threshold
        self.keep_tail = keep_tail
        self.use_onnx = use_onnx
        self.onnx_threshold = onnx_threshold
        self.active_cache_tokens = 0
        self.evicted_tokens_count = 0
        self._onnx_scorer = None
        if use_onnx:
            from onnx_kv_scorer import OnnxKeepScorer

            self._onnx_scorer = OnnxKeepScorer()

    def compute_token_entropy(self, token_probs: list[float]) -> float:
        return shannon_entropy(token_probs)

    def prune_reasoning_trajectory(
        self, tokens: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Drop low-value tokens; keep anchors, high-entropy / high-ONNX-score steps, and tail.
        If still over max_cache_tokens, drop oldest non-anchors first.
        """
        start = time.perf_counter()
        n = len(tokens)
        retained: list[dict[str, Any]] = []
        evicted = 0
        tail_start = max(0, n - self.keep_tail)
        onnx_backend = None
        scores = None
        if self.use_onnx and self._onnx_scorer is not None:
            scores = self._onnx_scorer.score(tokens, keep_tail=self.keep_tail)
            onnx_backend = self._onnx_scorer.backend

        for idx, item in enumerate(tokens):
            probs = item.get("probs") or [1.0]
            entropy = shannon_entropy(probs)
            is_anchor = bool(item.get("is_anchor", False))
            in_tail = idx >= tail_start
            onnx_keep = False
            if scores is not None and idx < len(scores):
                onnx_keep = float(scores[idx]) >= self.onnx_threshold
            if is_anchor or entropy >= self.entropy_threshold or in_tail or onnx_keep:
                retained.append(item)
            else:
                evicted += 1

        # Soft cap: drop oldest non-anchors until under budget
        pressure_evictions = 0
        if len(retained) > self.max_cache_tokens:
            survivors: list[dict[str, Any]] = []
            # keep from the end (recent) preferentially
            for item in reversed(retained):
                if len(survivors) >= self.max_cache_tokens:
                    if item.get("is_anchor"):
                        # still try to keep anchors by swapping oldest non-anchor
                        for j, s in enumerate(survivors):
                            if not s.get("is_anchor"):
                                survivors[j] = item
                                break
                    else:
                        pressure_evictions += 1
                    continue
                survivors.append(item)
            retained = list(reversed(survivors))
            evicted += pressure_evictions

        self.evicted_tokens_count += evicted
        self.active_cache_tokens = len(retained)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        initial = max(n, 1)
        compression_ratio = 1.0 - (len(retained) / initial)

        metrics = {
            "initial_tokens": n,
            "retained_tokens": len(retained),
            "evicted_tokens": evicted,
            "pressure_evictions": pressure_evictions,
            "compression_ratio": round(compression_ratio, 4),
            "latency_ms": round(elapsed_ms, 4),
            "use_onnx": bool(self.use_onnx),
            "onnx_backend": onnx_backend,
            "status": (
                "NOMINAL"
                if self.active_cache_tokens <= self.max_cache_tokens
                else "PRESSURE_HIGH"
            ),
        }
        return retained, metrics


class ZeroOverheadSchemaDispatcher:
    """Fast schema gate for tool/function calls (required + type checks)."""

    def __init__(self, schemas: list[dict[str, Any]]) -> None:
        self.registry = {s["name"]: s for s in schemas if "name" in s}

    def dispatch_tool_call(
        self, tool_name: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if tool_name not in self.registry:
            return {"ok": False, "error": f"Unknown tool: {tool_name}"}

        schema = self.registry[tool_name]
        params = schema.get("parameters") or {}
        required = params.get("required") or []
        properties = params.get("properties") or {}

        missing = [f for f in required if f not in payload]
        if missing:
            return {"ok": False, "error": f"Missing required fields: {missing}"}

        type_errors: list[str] = []
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        for key, prop in properties.items():
            if key not in payload:
                continue
            expected = prop.get("type")
            if not expected:
                continue
            py = type_map.get(expected)
            if py is None:
                continue
            val = payload[key]
            if expected == "number" and isinstance(val, bool):
                type_errors.append(f"{key}: expected number")
            elif expected == "integer" and isinstance(val, bool):
                type_errors.append(f"{key}: expected integer")
            elif not isinstance(val, py):
                type_errors.append(f"{key}: expected {expected}")

        if type_errors:
            return {"ok": False, "error": f"Type errors: {type_errors}"}

        return {
            "ok": True,
            "tool_name": tool_name,
            "dispatched_bytes": len(str(payload).encode("utf-8")),
            "status": "VALIDATED",
            "fields": sorted(payload.keys()),
        }
