"""ONNX keep-importance path for reasoning KV sentinel."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onnx_kv_scorer import OnnxKeepScorer, features_from_tokens  # noqa: E402
from reasoning_kv_sentinel import ReasoningKVSentinel  # noqa: E402


class TestOnnxKv(unittest.TestCase):
    def test_features_shape(self):
        tokens = [
            {"probs": [0.5, 0.5], "is_anchor": False},
            {"probs": [1.0], "is_anchor": True},
        ]
        f = features_from_tokens(tokens, keep_tail=1)
        self.assertEqual(f.shape, (2, 4))

    def test_scorer_scores_and_backend(self):
        scorer = OnnxKeepScorer()
        tokens = [{"probs": [0.99, 0.01], "is_anchor": True}] * 5
        tokens += [{"probs": [0.99, 0.01], "is_anchor": False}] * 10
        s = scorer.score(tokens, keep_tail=2)
        self.assertEqual(len(s), 15)
        # anchors should score higher than low-entropy non-anchors on average
        self.assertGreater(float(s[0]), float(s[6]))
        self.assertIn(
            scorer.backend,
            ("onnxruntime", "missing_onnxruntime", "missing_model", "unavailable"),
        )

    def test_prune_with_onnx_flag(self):
        s = ReasoningKVSentinel(
            max_cache_tokens=100,
            entropy_threshold=0.99,  # force most pure-entropy path to drop
            keep_tail=0,
            use_onnx=True,
            onnx_threshold=0.3,
        )
        tokens = []
        for i in range(40):
            tokens.append(
                {
                    "id": i,
                    "probs": [0.5, 0.5],  # high entropy
                    "is_anchor": i == 0,
                }
            )
        ret, m = s.prune_reasoning_trajectory(tokens)
        self.assertTrue(m["use_onnx"])
        self.assertIsNotNone(m.get("onnx_backend"))
        # anchor 0 retained
        self.assertTrue(any(t.get("id") == 0 for t in ret))
        self.assertLessEqual(len(ret), 100)


if __name__ == "__main__":
    unittest.main()
