"""Heavy tests for OpenAI Reasoning KV-Sentinel — real entropy + schema gates."""

from __future__ import annotations

import unittest

from reasoning_kv_sentinel import (
    ReasoningKVSentinel,
    ZeroOverheadSchemaDispatcher,
    shannon_entropy,
)


class TestShannonEntropy(unittest.TestCase):
    def test_deterministic_is_zero(self):
        self.assertAlmostEqual(shannon_entropy([1.0]), 0.0, places=9)

    def test_fair_coin_is_one_bit(self):
        self.assertAlmostEqual(shannon_entropy([0.5, 0.5]), 1.0, places=9)

    def test_empty(self):
        self.assertEqual(shannon_entropy([]), 0.0)


class TestOpenAIReasoningKVSentinel(unittest.TestCase):
    def test_trajectory_pruning_drops_low_entropy(self):
        sentinel = ReasoningKVSentinel(
            max_cache_tokens=1000, entropy_threshold=0.30, keep_tail=5
        )
        tokens = []
        for i in range(100):
            if i % 2 == 0:
                tokens.append(
                    {"id": i, "probs": [0.99, 0.01], "is_anchor": i % 10 == 0}
                )
            else:
                tokens.append({"id": i, "probs": [0.5, 0.5], "is_anchor": False})

        retained, metrics = sentinel.prune_reasoning_trajectory(tokens)
        self.assertLess(metrics["retained_tokens"], len(tokens))
        self.assertEqual(metrics["status"], "NOMINAL")
        self.assertGreater(metrics["evicted_tokens"], 0)
        # all anchors retained
        anchor_ids = {t["id"] for t in tokens if t.get("is_anchor")}
        retained_ids = {t["id"] for t in retained}
        self.assertTrue(anchor_ids.issubset(retained_ids))

    def test_pressure_soft_cap(self):
        sentinel = ReasoningKVSentinel(
            max_cache_tokens=20, entropy_threshold=0.0, keep_tail=0
        )
        # all high entropy + anchors → would retain all without cap
        tokens = [{"id": i, "probs": [0.5, 0.5], "is_anchor": i < 3} for i in range(50)]
        retained, metrics = sentinel.prune_reasoning_trajectory(tokens)
        self.assertLessEqual(len(retained), 20)
        self.assertEqual(metrics["status"], "NOMINAL")
        self.assertGreaterEqual(metrics["pressure_evictions"], 1)

    def test_schema_dispatcher_ok(self):
        schemas = [
            {
                "name": "execute_code",
                "parameters": {
                    "type": "object",
                    "required": ["language", "code"],
                    "properties": {
                        "language": {"type": "string"},
                        "code": {"type": "string"},
                    },
                },
            }
        ]
        dispatcher = ZeroOverheadSchemaDispatcher(schemas)
        result = dispatcher.dispatch_tool_call(
            "execute_code", {"language": "python", "code": "print(1)"}
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "VALIDATED")
        self.assertNotIn("answer", result)  # no magic placeholders

    def test_schema_dispatcher_missing_and_types(self):
        schemas = [
            {
                "name": "set_temp",
                "parameters": {
                    "required": ["celsius"],
                    "properties": {"celsius": {"type": "number"}},
                },
            }
        ]
        d = ZeroOverheadSchemaDispatcher(schemas)
        self.assertFalse(d.dispatch_tool_call("set_temp", {})["ok"])
        self.assertFalse(d.dispatch_tool_call("set_temp", {"celsius": "hot"})["ok"])
        self.assertTrue(d.dispatch_tool_call("set_temp", {"celsius": 22.5})["ok"])

    def test_invalid_ctor(self):
        with self.assertRaises(ValueError):
            ReasoningKVSentinel(max_cache_tokens=0)


if __name__ == "__main__":
    unittest.main()
