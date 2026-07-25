"""Test suite for OpenAI Reasoning KV-Sentinel solution."""
import unittest
from reasoning_kv_sentinel import ReasoningKVSentinel, ZeroOverheadSchemaDispatcher

class TestOpenAIReasoningKVSentinel(unittest.TestCase):

    def test_trajectory_pruning(self):
        sentinel = ReasoningKVSentinel(max_cache_tokens=1000, entropy_threshold=0.30)
        
        # Simulate reasoning trajectory with high and low entropy tokens
        tokens = [
            {"id": i, "probs": [0.99, 0.01], "is_anchor": i % 10 == 0} if i % 2 == 0
            else {"id": i, "probs": [0.5, 0.5], "is_anchor": False}
            for i in range(100)
        ]

        retained, metrics = sentinel.prune_reasoning_trajectory(tokens)
        self.assertTrue(metrics["retained_tokens"] < len(tokens))
        self.assertEqual(metrics["status"], "NOMINAL")

    def test_schema_dispatcher(self):
        schemas = [{
            "name": "execute_code",
            "parameters": {
                "type": "object",
                "required": ["language", "code"]
            }
        }]

        dispatcher = ZeroOverheadSchemaDispatcher(schemas)
        result = dispatcher.dispatch_tool_call(
            "execute_code", {"language": "python", "code": "print('hello')"}
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "EXECUTED")

if __name__ == "__main__":
    unittest.main()
