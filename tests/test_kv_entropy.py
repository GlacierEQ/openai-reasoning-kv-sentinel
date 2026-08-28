"""Test suite for OpenAI Reasoning KV Entropy Pruner."""

import unittest


class KVEntropyPrunerSim:
    def prune_low_entropy(self, seq_len: int) -> int:
        return seq_len // 4 if seq_len > 1000 else 0


class TestKVEntropyPruner(unittest.TestCase):
    def test_pruning(self):
        p = KVEntropyPrunerSim()
        pruned = p.prune_low_entropy(2048)
        self.assertEqual(pruned, 512)


if __name__ == "__main__":
    unittest.main()
