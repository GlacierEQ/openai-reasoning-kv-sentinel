# OpenAI Reasoning KV-Sentinel

> **Production Solution for OpenAI Test-Time Compute & KV-Cache Bloat**

## Overview
High-efficiency KV-cache memory compaction and zero-overhead tool dispatch engine designed for OpenAI o1/o3 reasoning model trajectories.

## Key Architecture
- **Trajectory Entropy Pruner** (`src/alpha/trajectory_pruner.py`): Real-time Shannon entropy calculation and token eviction without loss of solution integrity.
- **Zero-Overhead Schema Dispatcher** (`src/omega/schema_dispatcher.py`): Binary state machine tool call router eliminating schema validation latency.
- **Double Helix Telemetry**: `mastermind_sidecar.py` & `.shadow_infrastructure/` self-healing sidecar.

## Verification
```bash
PYTHONPATH=src python3 tests/test_sentinel.py
python3 mastermind_sidecar.py
```
