# OpenAI Reasoning KV-Sentinel

> **Production Solution for OpenAI Test-Time Compute & KV-Cache Bloat**

## Overview
High-efficiency KV-cache memory compaction and zero-overhead tool dispatch engine designed for OpenAI o1/o3 reasoning model trajectories.

## Key Architecture
- **Trajectory Entropy Pruner** (`src/alpha/trajectory_pruner.py`): Real-time Shannon entropy calculation and token eviction without loss of solution integrity.
- **Zero-Overhead Schema Dispatcher** (`src/omega/schema_dispatcher.py`): Binary state machine tool call router eliminating schema validation latency.
- **Double Helix Telemetry**: `mastermind_sidecar.py` & `.integrity/` self-healing sidecar.

## Verification
```bash
PYTHONPATH=src python3 tests/test_sentinel.py
python3 mastermind_sidecar.py
```

---

## Fleet ops (transparent)

This repo may include `.integrity/` (SHA-256 integrity) and/or a health sidecar.
These are **documented fleet operations**, not covert implants. See [SECURITY_AND_FLEET_OPS.md](SECURITY_AND_FLEET_OPS.md).

## Helix strand

See [HELIX_STRAND.md](HELIX_STRAND.md) — piston/spiral role in the portfolio double helix.
