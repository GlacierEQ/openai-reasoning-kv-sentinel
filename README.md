# OpenAI Reasoning KV Sentinel — C++ KV-Cache Entropy Pruner 🧠

> **C++ KV-cache entropy computation and low-information tensor pruner for reasoning models.**

[![C++](https://img.shields.io/badge/C++-17-00599C)]()
[![Python](https://img.shields.io/badge/Python-3.9+-blue)]()
[![Domain](https://img.shields.io/badge/Domain-KV%20Cache%20Optimization-blue)]()

---

## 🎯 For Recruiters & Hiring Managers

This repository implements a **C++ KV-Cache Entropy Pruner** — calculating token entropy across long-context reasoning sequences to prune low-information KV tensors. It demonstrates:

- **C++ entropy computation** measuring information density per key-value vector
- **Sequence truncation algorithms** pruning up to 75% of KV entries for sequences >1000 tokens
- **Memory footprint reduction** freeing VRAM for higher batch sizes during inference
- **Python test harness** verifying entropy calculation accuracy against baseline

**Why this matters**: Extended reasoning models generate tens of thousands of intermediate tokens. Pruning redundant KV states preserves memory capacity without compromising reasoning accuracy.

---

## 🔬 For Engineers & Technical Reviewers

### Core Components

| Component | Language | Purpose |
|---|---|---|
| `src/kv_entropy_pruner.cpp` | C++ | C++ class for KV tensor entropy calculation & pruning |
| `tests/test_kv_entropy.py` | Python | Test wrapper simulating sequence pruning |

---

## 🤖 ML/AI & Programmatic Mesh Integration

- **MCP Tool**: `prune_kv_entropy()` — accessible to inference optimization agents
- **Mastermind Sidecar**: Fully connected to APEX Highway mesh
- **SHA-256 Integrity**: Tracked in `.integrity/file_hashes.json`

---

## ⚡ Quick Start

```bash
python3 tests/test_kv_entropy.py
```
