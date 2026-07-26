# Language / runtime diversification

## Why not all Python

| Layer | Choice | Advantage |
|-------|--------|-----------|
| **Domain logic / proofs** | Python | Fast iteration, SI math, tests |
| **Portable reasoning score** | **ONNX + ORT** | Same weights in Python, C++, Rust, JS, mobile |
| **Hot path (future)** | Rust / C++ ORT binding | Latency under pressure without rewrite of the model |

## ONNX for reasoning (this repo)

- **Model:** `models/token_keep_importance.onnx`  
- **Export:** `python3 scripts/export_onnx_importance.py`  
- **Infer:** `OnnxKeepScorer` via `onnxruntime` (CPU)  
- **Use:** `ReasoningKVSentinel(use_onnx=True)` ranks tokens for keep/drop under KV pressure  

Fallback: pure NumPy sigmoid with the same weights if ORT/model missing — proofs still pass.

## Other GlacierEQ leaves (where languages help)

| Domain | Advantageous stack |
|--------|--------------------|
| Telemetry / bus | Go or Rust for concurrent high-rate ingest |
| Mesh path / routing | Rust for hot graph kernels |
| GPU health | CUDA/C++ where talking to driver APIs |
| Control planes / CLI | TypeScript (UI), Rust (TUI like grok-build) |
| Thermal / orbital | Python remains correct for SI models + tests |

Rule: **diversify when the runtime contract is real** (portability, latency, FFI) — not for resume theater.
