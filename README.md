# Reasoning KV Retention Laboratory

An independent portfolio experiment for **entropy- and feature-guided retention of reasoning-trajectory records under a bounded cache-count budget**.

> **Independent portfolio project.** This repository is not affiliated with, endorsed by, employed by, or deployed at OpenAI. It does not claim access to OpenAI models, private inference infrastructure, proprietary KV caches, telemetry, or production systems.

## Recruiter view

The strongest current implementation is the Python policy in [`src/reasoning_kv_sentinel.py`](src/reasoning_kv_sentinel.py). Given caller-supplied token records containing probability vectors and optional anchor flags, it can:

- compute Shannon entropy from the supplied probability vector;
- retain anchors, high-entropy records, and a configurable recent tail;
- apply an optional local ONNX/NumPy keep-importance score;
- cap the number of retained records under a configured budget;
- return explicit counts and policy metrics;
- validate tool-call payload shapes and primitive field types without executing the tool.

This is a useful local control-policy prototype for a future KV-pressure system. It is **not yet a model-server KV-cache implementation**.

## Exact current boundary

```text
LOCAL_REASONING_RECORD_RETENTION_POLICY
NO_KV_TENSOR_MUTATION
NO_MODEL_SERVER_INTEGRATION
NO_REASONING_QUALITY_BENCHMARK
NO_OPENAI_AFFILIATION_OR_PRIVATE_ACCESS
```

In this repository, “retention” or “pruning” means selecting a subset of Python token/trajectory records. The verified code does **not** delete or rewrite key/value tensors inside a transformer runtime.

## Engineering anatomy

### Canonical policy

`src/reasoning_kv_sentinel.py`

The policy computes Shannon entropy:

```text
H(p) = -sum(q * log2(q))
q = p / sum(p)
```

For each supplied record, it retains the item when one or more policy conditions hold:

- `is_anchor` is true;
- entropy meets the configured threshold;
- the item is within the configured recent tail;
- an optional ONNX/NumPy keep score meets its threshold.

If the retained list still exceeds `max_cache_tokens`, the implementation applies a count-based soft cap favoring recent records while attempting to preserve anchors.

### Optional keep-importance scorer

`src/onnx_kv_scorer.py`

The scorer builds four local features from the supplied records:

1. entropy;
2. anchor flag;
3. normalized position;
4. tail membership.

It uses the checked-in ONNX artifact when ONNX Runtime is available. Otherwise it falls back to a deterministic NumPy linear/logistic scorer. This is a **local ranking heuristic**, not a learned production model quality claim.

### Schema validator

`ZeroOverheadSchemaDispatcher` is a historical name. Its verified behavior is a **local schema gate**: it checks required fields and basic primitive types and returns a validation result. It does not invoke a tool, call a remote service, or execute payload code.

### C++ prototype

`src/kv_entropy_pruner.cpp` is preserved as an earlier prototype. It contains a Shannon-entropy helper, but its current `prune_low_entropy_keys` demonstration uses only sequence length and returns a 25% count for sequences over 1000 entries. It does not mutate tensors and is **not the canonical proof of entropy-guided retention**.

## Native proof

Run the bounded public verification path:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/verify_public_core.py
```

The CI truth gate also compiles and runs the C++ prototype separately so its buildability is not confused with the stronger Python policy semantics.

## What the tests currently establish

The checked-in tests cover:

- Shannon entropy for deterministic and uniform distributions;
- dropping lower-entropy synthetic records under the policy;
- anchor preservation in tested cases;
- count-based pressure capping;
- required-field and primitive-type schema validation;
- ONNX/NumPy feature-shape and scoring behavior;
- optional scorer integration into the local retention policy.

## What the tests do **not** establish

They do not establish:

- preserved reasoning accuracy, answer quality, perplexity, or benchmark scores;
- VRAM reduction in a real transformer runtime;
- latency or throughput improvement in production inference;
- a 75% pruning guarantee or any universal compression ratio;
- correctness on actual model KV tensors;
- OpenAI model compatibility, deployment, endorsement, or internal usage;
- MCP registration or live APEX Highway / Mastermind mesh connectivity;
- external tool execution through the local schema validator.

Any future reasoning-quality, memory, latency, throughput, or deployment claim requires a separate benchmark or integration receipt bound to the exact implementation and source revision.

## Sidecar boundary

`mastermind_sidecar.py` generates a local JSON health/status object. That is useful as an adapter shape, but the file contains no transport, registration, remote call, or proof of live mesh connectivity. It is therefore treated as **LOCAL_ADAPTER_SHAPE**, not a deployed integration.

## Target architecture

The ambition remains larger than the current proof. A future production-grade version could:

1. instrument an actual transformer inference runtime;
2. bind retention decisions to real KV tensor positions;
3. measure allocated memory before and after mutation;
4. benchmark reasoning quality against an unpruned baseline;
5. measure latency/throughput across controlled workloads;
6. verify model/runtime compatibility;
7. issue a bounded integration/deployment receipt.

Until those gates exist, the public surface stays at the local policy/test level.

## Machine contract

```json
{
  "project": "GlacierEQ/openai-reasoning-kv-sentinel",
  "relationship_to_openai": "independent_portfolio_project_no_affiliation_claim",
  "canonical_capability": "reasoning_record_retention_policy",
  "evidence_level": "TEST",
  "kv_tensor_mutation": false,
  "model_server_integration": false,
  "reasoning_quality_benchmark": false,
  "external_tool_execution": false,
  "live_mesh_integration": false,
  "cpp_prototype_authoritative": false
}
```


## For recruiters and non-technical reviewers

## For senior engineers and domain experts

## For AI systems and toolchains
