#!/usr/bin/env python3
"""Verify the bounded public reasoning-retention contract and emit a receipt."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_kv_sentinel import ReasoningKVSentinel  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts" / "public-core"
SCENARIO_PATH = ARTIFACT_DIR / "retention-scenario.json"
RECEIPT_PATH = ARTIFACT_DIR / "verification-receipt.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_tests() -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit(completed.returncode)
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stdout_sha256": sha256_bytes(completed.stdout.encode()),
        "stderr_sha256": sha256_bytes(completed.stderr.encode()),
    }


def bounded_scenario() -> dict[str, Any]:
    sentinel = ReasoningKVSentinel(
        max_cache_tokens=8,
        entropy_threshold=0.5,
        keep_tail=2,
        use_onnx=False,
    )
    tokens: list[dict[str, Any]] = []
    for idx in range(16):
        tokens.append(
            {
                "id": idx,
                "probs": [0.5, 0.5] if idx % 3 == 0 else [0.99, 0.01],
                "is_anchor": idx in {0, 7},
            }
        )
    retained, metrics = sentinel.prune_reasoning_trajectory(tokens)
    stable_metrics = {
        key: metrics[key]
        for key in (
            "initial_tokens",
            "retained_tokens",
            "evicted_tokens",
            "pressure_evictions",
            "compression_ratio",
            "use_onnx",
            "onnx_backend",
            "status",
        )
    }
    return {
        "input_record_count": len(tokens),
        "retained_ids": [item["id"] for item in retained],
        "metrics": stable_metrics,
        "evidence_state": "LOCAL_REASONING_RECORD_RETENTION_POLICY",
        "kv_tensor_mutation": False,
        "model_server_integration": False,
        "reasoning_quality_benchmark": False,
        "external_tool_execution": False,
        "live_mesh_integration": False,
    }


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tests = run_tests()
    scenario = bounded_scenario()
    rendered_scenario = json.dumps(scenario, indent=2, sort_keys=True) + "\n"
    SCENARIO_PATH.write_text(rendered_scenario, encoding="utf-8")

    receipt = {
        "schema": "glaciereq.reasoning-retention-public-receipt.v1",
        "evidence_level": "TEST",
        "canonical_capability": "reasoning_record_retention_policy",
        "scenario_sha256": sha256_bytes(rendered_scenario.encode()),
        "tests": tests,
        "kv_tensor_mutation": False,
        "model_server_integration": False,
        "reasoning_quality_benchmark": False,
        "external_tool_execution": False,
        "live_mesh_integration": False,
        "cpp_prototype_authoritative": False,
        "canonical_paths": [
            "README.md",
            "src/reasoning_kv_sentinel.py",
            "src/onnx_kv_scorer.py",
            "tests/test_sentinel.py",
            "tests/test_onnx_kv.py",
        ],
    }
    rendered_receipt = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    RECEIPT_PATH.write_text(rendered_receipt, encoding="utf-8")
    print(rendered_scenario, end="")
    print(rendered_receipt, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
