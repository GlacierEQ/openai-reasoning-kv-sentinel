#!/usr/bin/env python3
"""Export a tiny ONNX model: token keep-importance for reasoning KV prune.

Features (float32[N, 4]):
  0: shannon entropy (bits)
  1: is_anchor (0/1)
  2: relative position in trajectory [0,1]
  3: tail_boost (1 if in last keep_tail fraction else 0)

Output (float32[N, 1]): keep_score (higher = retain under pressure)

This is a portable inference artifact — not "we use ONNX for marketing."
Python builds the graph once; onnxruntime (or any ORT binding in C++/Rust/etc.)
runs the same weights across stacks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError as e:
    raise SystemExit("pip install onnx") from e

OUT = Path(__file__).resolve().parents[1] / "models" / "token_keep_importance.onnx"


def build_model() -> onnx.ModelProto:
    # Linear: score = w·x + b with fixed weights encoding domain intent
    # high weight on entropy + anchor + tail
    w = np.array([[0.55], [1.40], [0.15], [0.90]], dtype=np.float32)  # 4x1
    b = np.array([-0.25], dtype=np.float32)

    W = numpy_helper.from_array(w, name="W")
    B = numpy_helper.from_array(b, name="B")

    X = helper.make_tensor_value_info("features", TensorProto.FLOAT, ["N", 4])
    Y = helper.make_tensor_value_info("keep_score", TensorProto.FLOAT, ["N", 1])

    # MatMul(features, W) + B  then Sigmoid for (0,1) scores
    nodes = [
        helper.make_node("MatMul", ["features", "W"], ["logits"]),
        helper.make_node("Add", ["logits", "B"], ["pre_act"]),
        helper.make_node("Sigmoid", ["pre_act"], ["keep_score"]),
    ]
    graph = helper.make_graph(nodes, "token_keep_importance", [X], [Y], [W, B])
    model = helper.make_model(
        graph,
        producer_name="glaciereq-reasoning-kv-sentinel",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = build_model()
    onnx.save(model, str(OUT))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
