"""
Smoke tests for onnxruntime in Pyodide (wasm32-emscripten).

These tests verify that the Pyodide wasm32 wheel built from
https://github.com/bartbroere/onnxruntime/tree/feature/pyodide-wasm32-wheel
loads correctly and can run basic ONNX model inference.
"""

import pytest
from pytest_pyodide import run_in_pyodide


@run_in_pyodide(packages=["onnxruntime"])
def test_import(selenium):
    import onnxruntime

    assert onnxruntime.__version__


@run_in_pyodide(packages=["onnxruntime"])
def test_version(selenium):
    import onnxruntime

    # Verify the version matches the recipe version.
    assert onnxruntime.__version__ == "1.25.0"


@run_in_pyodide(packages=["onnxruntime", "numpy"])
def test_inference_session_sigmoid(selenium):
    """
    Run inference on a minimal ONNX sigmoid model.

    The model is constructed from raw protobuf bytes so that the test has
    no dependency on the `onnx` package.  Sigmoid(0.0) = 0.5.
    """
    import numpy as np
    import onnxruntime as ort

    # Minimal ONNX sigmoid model serialised as bytes.
    # opset 11, single float32 input "X" → float32 output "Y" via Sigmoid.
    # Generated with:
    #   import onnx
    #   from onnx import helper, TensorProto
    #   X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
    #   Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    #   node = helper.make_node("Sigmoid", inputs=["X"], outputs=["Y"])
    #   graph = helper.make_graph([node], "sigmoid_graph", [X], [Y])
    #   model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    #   model.SerializeToString()
    model_bytes = (
        b"\x08\x07\x12\x00\"\x17\n\x07Sigmoid\x12\x01X\x1a\x01Y"
        b"\"\x0esigmoid_graph\x1a\x0b\n\x01X\x12\x06\n\x04\x08\x01"
        b"\x12\x00\"\x0b\n\x01Y\x12\x06\n\x04\x08\x01\x12\x00"
        b":\x0b\n\x07default\x10\x0b"
    )

    sess = ort.InferenceSession(model_bytes)
    input_name = sess.get_inputs()[0].name
    result = sess.run(None, {input_name: np.array([0.0], dtype=np.float32)})
    output = result[0][0]
    assert abs(output - 0.5) < 1e-5, f"sigmoid(0) expected ≈0.5, got {output}"


@run_in_pyodide(packages=["onnxruntime"])
def test_session_options(selenium):
    """Verify that SessionOptions can be instantiated and configured."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1


@run_in_pyodide(packages=["onnxruntime"])
def test_get_all_providers(selenium):
    """The wasm32 build only exposes the CPU execution provider."""
    import onnxruntime as ort

    providers = ort.get_all_providers()
    assert "CPUExecutionProvider" in providers
