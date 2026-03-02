# Tests adapted from:
# https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/python/onnxruntime_test_python.py
import struct

import numpy as np
import pytest


def _make_minimal_onnx_model(
    input_name: str,
    output_name: str,
    input_shape: list[int],
    output_shape: list[int],
    dtype: int = 1,  # FLOAT
) -> bytes:
    """Build a minimal ONNX identity model as raw bytes (no onnx package needed)."""
    # We hand-craft a tiny ONNX protobuf for an identity op model.
    # ONNX protobuf field numbers:
    #  ModelProto: ir_version=1, opset_import=8, graph=7
    #  GraphProto: node=1, input=11, output=12
    #  NodeProto: input=1, output=2, op_type=4
    #  ValueInfoProto: name=1, type=2
    #  TypeProto: tensor_type=1
    #  TypeProto.Tensor: elem_type=1, shape=2
    #  TensorShapeProto: dim=1
    #  TensorShapeProto.Dimension: dim_value=1

    def varint(v: int) -> bytes:
        result = b""
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                result += bytes([b | 0x80])
            else:
                result += bytes([b])
                break
        return result

    def field(field_num: int, wire_type: int, data: bytes) -> bytes:
        tag = (field_num << 3) | wire_type
        return varint(tag) + varint(len(data)) + data

    def string_field(field_num: int, s: str) -> bytes:
        return field(field_num, 2, s.encode())

    def varint_field(field_num: int, v: int) -> bytes:
        tag = (field_num << 3) | 0
        return varint(tag) + varint(v)

    def shape_proto(dims: list[int]) -> bytes:
        result = b""
        for d in dims:
            dim = varint_field(1, d)  # dim_value
            result += field(1, 2, dim)  # dim
        return result

    def type_proto(elem_type: int, shape: list[int]) -> bytes:
        tensor = varint_field(1, elem_type) + field(2, 2, shape_proto(shape))
        return field(1, 2, tensor)  # tensor_type

    def value_info(name: str, elem_type: int, shape: list[int]) -> bytes:
        return string_field(1, name) + field(2, 2, type_proto(elem_type, shape))

    def node_proto(op: str, inputs: list[str], outputs: list[str]) -> bytes:
        result = b""
        for i in inputs:
            result += string_field(1, i)
        for o in outputs:
            result += string_field(2, o)
        result += string_field(4, op)
        return result

    def graph_proto() -> bytes:
        node = field(1, 2, node_proto("Identity", [input_name], [output_name]))
        inp = field(11, 2, value_info(input_name, dtype, input_shape))
        out = field(12, 2, value_info(output_name, dtype, output_shape))
        return node + inp + out

    opset = field(1, 2, b"") + varint_field(2, 20)  # domain="" version=20
    model = (
        varint_field(1, 9)  # ir_version=9
        + field(8, 2, opset)  # opset_import
        + field(7, 2, graph_proto())  # graph
    )
    return model


def test_import():
    import onnxruntime

    assert hasattr(onnxruntime, "InferenceSession")


def test_version():
    import onnxruntime

    assert onnxruntime.__version__ == "1.24.2"


def test_inference_session_from_bytes():
    import onnxruntime as ort

    model_bytes = _make_minimal_onnx_model("input", "output", [1, 3], [1, 3])
    session = ort.InferenceSession(model_bytes)
    assert session is not None


def test_get_inputs_outputs():
    import onnxruntime as ort

    model_bytes = _make_minimal_onnx_model("x", "y", [2, 4], [2, 4])
    session = ort.InferenceSession(model_bytes)

    inputs = session.get_inputs()
    assert len(inputs) == 1
    assert inputs[0].name == "x"

    outputs = session.get_outputs()
    assert len(outputs) == 1
    assert outputs[0].name == "y"


def test_run_identity_float32():
    import onnxruntime as ort

    model_bytes = _make_minimal_onnx_model("input", "output", [2, 3], [2, 3])
    session = ort.InferenceSession(model_bytes)

    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    results = session.run(["output"], {"input": data})

    assert len(results) == 1
    np.testing.assert_array_equal(results[0], data)


def test_run_with_none_output_names():
    import onnxruntime as ort

    model_bytes = _make_minimal_onnx_model("input", "output", [3], [3])
    session = ort.InferenceSession(model_bytes)

    data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    results = session.run(None, {"input": data})

    assert len(results) == 1
    np.testing.assert_array_equal(results[0], data)


def test_get_providers():
    import onnxruntime as ort

    model_bytes = _make_minimal_onnx_model("input", "output", [1], [1])
    session = ort.InferenceSession(model_bytes)

    providers = session.get_providers()
    assert "WebAssemblyExecutionProvider" in providers


@pytest.mark.parametrize(
    "dtype,ort_dtype_id",
    [
        (np.float32, 1),
        (np.int32, 6),
        (np.int64, 7),
        (np.uint8, 2),
    ],
)
def test_run_various_dtypes(dtype, ort_dtype_id):
    import onnxruntime as ort

    model_bytes = _make_minimal_onnx_model("x", "y", [4], [4], dtype=ort_dtype_id)
    session = ort.InferenceSession(model_bytes)

    data = np.array([1, 2, 3, 4], dtype=dtype)
    results = session.run(["y"], {"x": data})

    assert len(results) == 1
    np.testing.assert_array_equal(results[0], data)
