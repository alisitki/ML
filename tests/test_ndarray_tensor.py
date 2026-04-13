"""test_ndarray_tensor.py

Tests for the RawScaleTensor ndarray representation fix.
Covers: validation, dtype, list coercion, JSON round-trip, features output.
"""
from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from quantlab_ml.contracts.learning_surface import RawScaleTensor


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_tensor(n: int = 4) -> RawScaleTensor:
    """Build a minimal valid RawScaleTensor with ndarray inputs."""
    return RawScaleTensor(
        scale_label="1m",
        shape=[n, 1, 1, 1, 1],
        values=np.arange(n, dtype=np.float32),
        age=np.ones(n, dtype=np.float32) * 30.0,
        padding=np.array([True] + [False] * (n - 1), dtype=np.bool_),
        unavailable_by_contract=np.zeros(n, dtype=np.bool_),
        missing=np.zeros(n, dtype=np.bool_),
        stale=np.zeros(n, dtype=np.bool_),
    )


# ---------------------------------------------------------------------------
# Field type tests
# ---------------------------------------------------------------------------

def test_raw_scale_tensor_fields_are_ndarray() -> None:
    """After construction, all array fields must be numpy ndarrays."""
    t = _make_tensor()
    for field_name in ("values", "age", "padding", "unavailable_by_contract", "missing", "stale"):
        val = getattr(t, field_name)
        assert isinstance(val, np.ndarray), (
            f"field '{field_name}' expected ndarray, got {type(val).__name__}"
        )


def test_raw_scale_tensor_float_fields_are_float32() -> None:
    t = _make_tensor()
    assert t.values.dtype == np.float32
    assert t.age.dtype == np.float32


def test_raw_scale_tensor_bool_fields_are_bool() -> None:
    t = _make_tensor()
    for field_name in ("padding", "unavailable_by_contract", "missing", "stale"):
        arr = getattr(t, field_name)
        assert arr.dtype == np.bool_, (
            f"field '{field_name}' expected bool_, got {arr.dtype}"
        )


# ---------------------------------------------------------------------------
# List coercion (test/fixture compatibility)
# ---------------------------------------------------------------------------

def test_raw_scale_tensor_accepts_float_list_input() -> None:
    """Python list[float] must be coerced to float32 ndarray."""
    n = 3
    t = RawScaleTensor(
        scale_label="1m",
        shape=[n, 1, 1, 1, 1],
        values=[1.0, 2.0, 3.0],
        age=[10.0, 20.0, 30.0],
        padding=[False, False, False],
        unavailable_by_contract=[False, False, False],
        missing=[False, True, False],
        stale=[False, False, True],
    )
    assert isinstance(t.values, np.ndarray)
    assert t.values.dtype == np.float32
    assert list(t.values) == pytest.approx([1.0, 2.0, 3.0])
    assert t.missing[1] is True or bool(t.missing[1]) is True


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

def test_raw_scale_tensor_json_roundtrip() -> None:
    """Serialise → JSON string → deserialise → array values must be equal."""
    orig = _make_tensor(n=6)
    json_str = orig.model_dump_json()
    loaded = RawScaleTensor.model_validate_json(json_str)

    assert isinstance(loaded.values, np.ndarray)
    assert np.array_equal(orig.values, loaded.values)
    assert np.array_equal(orig.age, loaded.age)
    assert np.array_equal(orig.padding, loaded.padding)
    assert np.array_equal(orig.missing, loaded.missing)
    assert orig.shape == loaded.shape


def test_raw_scale_tensor_json_uses_base64_dict() -> None:
    """Serialised JSON must encode arrays as base64 dicts, not plain lists."""
    t = _make_tensor(n=4)
    raw = json.loads(t.model_dump_json())
    for field_name in ("values", "age", "padding", "unavailable_by_contract", "missing", "stale"):
        field_val = raw[field_name]
        assert isinstance(field_val, dict), (
            f"expected dict encoding for '{field_name}', got {type(field_val).__name__}"
        )
        assert "data" in field_val, f"missing 'data' key in '{field_name}' encoding"
        assert "dtype" in field_val
        assert "shape" in field_val
        # Verify data is valid base64
        base64.b64decode(field_val["data"])  # must not raise


# ---------------------------------------------------------------------------
# Flat size validator still works
# ---------------------------------------------------------------------------

def test_raw_scale_tensor_rejects_wrong_size() -> None:
    """Shape mismatch must still raise a validation error."""
    with pytest.raises(Exception):
        RawScaleTensor(
            scale_label="1m",
            shape=[4, 1, 1, 1, 1],  # expects 4 elements
            values=np.zeros(3, dtype=np.float32),  # wrong size
            age=np.zeros(4, dtype=np.float32),
            padding=np.zeros(4, dtype=np.bool_),
            unavailable_by_contract=np.zeros(4, dtype=np.bool_),
            missing=np.zeros(4, dtype=np.bool_),
            stale=np.zeros(4, dtype=np.bool_),
        )


# ---------------------------------------------------------------------------
# observation_feature_vector compatibility
# ---------------------------------------------------------------------------

def test_observation_feature_vector_with_ndarray_tensors(
    trajectory_bundle,  # type: ignore[no-untyped-def]
) -> None:
    """observation_feature_vector must return a flat list[float] when tensors are ndarrays."""
    from quantlab_ml.models.features import observation_feature_vector

    step = trajectory_bundle.splits["train"][0].steps[0]
    features = observation_feature_vector(step.observation)

    assert isinstance(features, list)
    assert all(isinstance(f, float) for f in features)
    assert len(features) > 0
