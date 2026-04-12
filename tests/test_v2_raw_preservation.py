"""test_v2_raw_preservation.py

trade/BBO/funding/OI field ailelerinin observation içinde birlikte
kaldığını ve hiçbir stream'in tek scalar'a düşmediğini doğrular.
"""
from __future__ import annotations

from quantlab_ml.contracts import TrajectoryBundle
from quantlab_ml.contracts.dataset import REQUIRED_FIELDS_BY_STREAM


def test_all_stream_families_present_in_field_axis(trajectory_bundle: TrajectoryBundle) -> None:
    schema = trajectory_bundle.observation_schema
    for stream in schema.stream_axis:
        assert stream in schema.field_axis, f"stream {stream} missing from field_axis"
        assert len(schema.field_axis[stream]) > 1, (
            f"stream {stream} has only {len(schema.field_axis[stream])} field(s); "
            f"expected multi-field family"
        )


def test_required_fields_present_for_known_streams(trajectory_bundle: TrajectoryBundle) -> None:
    schema = trajectory_bundle.observation_schema
    for stream, required_fields in REQUIRED_FIELDS_BY_STREAM.items():
        if stream not in schema.field_axis:
            continue
        actual_fields = set(schema.field_axis[stream])
        for req_field in required_fields:
            assert req_field in actual_fields, (
                f"stream={stream} missing required field '{req_field}'; "
                f"actual={sorted(actual_fields)}"
            )


def test_no_stream_collapses_to_single_value(trajectory_bundle: TrajectoryBundle) -> None:
    """Hiçbir stream tek field ile temsil edilemez — min 2 field şart."""
    schema = trajectory_bundle.observation_schema
    for stream in schema.stream_axis:
        field_count = len(schema.field_axis.get(stream, []))
        assert field_count >= 2, (
            f"stream '{stream}' collapsed to {field_count} field(s); "
            f"expected >= 2 (no single-scalar collapse)"
        )


def test_raw_surface_contains_values_for_available_streams(
    trajectory_bundle: TrajectoryBundle,
) -> None:
    """Available stream koordinatlarında en az bir non-zero, non-NaN değer olmalı."""
    # Validation split son step'ini kullan (tam history mevcut)
    last_validation_step = trajectory_bundle.splits["validation"][0].steps[-1]
    tensor = last_validation_step.observation.raw_surface["1m"]

    available_has_value = False
    for i in range(tensor.flat_size):
        if (
            not tensor.padding[i]
            and not tensor.unavailable_by_contract[i]
            and not tensor.missing[i]
            and tensor.values[i] != 0.0
        ):
            available_has_value = True
            break
    assert available_has_value, "Expected at least one non-zero value in available observation coords"
