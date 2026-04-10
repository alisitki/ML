"""test_v2_observation_shape.py

V2 yüzeyinin [scale, time, symbol, exchange, stream, field] düzenini
ve paralel age/mask tensor boylarını doğrular.
"""
from __future__ import annotations

from quantlab_ml.contracts import TrajectoryBundle


def test_observation_shape_matches_schema(trajectory_bundle: TrajectoryBundle) -> None:
    schema = trajectory_bundle.observation_schema
    first_step = trajectory_bundle.splits["train"][0].steps[0]
    obs = first_step.observation

    for scale_spec in schema.scale_axis:
        tensor = obs.raw_surface[scale_spec.label]
        n_t = scale_spec.num_buckets
        n_sym = len(schema.asset_axis)
        n_exc = len(schema.exchange_axis)
        n_str = len(schema.stream_axis)
        n_fld = sum(len(schema.field_axis.get(s, [])) for s in schema.stream_axis)
        expected_flat = n_t * n_sym * n_exc * n_str * n_fld

        assert tensor.flat_size == expected_flat, (
            f"scale={scale_spec.label} flat_size mismatch: {tensor.flat_size} != {expected_flat}"
        )
        assert len(tensor.values) == expected_flat
        assert len(tensor.age) == expected_flat
        assert len(tensor.padding) == expected_flat
        assert len(tensor.unavailable_by_contract) == expected_flat
        assert len(tensor.missing) == expected_flat
        assert len(tensor.stale) == expected_flat


def test_scale_axis_labels_match_config(trajectory_bundle: TrajectoryBundle) -> None:
    schema = trajectory_bundle.observation_schema
    labels = [s.label for s in schema.scale_axis]
    # Fixture config: 1m×4 tek scale
    assert labels == ["1m"]
    assert schema.scale_axis[0].num_buckets == 4
    assert schema.scale_axis[0].resolution_seconds == 60


def test_field_axis_covers_all_streams(trajectory_bundle: TrajectoryBundle) -> None:
    schema = trajectory_bundle.observation_schema
    for stream in schema.stream_axis:
        assert stream in schema.field_axis, f"field_axis missing stream: {stream}"
        assert len(schema.field_axis[stream]) > 0, f"field_axis[{stream}] is empty"


def test_shape_attribute_consistent(trajectory_bundle: TrajectoryBundle) -> None:
    first_step = trajectory_bundle.splits["train"][0].steps[0]
    for scale_label, tensor in first_step.observation.raw_surface.items():
        assert len(tensor.shape) == 5, f"scale={scale_label} shape must be 5D"
        product = 1
        for dim in tensor.shape:
            product *= dim
        assert product == tensor.flat_size
