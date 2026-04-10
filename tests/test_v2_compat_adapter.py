"""test_v2_compat_adapter.py

Mevcut baseline trainer/runtime'ın yeni bundle'ı
legacy adapter üzerinden hâlâ tüketebildiğini doğrular.
"""
from __future__ import annotations

from quantlab_ml.contracts import TrajectoryBundle
from quantlab_ml.contracts.compat import flat_action_mask, flat_value_cube, target_stream_series
from quantlab_ml.training.compat_adapter import V2toV1BundleAdapter


def test_v2to_v1_adapter_exposes_train_steps(trajectory_bundle: TrajectoryBundle) -> None:
    adapter = V2toV1BundleAdapter(trajectory_bundle)
    steps = adapter.train_steps()
    assert len(steps) > 0


def test_mark_price_series_via_adapter(trajectory_bundle: TrajectoryBundle) -> None:
    adapter = V2toV1BundleAdapter(trajectory_bundle)
    steps = adapter.train_steps()
    series = steps[-1].mark_price_series()
    # Seri uzunluğu ilk scale'in bucket sayısına eşit
    expected_len = trajectory_bundle.trajectory_spec.scale_preset[0].num_buckets
    assert len(series) == expected_len


def test_flat_action_mask_abstain_always_true(trajectory_bundle: TrajectoryBundle) -> None:
    adapter = V2toV1BundleAdapter(trajectory_bundle)
    for step_view in adapter.train_steps():
        mask = step_view.flat_action_mask()
        assert mask.get("abstain") is True, "abstain must always be feasible"


def test_compat_target_stream_series_length(trajectory_bundle: TrajectoryBundle) -> None:
    first_step = trajectory_bundle.splits["train"][0].steps[0]
    series = target_stream_series(first_step.observation, "mark_price")
    expected_len = trajectory_bundle.trajectory_spec.scale_preset[0].num_buckets
    assert len(series) == expected_len


def test_compat_flat_value_cube_length(trajectory_bundle: TrajectoryBundle) -> None:
    first_step = trajectory_bundle.splits["train"][0].steps[0]
    cube = flat_value_cube(first_step.observation)
    tensor = first_step.observation.raw_surface["1m"]
    assert len(cube) == tensor.flat_size


def test_trainer_runs_with_v2_bundle(
    trajectory_bundle: TrajectoryBundle, policy_artifact
) -> None:
    """MomentumBaselineTrainer V2 bundle'ı adapter üzerinden işleyebilmeli."""
    assert policy_artifact.policy_id.startswith("policy-")
    assert policy_artifact.training_summary.get("surface_version") == "v2"
    assert policy_artifact.training_summary.get("train_step_count", 0) > 0
