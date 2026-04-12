from __future__ import annotations

from datetime import datetime, timezone

from quantlab_ml.contracts import DatasetSpec
from quantlab_ml.trajectories import TrajectoryBuilder


def test_bundle_persists_split_v1_artifact(trajectory_bundle) -> None:
    assert set(trajectory_bundle.splits) == {"train", "validation", "final_untouched_test"}
    artifact = trajectory_bundle.split_artifact
    assert artifact.split_version == "split_v1_walkforward"
    assert artifact.purge_width_steps == trajectory_bundle.reward_spec.horizon_steps
    assert artifact.embargo_width_steps == trajectory_bundle.reward_spec.horizon_steps
    assert artifact.train_window.start == trajectory_bundle.dataset_spec.train_range.start
    assert artifact.validation_window.end == trajectory_bundle.dataset_spec.validation_range.end
    assert (
        artifact.final_untouched_test_window.end
        == trajectory_bundle.dataset_spec.final_untouched_test_range.end
    )
    assert len(artifact.folds) >= 1


def test_walkforward_fold_generation_is_deterministic(training_bundle, reward_spec) -> None:
    trajectory_spec, action_space, _ = training_bundle
    dataset_spec = DatasetSpec.model_validate(
        {
            "dataset_hash": "split-test",
            "slice_id": "split-test-slice",
            "exchanges": ["binance"],
            "symbols": ["BTCUSDT"],
            "stream_universe": ["mark_price"],
            "available_streams_by_exchange": {"binance": ["mark_price"]},
            "train_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T00:03:00Z"},
            "validation_range": {"start": "2024-01-01T00:04:00Z", "end": "2024-01-01T00:07:00Z"},
            "final_untouched_test_range": {
                "start": "2024-01-01T00:08:00Z",
                "end": "2024-01-01T00:09:00Z",
            },
            "walkforward": {
                "train_window_steps": 3,
                "validation_window_steps": 2,
                "step_size_steps": 1,
            },
            "sampling_interval_seconds": 60,
        }
    )
    builder = TrajectoryBuilder(dataset_spec, trajectory_spec, action_space, reward_spec)
    folds = builder._generate_walkforward_folds(
        builder._timestamps(dataset_spec.development_range),
        purge_width_steps=reward_spec.horizon_steps,
        embargo_width_steps=reward_spec.horizon_steps,
    )

    assert [fold.fold_id for fold in folds] == ["wf-00", "wf-01"]
    assert folds[0].train_window.start == datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    assert folds[0].train_window.end == datetime(2024, 1, 1, 0, 2, tzinfo=timezone.utc)
    assert folds[0].validation_window.start == datetime(2024, 1, 1, 0, 3, tzinfo=timezone.utc)
    assert folds[0].validation_window.end == datetime(2024, 1, 1, 0, 4, tzinfo=timezone.utc)
    assert folds[1].validation_window.start == datetime(2024, 1, 1, 0, 6, tzinfo=timezone.utc)
    assert folds[1].validation_window.end == datetime(2024, 1, 1, 0, 7, tzinfo=timezone.utc)
    assert all(fold.purge_width_steps == reward_spec.horizon_steps for fold in folds)
    assert all(fold.embargo_width_steps == reward_spec.horizon_steps for fold in folds)


def test_final_untouched_test_split_is_not_default_evaluation_target(
    trajectory_bundle,
    policy_artifact,
    evaluation_boundary,
) -> None:
    from quantlab_ml.evaluation import EvaluationEngine

    report = EvaluationEngine(evaluation_boundary).evaluate(trajectory_bundle, policy_artifact)

    validation_steps = sum(len(item.steps) for item in trajectory_bundle.splits["validation"])
    final_test_steps = sum(len(item.steps) for item in trajectory_bundle.splits["final_untouched_test"])
    assert report.total_steps == validation_steps
    assert final_test_steps > 0
    assert report.total_steps != validation_steps + final_test_steps
