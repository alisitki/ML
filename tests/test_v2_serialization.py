"""test_v2_serialization.py

Yeni trajectory bundle ve policy artifact kontratlarının
yazılıp okunabildiğini (round-trip) doğrular.
"""
from __future__ import annotations

from pathlib import Path

from quantlab_ml.common import dump_model, hash_payload, load_model
from quantlab_ml.contracts import EvaluationReport, PolicyArtifact, TrajectoryBundle
from quantlab_ml.registry import LocalRegistryStore
from quantlab_ml.trajectories import TrajectoryStore


def test_v2_trajectory_bundle_round_trip(
    tmp_path: Path,
    trajectory_bundle: TrajectoryBundle,
) -> None:
    path = tmp_path / "v2-trajectories.json"
    TrajectoryStore.write(path, trajectory_bundle)
    loaded = TrajectoryStore.read(path)

    assert loaded.dataset_spec.dataset_hash == trajectory_bundle.dataset_spec.dataset_hash
    assert loaded.split_artifact.split_version == trajectory_bundle.split_artifact.split_version
    # V2 scale_preset round-trips correctly
    orig_labels = [s.label for s in trajectory_bundle.trajectory_spec.scale_preset]
    loaded_labels = [s.label for s in loaded.trajectory_spec.scale_preset]
    assert orig_labels == loaded_labels
    # raw_surface round-trips
    orig_step = trajectory_bundle.splits["train"][0].steps[0]
    loaded_step = loaded.splits["train"][0].steps[0]
    assert orig_step.observation.raw_surface.keys() == loaded_step.observation.raw_surface.keys()
    orig_tensor = orig_step.observation.raw_surface["1m"]
    loaded_tensor = loaded_step.observation.raw_surface["1m"]
    assert orig_tensor.shape == loaded_tensor.shape
    assert orig_tensor.values == loaded_tensor.values


def test_v2_policy_artifact_round_trip(
    tmp_path: Path,
    policy_artifact: PolicyArtifact,
) -> None:
    path = tmp_path / "v2-policy.json"
    dump_model(path, policy_artifact)
    loaded = load_model(path, PolicyArtifact)
    assert loaded.policy_id == policy_artifact.policy_id
    assert loaded.artifact_id == policy_artifact.artifact_id
    assert loaded.runtime_metadata.allowed_venues == policy_artifact.runtime_metadata.allowed_venues
    assert loaded.training_summary.get("surface_version") == "v2"


def test_v2_evaluation_round_trip(
    tmp_path: Path,
    trajectory_bundle: TrajectoryBundle,
    policy_artifact: PolicyArtifact,
    evaluation_report: EvaluationReport,
    policy_score,
    training_bundle: tuple,
) -> None:
    eval_path = tmp_path / "v2-evaluation.json"
    dump_model(eval_path, evaluation_report)
    loaded_report = load_model(eval_path, EvaluationReport)
    assert loaded_report.policy_id == policy_artifact.policy_id
    assert loaded_report.infeasible_penalty_total == evaluation_report.infeasible_penalty_total


def test_v2_registry_stores_v2_bundle(
    tmp_path: Path,
    trajectory_bundle: TrajectoryBundle,
    policy_artifact: PolicyArtifact,
    policy_score,
    evaluation_report: EvaluationReport,
    training_bundle: tuple,
) -> None:
    _, _, training_config = training_bundle
    registry = LocalRegistryStore(tmp_path / "registry")
    record = registry.register_candidate(
        policy_artifact,
        trajectory_bundle,
        reward_config_hash=hash_payload(trajectory_bundle.reward_spec),
        training_config_hash=hash_payload(training_config),
    )
    updated = registry.append_score(policy_artifact.policy_id, policy_score, evaluation_report)
    assert record.coverage.covered_symbols == trajectory_bundle.dataset_spec.symbols
    assert updated.score_history[-1].composite_rank == policy_score.composite_rank
    assert updated.search_budget_summary.total_candidate_count == 1
