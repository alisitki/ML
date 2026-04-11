from __future__ import annotations

from pathlib import Path

from quantlab_ml.common import hash_payload
from quantlab_ml.contracts import PolicyArtifact, PolicyScore, TrajectoryBundle
from quantlab_ml.registry import LocalRegistryStore


def test_registry_tracks_lineage_and_coverage(
    tmp_path: Path,
    trajectory_bundle: TrajectoryBundle,
    policy_artifact: PolicyArtifact,
    evaluation_report,
    policy_score: PolicyScore,
    training_bundle: tuple,
) -> None:
    _, _, training_config = training_bundle
    store = LocalRegistryStore(tmp_path / "registry")
    record = store.register_candidate(
        policy_artifact,
        trajectory_bundle,
        reward_config_hash=hash_payload(trajectory_bundle.reward_spec),
        training_config_hash=hash_payload(training_config),
    )
    updated = store.append_score(policy_artifact.policy_id, policy_score, evaluation_report)
    index = store.load_index()
    train_steps = sum(len(trajectory.steps) for trajectory in trajectory_bundle.splits["train"])

    assert index.champion_policy_id == policy_artifact.policy_id
    assert updated.coverage.train_sample_count == train_steps
    assert updated.coverage.eval_sample_count == evaluation_report.total_steps
    assert updated.coverage.reward_event_count == evaluation_report.total_steps
    assert updated.coverage.realized_trade_count == evaluation_report.realized_trade_count
    assert record.coverage.covered_venues == trajectory_bundle.dataset_spec.exchanges


def test_unscored_candidate_does_not_become_champion(
    tmp_path: Path,
    trajectory_bundle: TrajectoryBundle,
    policy_artifact: PolicyArtifact,
    training_bundle: tuple,
) -> None:
    _, _, training_config = training_bundle
    store = LocalRegistryStore(tmp_path / "registry")
    store.register_candidate(
        policy_artifact,
        trajectory_bundle,
        reward_config_hash=hash_payload(trajectory_bundle.reward_spec),
        training_config_hash=hash_payload(training_config),
    )
    index = store.load_index()
    record = store.get_record(policy_artifact.policy_id)

    assert index.champion_policy_id is None
    assert record is not None
    assert record.status == "candidate"
