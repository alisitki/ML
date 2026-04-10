from __future__ import annotations

from pathlib import Path

from quantlab_ml.common import dump_model, hash_payload, load_model
from quantlab_ml.contracts import EvaluationReport, PolicyArtifact, PolicyScore, TrajectoryBundle
from quantlab_ml.registry import LocalRegistryStore
from quantlab_ml.trajectories import TrajectoryStore


def test_contract_round_trip(
    tmp_path: Path,
    trajectory_bundle: TrajectoryBundle,
    policy_artifact: PolicyArtifact,
    evaluation_report: EvaluationReport,
    policy_score: PolicyScore,
    training_bundle: tuple,
) -> None:
    trajectories_path = tmp_path / "trajectories.json"
    policy_path = tmp_path / "policy.json"
    evaluation_path = tmp_path / "evaluation.json"
    score_path = tmp_path / "score.json"

    TrajectoryStore.write(trajectories_path, trajectory_bundle)
    dump_model(policy_path, policy_artifact)
    dump_model(evaluation_path, evaluation_report)
    dump_model(score_path, policy_score)

    loaded_bundle = TrajectoryStore.read(trajectories_path)
    loaded_policy = load_model(policy_path, PolicyArtifact)
    loaded_report = load_model(evaluation_path, EvaluationReport)
    loaded_score = load_model(score_path, PolicyScore)
    _, _, training_config = training_bundle

    registry = LocalRegistryStore(tmp_path / "registry")
    record = registry.register_candidate(
        loaded_policy,
        loaded_bundle,
        reward_config_hash=hash_payload(loaded_bundle.reward_spec),
        training_config_hash=hash_payload(training_config),
    )
    updated = registry.append_score(loaded_policy.policy_id, loaded_score, loaded_report)

    assert loaded_bundle.dataset_spec.dataset_hash == trajectory_bundle.dataset_spec.dataset_hash
    assert loaded_policy.policy_id == policy_artifact.policy_id
    assert loaded_report.policy_id == policy_artifact.policy_id
    assert loaded_report.infeasible_penalty_total == evaluation_report.infeasible_penalty_total
    assert updated.score_history[-1].score.composite_rank == policy_score.composite_rank
    assert record.coverage.covered_symbols == trajectory_bundle.dataset_spec.symbols
