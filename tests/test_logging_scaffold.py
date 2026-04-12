from __future__ import annotations

import logging
from pathlib import Path

from quantlab_ml.data import LocalFixtureSource
from quantlab_ml.registry import LocalRegistryStore
from quantlab_ml.training import LinearPolicyTrainer
from quantlab_ml.trajectories import TrajectoryBuilder


def _logger_messages(caplog, logger_name: str) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == logger_name]


def test_builder_emits_summary_logs(
    caplog,
    fixture_path: Path,
    dataset_spec,
    training_bundle,
    reward_spec,
) -> None:
    trajectory_spec, action_space, _ = training_bundle
    events = LocalFixtureSource(fixture_path).load_events(dataset_spec)
    builder = TrajectoryBuilder(dataset_spec, trajectory_spec, action_space, reward_spec)
    caplog.set_level(logging.INFO, logger="quantlab_ml.trajectories.builder")

    builder.build(events)

    messages = _logger_messages(caplog, "quantlab_ml.trajectories.builder")
    assert any("trajectory_build_started" in message for message in messages)
    assert any("trajectory_build_completed" in message for message in messages)


def test_trainer_emits_summary_logs(caplog, trajectory_bundle, training_bundle) -> None:
    _, _, training_config = training_bundle
    trainer = LinearPolicyTrainer(training_config)
    caplog.set_level(logging.INFO, logger="quantlab_ml.training.trainer")

    trainer.train_search(trajectory_bundle)

    messages = _logger_messages(caplog, "quantlab_ml.training.trainer")
    assert any("training_data_prepared" in message for message in messages)
    assert any("training_candidate_completed" in message for message in messages)
    assert any("training_search_completed" in message for message in messages)


def test_registry_emits_summary_logs(
    caplog,
    tmp_path: Path,
    trajectory_bundle,
    policy_artifact,
    policy_score,
    evaluation_report,
) -> None:
    registry = LocalRegistryStore(tmp_path / "registry")
    caplog.set_level(logging.INFO, logger="quantlab_ml.registry.store")

    registry.register_candidate(
        policy_artifact,
        trajectory_bundle,
        reward_config_hash="reward-config-hash",
        training_config_hash=policy_artifact.training_config_hash,
    )
    registry.append_score(policy_artifact.policy_id, policy_score, evaluation_report)

    messages = _logger_messages(caplog, "quantlab_ml.registry.store")
    assert any("registry_candidate_registered" in message for message in messages)
    assert any("registry_score_appended" in message for message in messages)
    assert any("registry_index_recomputed" in message for message in messages)
