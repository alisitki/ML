from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quantlab_ml.common import load_yaml
from quantlab_ml.contracts import ActionSpaceSpec, TrajectorySpec
from quantlab_ml.data import LocalFixtureSource
from quantlab_ml.models.features import observation_feature_vector
from quantlab_ml.models.linear_policy import LinearPolicyParameters
from quantlab_ml.policies import PolicyRuntimeBridge
from quantlab_ml.training import LinearPolicyTrainer, TrainingConfig
from quantlab_ml.trajectories import TrajectoryBuilder


def test_pytorch_training_matches_numpy_reference_on_default_profile(
    trajectory_bundle,
    training_bundle,
) -> None:
    _, _, training_config = training_bundle
    pytorch_artifact = LinearPolicyTrainer(training_config).train(trajectory_bundle)
    numpy_artifact = LinearPolicyTrainer(training_config, backend_name="numpy").train(trajectory_bundle)

    assert pytorch_artifact.policy_payload.runtime_adapter == numpy_artifact.policy_payload.runtime_adapter
    assert pytorch_artifact.training_summary["training_backend"] == "pytorch"
    assert numpy_artifact.training_summary["training_backend"] == "numpy"
    _assert_summary_parity(
        pytorch_artifact.training_summary,
        numpy_artifact.training_summary,
        compare_candidate_rank=False,
        compare_selected_candidate=True,
    )
    _assert_parameter_parity(pytorch_artifact, numpy_artifact)
    _assert_runtime_decision_parity(trajectory_bundle, pytorch_artifact, numpy_artifact)


def test_pytorch_candidate_search_matches_numpy_reference_where_fixture_margin_is_clear(
    trajectory_bundle,
    search_training_bundle,
) -> None:
    _, _, search_training_config = search_training_bundle
    pytorch_result = LinearPolicyTrainer(search_training_config).train_search(trajectory_bundle)
    numpy_result = LinearPolicyTrainer(search_training_config, backend_name="numpy").train_search(trajectory_bundle)

    assert pytorch_result.search_budget_summary == numpy_result.search_budget_summary
    assert len(pytorch_result.candidate_results) == len(numpy_result.candidate_results)

    pytorch_by_index = {candidate.candidate_index: candidate for candidate in pytorch_result.candidate_results}
    numpy_by_index = {candidate.candidate_index: candidate for candidate in numpy_result.candidate_results}
    assert pytorch_by_index.keys() == numpy_by_index.keys()

    for candidate_index in sorted(numpy_by_index):
        pytorch_candidate = pytorch_by_index[candidate_index]
        numpy_candidate = numpy_by_index[candidate_index]
        assert pytorch_candidate.candidate_spec == numpy_candidate.candidate_spec
        _assert_summary_parity(
            pytorch_candidate.artifact.training_summary,
            numpy_candidate.artifact.training_summary,
            compare_candidate_rank=False,
            compare_selected_candidate=False,
        )

    numpy_top = float(numpy_result.candidate_results[0].artifact.training_summary["selection_aggregate_total_net_return"])
    numpy_second = float(numpy_result.candidate_results[1].artifact.training_summary["selection_aggregate_total_net_return"])
    if abs(numpy_top - numpy_second) > 1e-6:
        assert [candidate.candidate_spec for candidate in pytorch_result.candidate_results] == [
            candidate.candidate_spec for candidate in numpy_result.candidate_results
        ]
    else:
        pytorch_top = float(
            pytorch_result.candidate_results[0].artifact.training_summary["selection_aggregate_total_net_return"]
        )
        assert pytorch_top == pytest.approx(numpy_top, abs=1e-8)


def test_pytorch_training_supports_production_profile_with_numpy_reference_parity(
    repo_root: Path,
    fixture_path: Path,
    dataset_spec,
    reward_spec,
) -> None:
    raw = load_yaml(repo_root / "configs" / "training" / "production.yaml")
    trajectory_spec = TrajectorySpec.model_validate(raw["trajectory"])
    action_space = ActionSpaceSpec.model_validate(raw["action_space"])
    training_config = TrainingConfig.model_validate(raw["trainer"])
    events = LocalFixtureSource(fixture_path).load_events(dataset_spec)
    bundle = TrajectoryBuilder(
        dataset_spec,
        trajectory_spec,
        action_space,
        reward_spec,
    ).build(events)

    pytorch_artifact = LinearPolicyTrainer(training_config).train(bundle)
    numpy_artifact = LinearPolicyTrainer(training_config, backend_name="numpy").train(bundle)
    strict_contract = pytorch_artifact.runtime_metadata.strict_runtime_contract
    observation = bundle.splits["validation"][0].steps[0].observation

    assert strict_contract is not None
    assert pytorch_artifact.training_summary["training_backend"] == "pytorch"
    assert [scale.label for scale in bundle.trajectory_spec.scale_preset] == ["1m", "5m", "15m", "60m"]
    assert strict_contract.expected_feature_dim == len(observation_feature_vector(observation))
    _assert_summary_parity(
        pytorch_artifact.training_summary,
        numpy_artifact.training_summary,
        compare_candidate_rank=False,
        compare_selected_candidate=True,
    )
    _assert_parameter_parity(pytorch_artifact, numpy_artifact)
    _assert_runtime_decision_parity(bundle, pytorch_artifact, numpy_artifact)


def _assert_summary_parity(
    pytorch_summary: dict[str, object],
    numpy_summary: dict[str, object],
    *,
    compare_candidate_rank: bool,
    compare_selected_candidate: bool,
) -> None:
    assert set(pytorch_summary) == set(numpy_summary)
    assert pytorch_summary["trainer_name"] == numpy_summary["trainer_name"]
    assert pytorch_summary["surface_version"] == numpy_summary["surface_version"]
    assert pytorch_summary["feature_dim"] == numpy_summary["feature_dim"]
    assert pytorch_summary["epochs"] == numpy_summary["epochs"]
    assert pytorch_summary["seed"] == numpy_summary["seed"]
    assert pytorch_summary["learning_rate"] == numpy_summary["learning_rate"]
    assert pytorch_summary["l2_weight"] == numpy_summary["l2_weight"]
    assert pytorch_summary["candidate_index"] == numpy_summary["candidate_index"]
    assert pytorch_summary["candidate_spec"] == numpy_summary["candidate_spec"]
    assert pytorch_summary["selection_protocol"] == numpy_summary["selection_protocol"]
    assert pytorch_summary["selection_fold_count"] == numpy_summary["selection_fold_count"]
    assert pytorch_summary["selection_aggregate_metric"] == numpy_summary["selection_aggregate_metric"]
    assert pytorch_summary["selection_split"] == numpy_summary["selection_split"]
    assert pytorch_summary["selection_metric"] == numpy_summary["selection_metric"]
    assert pytorch_summary["final_untouched_test_used"] is False
    assert numpy_summary["final_untouched_test_used"] is False
    assert pytorch_summary["learned_normalization_fit_split"] == "train"
    assert numpy_summary["learned_normalization_fit_split"] == "train"
    assert pytorch_summary["search_budget_summary"] == numpy_summary["search_budget_summary"]

    if compare_candidate_rank:
        assert pytorch_summary["candidate_rank"] == numpy_summary["candidate_rank"]
    if compare_selected_candidate:
        assert pytorch_summary["selected_candidate"] == numpy_summary["selected_candidate"]

    for field_name in (
        "train_step_count",
        "validation_step_count",
        "selection_fold_count",
        "best_epoch",
    ):
        assert pytorch_summary[field_name] == numpy_summary[field_name]

    for field_name in (
        "selection_aggregate_total_net_return",
        "selection_aggregate_composite_rank",
        "best_validation_total_net_return",
        "best_validation_composite_rank",
    ):
        assert float(pytorch_summary[field_name]) == pytest.approx(float(numpy_summary[field_name]), abs=1e-8)

    pytorch_loss = list(pytorch_summary["training_loss_history"])
    numpy_loss = list(numpy_summary["training_loss_history"])
    pytorch_validation = list(pytorch_summary["validation_objective_history"])
    numpy_validation = list(numpy_summary["validation_objective_history"])
    np.testing.assert_allclose(np.asarray(pytorch_loss), np.asarray(numpy_loss), atol=1e-8)
    np.testing.assert_allclose(np.asarray(pytorch_validation), np.asarray(numpy_validation), atol=1e-8)

    pytorch_fold_scores = list(pytorch_summary["candidate_fold_scores"])
    numpy_fold_scores = list(numpy_summary["candidate_fold_scores"])
    assert len(pytorch_fold_scores) == len(numpy_fold_scores)
    for pytorch_fold, numpy_fold in zip(pytorch_fold_scores, numpy_fold_scores, strict=True):
        assert pytorch_fold["fold_id"] == numpy_fold["fold_id"]
        assert pytorch_fold["validation_step_count"] == numpy_fold["validation_step_count"]
        assert float(pytorch_fold["validation_total_net_return"]) == pytest.approx(
            float(numpy_fold["validation_total_net_return"]),
            abs=1e-8,
        )
        assert float(pytorch_fold["validation_composite_rank"]) == pytest.approx(
            float(numpy_fold["validation_composite_rank"]),
            abs=1e-8,
        )


def _assert_parameter_parity(pytorch_artifact, numpy_artifact) -> None:
    pytorch_params = LinearPolicyParameters.model_validate_json(pytorch_artifact.policy_payload.blob)
    numpy_params = LinearPolicyParameters.model_validate_json(numpy_artifact.policy_payload.blob)

    assert pytorch_params.action_keys == numpy_params.action_keys
    assert pytorch_params.venue_choices == numpy_params.venue_choices
    assert pytorch_params.preferred_size_band == numpy_params.preferred_size_band
    assert pytorch_params.preferred_leverage_band == numpy_params.preferred_leverage_band
    np.testing.assert_allclose(np.asarray(pytorch_params.feature_mean), np.asarray(numpy_params.feature_mean), atol=1e-10)
    np.testing.assert_allclose(np.asarray(pytorch_params.feature_std), np.asarray(numpy_params.feature_std), atol=1e-10)
    np.testing.assert_allclose(np.asarray(pytorch_params.action_weight), np.asarray(numpy_params.action_weight), atol=1e-8)
    np.testing.assert_allclose(np.asarray(pytorch_params.action_bias), np.asarray(numpy_params.action_bias), atol=1e-8)
    np.testing.assert_allclose(np.asarray(pytorch_params.venue_weight), np.asarray(numpy_params.venue_weight), atol=1e-8)
    np.testing.assert_allclose(np.asarray(pytorch_params.venue_bias), np.asarray(numpy_params.venue_bias), atol=1e-8)


def _assert_runtime_decision_parity(bundle, pytorch_artifact, numpy_artifact) -> None:
    observation = bundle.splits["validation"][0].steps[0].observation
    bridge = PolicyRuntimeBridge()
    pytorch_decision = bridge.decide(pytorch_artifact, observation)
    numpy_decision = bridge.decide(numpy_artifact, observation)

    assert pytorch_decision.action_key == numpy_decision.action_key
    assert pytorch_decision.venue == numpy_decision.venue
    assert pytorch_decision.size_band_key == numpy_decision.size_band_key
    assert pytorch_decision.leverage_band_key == numpy_decision.leverage_band_key
    assert pytorch_decision.confidence == pytest.approx(numpy_decision.confidence, abs=1e-8)
