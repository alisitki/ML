from __future__ import annotations

from types import SimpleNamespace

from quantlab_ml.common import hash_payload
from quantlab_ml.training import CandidateSearchConfig, LinearPolicyTrainer
from quantlab_ml.policies import PolicyRuntimeBridge


def test_training_loop_records_search_budget_and_validation_selection(policy_artifact) -> None:
    summary = policy_artifact.training_summary
    search_budget = summary.get("search_budget_summary", {})

    assert policy_artifact.policy_family == "linear-policy-trainer"
    assert policy_artifact.policy_payload.runtime_adapter == "linear-policy-v1"
    assert policy_artifact.training_run_id.startswith("trainrun-")
    assert summary.get("selection_split") == "validation"
    assert summary.get("selection_metric") == "total_net_return"
    assert summary.get("final_untouched_test_used") is False
    assert summary.get("learned_normalization_fit_split") == "train"
    assert summary.get("best_epoch", 0) >= 1
    assert summary.get("candidate_index") == 0
    assert summary.get("candidate_rank") == 1
    assert summary.get("selected_candidate") is True
    assert search_budget.get("tried_models") == 1
    assert search_budget.get("total_candidate_count") == 1
    assert _search_tag_map(policy_artifact)["search_selected"] == "true"
    assert _search_tag_map(policy_artifact)["search_candidate_rank"] == "1"


def test_train_search_explicit_candidate_search_produces_ranked_candidates(
    trajectory_bundle,
    search_training_bundle,
) -> None:
    _, _, search_training_config = search_training_bundle
    trainer = LinearPolicyTrainer(search_training_config)

    search_result = trainer.train_search(trajectory_bundle)

    assert search_result.training_run_id.startswith("trainrun-")
    assert search_result.search_budget_summary.total_candidate_count == 4
    assert search_result.search_budget_summary.tried_models == 4
    assert search_result.search_budget_summary.tried_seeds == 2
    assert search_result.search_budget_summary.tried_hyperparameter_variants == 2
    assert len(search_result.candidate_results) == 4
    assert search_result.selected_artifact.policy_id == search_result.candidate_results[0].artifact.policy_id

    for expected_rank, candidate in enumerate(search_result.candidate_results, start=1):
        summary = candidate.artifact.training_summary
        tag_map = _search_tag_map(candidate.artifact)
        assert candidate.candidate_rank == expected_rank
        assert summary.get("training_run_id") == search_result.training_run_id
        assert summary.get("candidate_rank") == expected_rank
        assert summary.get("candidate_index") == candidate.candidate_index
        assert summary.get("candidate_spec") == candidate.candidate_spec.as_dict()
        assert summary.get("best_validation_composite_rank") == candidate.best_validation_composite_rank
        assert summary.get("search_budget_summary", {}).get("total_candidate_count") == 4
        assert tag_map["search_run_id"] == search_result.training_run_id
        assert tag_map["search_candidate_index"] == str(candidate.candidate_index)
        assert tag_map["search_candidate_rank"] == str(expected_rank)
        assert tag_map["search_selected"] == str(candidate.selected_candidate).lower()


def test_train_search_uses_higher_composite_rank_as_tie_break(
    trajectory_bundle,
    training_bundle,
    monkeypatch,
) -> None:
    _, _, base_training_config = training_bundle
    search_config = base_training_config.model_copy(
        update={
            "candidate_search": CandidateSearchConfig(
                seeds=[17, 23],
                learning_rates=[base_training_config.learning_rate],
                l2_weights=[base_training_config.l2_weight],
            )
        }
    )
    preferred_config = search_config.model_copy(
        update={
            "seed": 23,
            "candidate_search": None,
        }
    )
    preferred_hash = hash_payload(preferred_config)
    trainer = LinearPolicyTrainer(search_config)

    def fake_validation_report(_, bundle, artifact):
        step_reward_std = 0.1 if artifact.training_config_hash == preferred_hash else 10.0
        return SimpleNamespace(
            policy_id=artifact.policy_id,
            evaluation_id=f"eval-{artifact.policy_id}",
            total_steps=10,
            realized_trade_count=0,
            infeasible_action_count=0,
            infeasible_penalty_total=0.0,
            total_net_return=0.5,
            average_net_return=0.05,
            risk_penalty_total=0.0,
            turnover_penalty_total=0.0,
            fee_total=0.0,
            funding_total=0.0,
            slippage_total=0.0,
            action_counts={},
            step_reward_std=step_reward_std,
            coverage_symbols=bundle.dataset_spec.symbols,
            coverage_venues=bundle.dataset_spec.exchanges,
            coverage_streams=bundle.dataset_spec.stream_universe,
            active_date_range=bundle.dataset_spec.validation_range,
            notes=[],
        )

    monkeypatch.setattr(LinearPolicyTrainer, "_validation_report", fake_validation_report)

    search_result = trainer.train_search(trajectory_bundle)

    assert search_result.candidate_results[0].candidate_spec.seed == 23
    assert search_result.candidate_results[0].best_validation_total_net_return == 0.5
    assert (
        search_result.candidate_results[0].best_validation_composite_rank
        > search_result.candidate_results[1].best_validation_composite_rank
    )


def test_linear_policy_runtime_decision_preserves_explicit_decision_dimensions(
    trajectory_bundle,
    policy_artifact,
) -> None:
    observation = trajectory_bundle.splits["validation"][0].steps[0].observation
    decision = PolicyRuntimeBridge().decide(policy_artifact, observation)

    if decision.action_key == "abstain":
        assert decision.venue is None
        assert decision.size_band_key is None
        assert decision.leverage_band_key is None
    else:
        assert decision.venue in policy_artifact.allowed_venues
        assert decision.size_band_key == policy_artifact.runtime_metadata.size_bounds.key
        assert decision.leverage_band_key == policy_artifact.runtime_metadata.leverage_bounds.key


def _search_tag_map(policy_artifact) -> dict[str, str]:
    tag_map: dict[str, str] = {}
    for tag in policy_artifact.runtime_metadata.artifact_compatibility_tags:
        if ":" not in tag:
            continue
        key, value = tag.split(":", 1)
        tag_map[key] = value
    return tag_map
