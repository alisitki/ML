from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from quantlab_ml.common import hash_payload
from quantlab_ml.contracts import POLICY_ARTIFACT_SCHEMA_VERSION, DatasetSpec
from quantlab_ml.data import LocalFixtureSource
from quantlab_ml.training import CandidateSearchConfig, LinearPolicyTrainer
from quantlab_ml.policies import PolicyRuntimeBridge
from quantlab_ml.training.trainer import _resolve_torch_device
from quantlab_ml.trajectories import TrajectoryBuilder


def test_training_loop_records_search_budget_and_validation_selection(policy_artifact) -> None:
    summary = policy_artifact.training_summary
    search_budget = summary.get("search_budget_summary", {})
    strict_contract = policy_artifact.runtime_metadata.strict_runtime_contract
    fold_scores = summary.get("candidate_fold_scores", [])

    assert policy_artifact.artifact_version == POLICY_ARTIFACT_SCHEMA_VERSION
    assert policy_artifact.policy_family == "linear-policy-trainer"
    assert policy_artifact.policy_payload.runtime_adapter == "linear-policy-v1"
    assert policy_artifact.training_run_id.startswith("trainrun-")
    assert strict_contract is not None
    assert summary.get("selection_split") == "validation"
    assert summary.get("selection_metric") == "total_net_return"
    assert summary.get("final_untouched_test_used") is False
    assert summary.get("learned_normalization_fit_split") == "train"
    assert summary.get("training_backend") == "pytorch"
    assert summary.get("training_device") in {"cpu", "cuda"}
    assert summary.get("cuda_available") == (summary.get("training_device") == "cuda")
    assert summary.get("device_name")
    assert summary.get("selection_protocol") == "walkforward_cv_then_canonical_refit"
    assert summary.get("selection_fold_count") == len(fold_scores)
    assert summary.get("selection_aggregate_metric") == "step_weighted_mean_validation_total_net_return"
    assert summary.get("selection_aggregate_total_net_return") is not None
    assert summary.get("selection_aggregate_composite_rank") is not None
    assert summary.get("best_epoch", 0) >= 1
    assert summary.get("candidate_index") == 0
    assert summary.get("candidate_rank") == 1
    assert summary.get("selected_candidate") is True
    assert search_budget.get("tried_models") == 1
    assert search_budget.get("total_candidate_count") == 1
    assert fold_scores
    assert all(score["validation_step_count"] > 0 for score in fold_scores)
    assert _search_tag_map(policy_artifact)["search_selected"] == "true"
    assert _search_tag_map(policy_artifact)["search_candidate_rank"] == "1"
    assert _search_tag_map(policy_artifact)["runtime_contract"] == strict_contract.runtime_contract_version
    assert _search_tag_map(policy_artifact)["policy_kind"] == "linear-policy-v1"
    assert _search_tag_map(policy_artifact)["derived_contract"] == strict_contract.derived_contract_version
    assert _search_tag_map(policy_artifact)["feature_dim"] == str(strict_contract.expected_feature_dim)
    assert _search_tag_map(policy_artifact)["compat_mode"] == "strict"


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
        assert summary.get("training_backend") == "pytorch"
        assert summary.get("training_device") in {"cpu", "cuda"}
        assert summary.get("cuda_available") == (summary.get("training_device") == "cuda")
        assert summary.get("device_name")
        assert summary.get("selection_protocol") == "walkforward_cv_then_canonical_refit"
        assert summary.get("selection_fold_count", 0) >= 1
        assert summary.get("candidate_fold_scores")
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


def test_walkforward_fold_bundle_uses_development_records_and_applies_purge(
    tmp_path: Path,
    training_bundle,
    reward_spec,
) -> None:
    trajectory_spec, action_space, training_config = training_bundle
    dataset_spec = DatasetSpec.model_validate(
        {
            "dataset_hash": "walkforward-training-test",
            "slice_id": "walkforward-training-test",
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
    events_path = tmp_path / "mark-price-events.ndjson"
    events_path.write_text(
        "\n".join(
            (
                f'{{"event_time":"2024-01-01T00:{minute:02d}:00Z","exchange":"binance","symbol":"BTCUSDT",'
                f'"stream_type":"mark_price","price":{100.0 + minute}}}'
            )
            for minute in range(10)
        )
        + "\n",
        encoding="utf-8",
    )
    events = LocalFixtureSource(events_path).load_events(dataset_spec)
    bundle = TrajectoryBuilder(dataset_spec, trajectory_spec, action_space, reward_spec).build(events)

    trainer = LinearPolicyTrainer(training_config)
    first_fold = bundle.split_artifact.folds[0]
    second_fold = bundle.split_artifact.folds[1]
    first_fold_bundle = trainer._build_fold_bundle(bundle, first_fold)
    second_fold_bundle = trainer._build_fold_bundle(bundle, second_fold)

    first_train_times = [step.event_time.isoformat() for record in first_fold_bundle.splits["train"] for step in record.steps]
    first_validation_times = [
        step.event_time.isoformat() for record in first_fold_bundle.splits["validation"] for step in record.steps
    ]
    second_train_times = [
        step.event_time.isoformat() for record in second_fold_bundle.splits["train"] for step in record.steps
    ]

    assert "2024-01-01T00:02:00+00:00" not in first_train_times
    assert "2024-01-01T00:03:00+00:00" in first_validation_times
    assert "2024-01-01T00:04:00+00:00" in first_validation_times
    assert "2024-01-01T00:05:00+00:00" not in second_train_times
    assert "2024-01-01T00:04:00+00:00" in second_train_times
    assert all(step.event_time.isoformat() != "2024-01-01T00:07:00+00:00" for record in second_fold_bundle.splits["validation"] for step in record.steps)


def test_resolve_torch_device_prefers_cuda_when_available() -> None:
    class FakeCudaModule:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(index: int) -> str:
            assert index == 0
            return "Fake GPU"

    class FakeTorchModule:
        cuda = FakeCudaModule()

        @staticmethod
        def device(label: str) -> str:
            return label

    resolution = _resolve_torch_device(FakeTorchModule())

    assert resolution.training_device == "cuda"
    assert resolution.cuda_available is True
    assert resolution.device_name == "Fake GPU"
    assert resolution.compute_device == "cuda"


def test_resolve_torch_device_falls_back_to_cpu_when_cuda_missing() -> None:
    class FakeCudaModule:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorchModule:
        cuda = FakeCudaModule()

        @staticmethod
        def device(label: str) -> str:
            return label

    resolution = _resolve_torch_device(FakeTorchModule())

    assert resolution.training_device == "cpu"
    assert resolution.cuda_available is False
    assert resolution.device_name == "cpu"
    assert resolution.compute_device == "cpu"


def _search_tag_map(policy_artifact) -> dict[str, str]:
    tag_map: dict[str, str] = {}
    for tag in policy_artifact.runtime_metadata.artifact_compatibility_tags:
        if ":" not in tag:
            continue
        key, value = tag.split(":", 1)
        tag_map[key] = value
    return tag_map
