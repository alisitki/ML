from __future__ import annotations

from pathlib import Path

from quantlab_ml.common import hash_payload
from quantlab_ml.contracts import (
    LEGACY_POLICY_ARTIFACT_SCHEMA_VERSION,
    PaperSimEvidenceRecord,
    PolicyArtifact,
    PolicyScore,
    PromotionEvidence,
    ReproducibilityMetadata,
    TrajectoryBundle,
)
from quantlab_ml.registry import LocalRegistryStore, audit_registry_continuity
from quantlab_ml.training import LinearPolicyTrainer


def test_scored_candidate_becomes_challenger_until_promotion(
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

    assert index.champion_policy_id is None
    assert policy_artifact.policy_id in index.challenger_policy_ids
    assert updated.status == "challenger"
    assert updated.coverage.train_sample_count == train_steps
    assert updated.coverage.eval_sample_count == evaluation_report.total_steps
    assert updated.coverage.reward_event_count == evaluation_report.total_steps
    assert updated.coverage.realized_trade_count == evaluation_report.realized_trade_count
    assert record.coverage.covered_venues == trajectory_bundle.dataset_spec.exchanges
    assert updated.score_history[-1].composite_rank == policy_score.composite_rank
    assert _tag_map(updated)["search_selected"] == "true"
    assert _tag_map(updated)["search_candidate_rank"] == "1"


def test_registry_records_search_linkage_for_multi_candidate_run(
    tmp_path: Path,
    trajectory_bundle: TrajectoryBundle,
    search_training_bundle: tuple,
) -> None:
    _, _, search_training_config = search_training_bundle
    trainer = LinearPolicyTrainer(search_training_config)
    search_result = trainer.train_search(trajectory_bundle)
    store = LocalRegistryStore(tmp_path / "registry")

    for candidate in search_result.candidate_results:
        store.register_candidate(
            candidate.artifact,
            trajectory_bundle,
            reward_config_hash=hash_payload(trajectory_bundle.reward_spec),
            training_config_hash=candidate.artifact.training_config_hash,
        )

    selected_records = []
    for candidate in search_result.candidate_results:
        record = store.get_record(candidate.artifact.policy_id)
        assert record is not None
        tags = _tag_map(record)
        assert tags["search_run_id"] == search_result.training_run_id
        assert tags["search_candidate_index"] == str(candidate.candidate_index)
        assert tags["search_candidate_rank"] == str(candidate.candidate_rank)
        assert tags["search_selected"] == str(candidate.selected_candidate).lower()
        if tags["search_selected"] == "true":
            selected_records.append(record.policy_id)

    assert selected_records == [search_result.selected_artifact.policy_id]


def test_registry_continuity_audit_counts_numpy_and_legacy_compat_dependencies(
    tmp_path: Path,
    trajectory_bundle: TrajectoryBundle,
    training_bundle: tuple,
) -> None:
    _, _, training_config = training_bundle
    numpy_artifact = LinearPolicyTrainer(training_config, backend_name="numpy").train(trajectory_bundle)
    store = LocalRegistryStore(tmp_path / "registry")
    reward_hash = hash_payload(trajectory_bundle.reward_spec)
    training_hash = hash_payload(training_config)
    store.register_candidate(
        numpy_artifact,
        trajectory_bundle,
        reward_config_hash=reward_hash,
        training_config_hash=training_hash,
    )

    legacy_artifact = _legacy_linear_artifact(numpy_artifact)
    legacy_artifact = legacy_artifact.model_copy(
        update={
            "policy_id": f"{legacy_artifact.policy_id}-legacy",
            "artifact_id": f"{legacy_artifact.artifact_id}-legacy",
            "training_run_id": f"{legacy_artifact.training_run_id}-legacy",
        },
        deep=True,
    )
    store.register_candidate(
        legacy_artifact,
        trajectory_bundle,
        reward_config_hash=reward_hash,
        training_config_hash=training_hash,
    )

    summary = audit_registry_continuity(store)

    assert summary["record_count"] == 2
    assert summary["active_record_count"] == 2
    assert summary["active_training_backend_counts"] == {"numpy": 2}
    assert summary["active_numpy_training_backend_count"] == 2
    assert summary["active_legacy_compat_artifact_count"] == 1
    assert summary["ready_to_close_numpy_continuity_window"] is False
    assert summary["ready_to_retire_legacy_compat_window"] is False


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


def test_promotion_gate_requires_paper_sim_and_runtime_artifact(
    tmp_path: Path,
    trajectory_bundle: TrajectoryBundle,
    policy_artifact: PolicyArtifact,
    evaluation_report,
    policy_score: PolicyScore,
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
    store.append_score(policy_artifact.policy_id, policy_score, evaluation_report)
    paper_sim_evidence = _record_paper_sim_evidence(store, policy_artifact.policy_id, tmp_path / "paper-sim.md")

    decision = store.promote_candidate(
        policy_artifact.policy_id,
        evidence=_promotion_evidence(
            policy_artifact,
            paper_sim_evidence_id=paper_sim_evidence.evidence_id,
            deployment_artifact_path=str(tmp_path / "missing-runtime-artifact.json"),
        ),
    )
    record = store.get_record(policy_artifact.policy_id)

    assert decision.decision == "reject"
    assert "artifacts.deployment_artifact_exists" in decision.failure_reasons
    assert record is not None
    assert record.status == "challenger"
    assert store.load_index().champion_policy_id is None


def test_record_paper_sim_evidence_links_report_to_registry_lineage(
    tmp_path: Path,
    trajectory_bundle: TrajectoryBundle,
    policy_artifact: PolicyArtifact,
    evaluation_report,
    policy_score: PolicyScore,
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
    store.append_score(policy_artifact.policy_id, policy_score, evaluation_report)

    evidence = _record_paper_sim_evidence(store, policy_artifact.policy_id, tmp_path / "paper-sim.md")
    reloaded_record = store.get_record(policy_artifact.policy_id)
    reloaded_evidence = store.get_paper_sim_evidence(evidence.evidence_id)

    assert reloaded_record is not None
    assert reloaded_record.paper_sim_evidence_id == evidence.evidence_id
    assert reloaded_evidence is not None
    assert reloaded_evidence.evaluation_report_id == evaluation_report.evaluation_id
    assert reloaded_evidence.policy_id == policy_artifact.policy_id


def test_promotion_gate_promotes_scored_candidate_with_complete_evidence(
    tmp_path: Path,
    trajectory_bundle: TrajectoryBundle,
    policy_artifact: PolicyArtifact,
    evaluation_report,
    policy_score: PolicyScore,
    training_bundle: tuple,
) -> None:
    _, _, training_config = training_bundle
    promotable_report = evaluation_report.model_copy(
        update={
            "total_net_return": 0.5,
            "average_net_return": 0.1,
        }
    )
    promotable_score = policy_score.model_copy(
        update={
            "expected_return_score": 0.1,
            "composite_rank": max(policy_score.composite_rank, 0.9),
        }
    )
    store = LocalRegistryStore(tmp_path / "registry")
    store.register_candidate(
        policy_artifact,
        trajectory_bundle,
        reward_config_hash=hash_payload(trajectory_bundle.reward_spec),
        training_config_hash=hash_payload(training_config),
    )
    store.append_score(policy_artifact.policy_id, promotable_score, promotable_report)

    deployment_artifact = tmp_path / "inference-artifact.json"
    deployment_artifact.write_text("{}", encoding="utf-8")
    paper_sim_evidence = _record_paper_sim_evidence(store, policy_artifact.policy_id, tmp_path / "paper-sim.md")

    decision = store.promote_candidate(
        policy_artifact.policy_id,
        evidence=_promotion_evidence(
            policy_artifact,
            deployment_artifact_path=str(deployment_artifact),
            paper_sim_evidence_id=paper_sim_evidence.evidence_id,
        ),
    )
    record = store.get_record(policy_artifact.policy_id)
    index = store.load_index()

    assert decision.decision == "promote"
    assert decision.failure_reasons == []
    assert record is not None
    assert record.status == "champion"
    assert record.paper_sim_evidence_id == paper_sim_evidence.evidence_id
    assert record.deployment_artifact_path == str(deployment_artifact)
    assert index.champion_policy_id == policy_artifact.policy_id


def test_promotion_gate_requires_champion_comparison_evidence_for_challenger(
    tmp_path: Path,
    trajectory_bundle: TrajectoryBundle,
    policy_artifact: PolicyArtifact,
    evaluation_report,
    policy_score: PolicyScore,
    training_bundle: tuple,
) -> None:
    _, _, training_config = training_bundle
    champion_report = evaluation_report.model_copy(
        update={
            "total_net_return": 0.4,
            "average_net_return": 0.08,
        }
    )
    champion_score = policy_score.model_copy(
        update={
            "expected_return_score": 0.08,
            "composite_rank": max(policy_score.composite_rank, 0.85),
        }
    )
    store = LocalRegistryStore(tmp_path / "registry")
    reward_hash = hash_payload(trajectory_bundle.reward_spec)
    training_hash = hash_payload(training_config)
    store.register_candidate(
        policy_artifact,
        trajectory_bundle,
        reward_config_hash=reward_hash,
        training_config_hash=training_hash,
    )
    store.append_score(policy_artifact.policy_id, champion_score, champion_report)

    champion_artifact_path = tmp_path / "champion-inference-artifact.json"
    champion_artifact_path.write_text("{}", encoding="utf-8")
    champion_paper_sim = _record_paper_sim_evidence(
        store,
        policy_artifact.policy_id,
        tmp_path / "champion-paper-sim.md",
    )
    champion_decision = store.promote_candidate(
        policy_artifact.policy_id,
        evidence=_promotion_evidence(
            policy_artifact,
            deployment_artifact_path=str(champion_artifact_path),
            paper_sim_evidence_id=champion_paper_sim.evidence_id,
        ),
    )
    assert champion_decision.decision == "promote"

    challenger_artifact = policy_artifact.model_copy(
        update={
            "policy_id": f"{policy_artifact.policy_id}-challenger",
            "artifact_id": f"{policy_artifact.artifact_id}-challenger",
            "training_run_id": f"{policy_artifact.training_run_id}-challenger",
        },
        deep=True,
    )
    challenger_report = evaluation_report.model_copy(
        update={
            "policy_id": challenger_artifact.policy_id,
            "evaluation_id": f"{evaluation_report.evaluation_id}-challenger",
            "total_net_return": 0.9,
            "average_net_return": 0.18,
        }
    )
    challenger_score = policy_score.model_copy(
        update={
            "policy_id": challenger_artifact.policy_id,
            "evaluation_id": challenger_report.evaluation_id,
            "expected_return_score": 0.18,
            "composite_rank": max(policy_score.composite_rank, 0.95),
        }
    )
    store.register_candidate(
        challenger_artifact,
        trajectory_bundle,
        reward_config_hash=reward_hash,
        training_config_hash=training_hash,
    )
    store.append_score(challenger_artifact.policy_id, challenger_score, challenger_report)

    challenger_artifact_path = tmp_path / "challenger-inference-artifact.json"
    challenger_artifact_path.write_text("{}", encoding="utf-8")
    challenger_paper_sim = _record_paper_sim_evidence(
        store,
        challenger_artifact.policy_id,
        tmp_path / "challenger-paper-sim.md",
    )

    rejected = store.promote_candidate(
        challenger_artifact.policy_id,
        evidence=_promotion_evidence(
            challenger_artifact,
            deployment_artifact_path=str(challenger_artifact_path),
            paper_sim_evidence_id=challenger_paper_sim.evidence_id,
        ),
    )
    assert rejected.decision == "reject"
    assert "comparison.report_attached" in rejected.failure_reasons

    linked_challenger_paper_sim = store.record_paper_sim_evidence(
        challenger_artifact.policy_id,
        challenger_paper_sim.report_path,
        comparison_report_id="comparison-001",
    )
    approved = store.promote_candidate(
        challenger_artifact.policy_id,
        evidence=_promotion_evidence(
            challenger_artifact,
            deployment_artifact_path=str(challenger_artifact_path),
            paper_sim_evidence_id=linked_challenger_paper_sim.evidence_id,
            comparison_report_id="comparison-001",
        ),
    )
    index = store.load_index()
    previous_champion = store.get_record(policy_artifact.policy_id)
    new_champion = store.get_record(challenger_artifact.policy_id)

    assert approved.decision == "promote"
    assert index.champion_policy_id == challenger_artifact.policy_id
    assert previous_champion is not None and previous_champion.status == "challenger"
    assert new_champion is not None and new_champion.status == "champion"


def _promotion_evidence(
    policy_artifact: PolicyArtifact,
    *,
    deployment_artifact_path: str,
    paper_sim_evidence_id: str,
    comparison_report_id: str | None = None,
) -> PromotionEvidence:
    return PromotionEvidence(
        preprocessing_fit_on_train_only=True,
        no_future_features=True,
        no_future_masks=True,
        no_future_reward_construction=True,
        no_cross_split_contamination=True,
        final_untouched_test_unused_for_selection=True,
        realistic_execution_assumptions=True,
        superiority_not_one_lucky_slice_only=True,
        comparison_report_id=comparison_report_id,
        paper_sim_evidence_id=paper_sim_evidence_id,
        deployment_artifact_path=deployment_artifact_path,
        runtime_uses_inference_artifact_only=True,
        no_live_learning=True,
        executor_boundary_respected=True,
        selector_boundary_respected=True,
        reproducibility=ReproducibilityMetadata(
            data_snapshot_id=policy_artifact.training_snapshot_id,
            code_commit_hash=policy_artifact.code_commit_hash,
            config_hash=policy_artifact.training_config_hash,
            seed=7,
            runtime_stack={"python": "3.12", "framework": "pytorch"},
            reproducible_within_tolerance=True,
        ),
    )


def _record_paper_sim_evidence(
    store: LocalRegistryStore,
    policy_id: str,
    report_path: Path,
) -> PaperSimEvidenceRecord:
    report_path.write_text("# paper sim\n", encoding="utf-8")
    return store.record_paper_sim_evidence(policy_id, report_path)


def _legacy_linear_artifact(policy_artifact: PolicyArtifact) -> PolicyArtifact:
    legacy_tags = [
        tag
        for tag in policy_artifact.runtime_metadata.artifact_compatibility_tags
        if not tag.startswith(
            (
                "runtime_contract:",
                "policy_kind:",
                "derived_contract:",
                "derived_signature:",
                "feature_dim:",
                "compat_mode:",
            )
        )
    ]
    legacy_metadata = policy_artifact.runtime_metadata.model_copy(
        update={
            "strict_runtime_contract": None,
            "artifact_compatibility_tags": legacy_tags,
        },
        deep=True,
    )
    return policy_artifact.model_copy(
        update={
            "schema_version": LEGACY_POLICY_ARTIFACT_SCHEMA_VERSION,
            "artifact_version": LEGACY_POLICY_ARTIFACT_SCHEMA_VERSION,
            "runtime_metadata": legacy_metadata,
        },
        deep=True,
    )


def _tag_map(record) -> dict[str, str]:
    tag_map: dict[str, str] = {}
    for tag in record.artifact_compatibility_tags:
        if ":" not in tag:
            continue
        key, value = tag.split(":", 1)
        tag_map[key] = value
    return tag_map
