from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from typer.testing import CliRunner

from quantlab_ml.cli.app import app
from quantlab_ml.common import dump_model, hash_payload
from quantlab_ml.contracts import (
    PolicyArtifact,
    PolicyScore,
    PromotionEvidence,
    ReproducibilityMetadata,
    TimeRange,
)
from quantlab_ml.data import LocalFixtureSource
from quantlab_ml.evaluation import EvaluationEngine
from quantlab_ml.registry import LocalRegistryStore
from quantlab_ml.registry.analysis import (
    build_blocker_inventory,
    candidate_surface_identity,
    discover_retained_roots,
    import_diagnostic_retained_run,
    preflight_distinct_surface,
    preflight_same_root_comparison,
)
from quantlab_ml.scoring import PolicyScorer
from quantlab_ml.training import LinearPolicyTrainer
from quantlab_ml.trajectories import TrajectoryBuilder, TrajectoryDirectoryStore


def test_import_diagnostic_retained_run_marks_analysis_only(
    tmp_path: Path,
    fixture_path: Path,
    dataset_spec,
    training_bundle: tuple,
    reward_spec,
    evaluation_boundary,
) -> None:
    source_root, artifact, report, score, manifest = _build_retained_run_root(
        tmp_path=tmp_path,
        fixture_path=fixture_path,
        dataset_spec=dataset_spec,
        training_bundle=training_bundle,
        reward_spec=reward_spec,
        evaluation_boundary=evaluation_boundary,
    )

    output_root = tmp_path / "diagnostic-import"
    classification = import_diagnostic_retained_run(
        source_root=source_root,
        output_root=output_root,
    )
    imported_store = LocalRegistryStore(output_root / "registry")
    imported_record = imported_store.get_record(artifact.policy_id)

    assert classification["analysis_only"] is True
    assert classification["promotion_eligible"] is False
    assert classification["comparison_eligible"] is False
    assert classification["inspected_evidence_kind"] == "external_retained_evidence"
    assert classification["authority_status"] == "unconfirmed"
    assert imported_record is not None
    assert imported_record.evaluation_surface_id == artifact.evaluation_surface_id
    assert imported_record.score_history[-1].evaluation_id == report.evaluation_id
    assert json.loads((output_root / "import_classification.json").read_text(encoding="utf-8"))["analysis_only"] is True


def test_blocker_inventory_carries_evidence_class_and_analysis_flags(
    tmp_path: Path,
    fixture_path: Path,
    dataset_spec,
    training_bundle: tuple,
    reward_spec,
    evaluation_boundary,
) -> None:
    source_root, artifact, report, score, manifest = _build_retained_run_root(
        tmp_path=tmp_path,
        fixture_path=fixture_path,
        dataset_spec=dataset_spec,
        training_bundle=training_bundle,
        reward_spec=reward_spec,
        evaluation_boundary=evaluation_boundary,
    )
    registry_root = tmp_path / "retained-registry"
    _register_retained_run(
        registry_root=registry_root,
        artifact=artifact,
        manifest=manifest,
        report=report,
        score=score,
    )
    diagnostic_root = tmp_path / "diagnostic-import"
    import_diagnostic_retained_run(source_root=source_root, output_root=diagnostic_root)

    inventory = build_blocker_inventory(
        registry_roots=[registry_root, diagnostic_root / "registry"],
        inspected_evidence_kinds=["external-retained-evidence", "external-retained-evidence"],
        authority_statuses=[None, None],
    )
    direct_source = next(source for source in inventory["sources"] if source["analysis_only"] is False)
    diagnostic_source = next(source for source in inventory["sources"] if source["analysis_only"] is True)

    assert inventory["source_count"] == 2
    assert direct_source["promotion_eligible"] is True
    assert direct_source["comparison_eligible"] is False
    assert direct_source["inspected_evidence_kind"] == "external_retained_evidence"
    assert diagnostic_source["promotion_eligible"] is False
    assert diagnostic_source["comparison_eligible"] is False
    assert any(
        "analysis-only" in limitation.lower()
        for limitation in diagnostic_source["limitations"]
    )


def test_preflight_same_root_comparison_requires_and_finds_same_root_champion(
    tmp_path: Path,
    trajectory_bundle,
    policy_artifact: PolicyArtifact,
    evaluation_report,
    policy_score: PolicyScore,
    training_bundle: tuple,
) -> None:
    _, _, training_config = training_bundle
    store = LocalRegistryStore(tmp_path / "registry")
    reward_hash = hash_payload(trajectory_bundle.reward_spec)
    training_hash = hash_payload(training_config)

    champion_artifact = policy_artifact
    champion_report = evaluation_report.model_copy(
        update={
            "total_net_return": 0.45,
            "average_net_return": 0.09,
        }
    )
    champion_score = policy_score.model_copy(
        update={
            "expected_return_score": 0.09,
            "composite_rank": max(policy_score.composite_rank, 0.91),
        }
    )
    store.register_candidate(
        champion_artifact,
        trajectory_bundle,
        reward_config_hash=reward_hash,
        training_config_hash=training_hash,
    )
    store.append_score(champion_artifact.policy_id, champion_score, champion_report)
    paper_sim = tmp_path / "champion-paper-sim.md"
    paper_sim.write_text("# paper sim\n", encoding="utf-8")
    evidence = store.record_paper_sim_evidence(champion_artifact.policy_id, paper_sim)
    store.promote_candidate(
        champion_artifact.policy_id,
        evidence=_promotion_evidence(
            champion_artifact,
            deployment_artifact_path=str(store.artifacts_dir / f"{champion_artifact.policy_id}.json"),
            paper_sim_evidence_id=evidence.evidence_id,
        ),
    )

    challenger_artifact = champion_artifact.model_copy(
        update={
            "policy_id": f"{champion_artifact.policy_id}-challenger",
            "artifact_id": f"{champion_artifact.artifact_id}-challenger",
            "training_run_id": f"{champion_artifact.training_run_id}-challenger",
        },
        deep=True,
    )
    challenger_report = champion_report.model_copy(
        update={
            "policy_id": challenger_artifact.policy_id,
            "evaluation_id": f"{champion_report.evaluation_id}-challenger",
            "total_net_return": 0.32,
            "average_net_return": 0.064,
        }
    )
    challenger_score = champion_score.model_copy(
        update={
            "policy_id": challenger_artifact.policy_id,
            "evaluation_id": challenger_report.evaluation_id,
            "expected_return_score": 0.064,
            "composite_rank": 0.88,
        }
    )
    store.register_candidate(
        challenger_artifact,
        trajectory_bundle,
        reward_config_hash=reward_hash,
        training_config_hash=training_hash,
    )
    store.append_score(challenger_artifact.policy_id, challenger_score, challenger_report)

    summary = preflight_same_root_comparison(registry_root=tmp_path / "registry")

    assert summary["allowed"] is True
    assert summary["champion_policy_id"] == champion_artifact.policy_id
    assert summary["challenger_policy_id"] == challenger_artifact.policy_id
    assert challenger_artifact.policy_id in summary["eligible_challenger_policy_ids"]
    assert summary["blocking_reasons"] == []


def test_preflight_distinct_surface_fails_closed_on_collisions_and_uses_shared_helper(
    tmp_path: Path,
    trajectory_bundle,
    policy_artifact: PolicyArtifact,
    training_bundle: tuple,
    reward_spec,
    dataset_spec,
) -> None:
    _, _, training_config = training_bundle
    store = LocalRegistryStore(tmp_path / "registry")
    store.register_candidate(
        policy_artifact,
        trajectory_bundle,
        reward_config_hash=hash_payload(trajectory_bundle.reward_spec),
        training_config_hash=hash_payload(training_config),
    )

    collision = preflight_distinct_surface(
        dataset_spec=dataset_spec,
        reward_spec=reward_spec,
        registry_roots=[tmp_path / "registry"],
    )
    assert collision["allowed"] is False
    assert {item["field"] for item in collision["collisions"]} == {
        "evaluation_surface_id",
        "slice_id",
        "train_window",
    }
    assert collision["candidate"] == candidate_surface_identity(
        dataset_spec=dataset_spec,
        reward_spec=reward_spec,
    )

    shifted_dataset_spec = dataset_spec.model_copy(
        update={
            "slice_id": "fixture-slice-shifted",
            "train_range": TimeRange(
                start=dataset_spec.train_range.start + timedelta(days=1),
                end=dataset_spec.train_range.end + timedelta(days=1),
            ),
            "validation_range": TimeRange(
                start=dataset_spec.validation_range.start + timedelta(days=1),
                end=dataset_spec.validation_range.end + timedelta(days=1),
            ),
            "final_untouched_test_range": TimeRange(
                start=dataset_spec.final_untouched_test_range.start + timedelta(days=1),
                end=dataset_spec.final_untouched_test_range.end + timedelta(days=1),
            ),
        }
    )
    distinct = preflight_distinct_surface(
        dataset_spec=shifted_dataset_spec,
        reward_spec=reward_spec,
        registry_roots=[tmp_path / "registry"],
    )
    assert distinct["allowed"] is True
    assert distinct["collisions"] == []


def test_discover_retained_roots_marks_analysis_only_candidates(
    fixture_path: Path,
    tmp_path: Path,
    dataset_spec,
    training_bundle: tuple,
    reward_spec,
    evaluation_boundary,
) -> None:
    source_root, artifact, report, score, manifest = _build_retained_run_root(
        tmp_path=tmp_path,
        fixture_path=fixture_path,
        dataset_spec=dataset_spec,
        training_bundle=training_bundle,
        reward_spec=reward_spec,
        evaluation_boundary=evaluation_boundary,
    )
    diagnostic_root = tmp_path / "analysis-only-import"
    import_diagnostic_retained_run(source_root=source_root, output_root=diagnostic_root)

    registry_root = tmp_path / "retained-registry"
    _register_retained_run(
        registry_root=registry_root,
        artifact=artifact,
        manifest=manifest,
        report=report,
        score=score,
    )

    discovery = discover_retained_roots(search_root=tmp_path)
    assert discovery["candidate_count"] >= 2
    assert any(candidate["has_bundle_artifacts"] for candidate in discovery["candidates"])
    assert any(candidate["analysis_only"] for candidate in discovery["candidates"])
    assert any(candidate["comparison_eligible"] is False for candidate in discovery["candidates"])


def test_cli_compare_policies_fails_closed_without_same_root_champion(
    tmp_path: Path,
    trajectory_bundle,
    policy_artifact: PolicyArtifact,
    evaluation_report,
    policy_score: PolicyScore,
    training_bundle: tuple,
) -> None:
    runner = CliRunner()
    _, _, training_config = training_bundle
    store = LocalRegistryStore(tmp_path / "registry")
    store.register_candidate(
        policy_artifact,
        trajectory_bundle,
        reward_config_hash=hash_payload(trajectory_bundle.reward_spec),
        training_config_hash=hash_payload(training_config),
    )
    store.append_score(policy_artifact.policy_id, policy_score, evaluation_report)

    result = runner.invoke(
        app,
        [
            "compare-policies",
            "--registry-root",
            str(tmp_path / "registry"),
            "--challenger-policy-id",
            policy_artifact.policy_id,
        ],
    )

    assert result.exit_code != 0
    error_text = "\n".join(
        part
        for part in (
            result.stdout,
            getattr(result, "stderr", ""),
            result.output,
            str(result.exception) if result.exception is not None else "",
        )
        if part
    )
    assert "same-root champion" in error_text


def _build_retained_run_root(
    *,
    tmp_path: Path,
    fixture_path: Path,
    dataset_spec,
    training_bundle: tuple,
    reward_spec,
    evaluation_boundary,
) -> tuple[Path, PolicyArtifact, object, PolicyScore, object]:
    trajectory_spec, action_space, training_config = training_bundle
    source_root = tmp_path / "retained-run-source"
    events = LocalFixtureSource(fixture_path).load_events(dataset_spec)
    builder = TrajectoryBuilder(dataset_spec, trajectory_spec, action_space, reward_spec)
    builder.build_to_directory(events, source_root)
    manifest = TrajectoryDirectoryStore.read_manifest(source_root)
    artifact = LinearPolicyTrainer(training_config).train_search_from_directory(
        manifest,
        source_root,
    ).selected_artifact
    report = EvaluationEngine(evaluation_boundary).evaluate_directory(
        manifest=manifest,
        directory=source_root,
        artifact=artifact,
        split_name="final_untouched_test",
    )
    score = PolicyScorer().score(report)
    dump_model(source_root / "policy.json", artifact)
    dump_model(source_root / "evaluation.json", report)
    dump_model(source_root / "score.json", score)
    return source_root, artifact, report, score, manifest


def _register_retained_run(
    *,
    registry_root: Path,
    artifact: PolicyArtifact,
    manifest,
    report,
    score: PolicyScore,
) -> None:
    store = LocalRegistryStore(registry_root)
    store.register_candidate_from_manifest(
        artifact,
        manifest,
        reward_config_hash=hash_payload(manifest.reward_spec),
        training_config_hash=artifact.training_config_hash,
    )
    store.append_score(artifact.policy_id, score, report)


def _promotion_evidence(
    policy_artifact: PolicyArtifact,
    *,
    deployment_artifact_path: str,
    paper_sim_evidence_id: str,
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
        comparison_report_id=None,
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
