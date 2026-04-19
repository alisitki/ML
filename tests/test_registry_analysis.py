from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

from typer.testing import CliRunner

from quantlab_ml.cli.app import app
from quantlab_ml.common import dump_model, hash_payload, load_yaml
from quantlab_ml.contracts import (
    DatasetSpec,
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
    _add_incomplete_registry_scaffold(source_root)

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
    assert "source_root_had_incomplete_registry_scaffold" in classification["classification_reasons"]
    assert imported_record is not None
    assert imported_record.evaluation_surface_id == artifact.evaluation_surface_id
    assert imported_record.score_history[-1].evaluation_id == report.evaluation_id
    assert json.loads((output_root / "import_classification.json").read_text(encoding="utf-8"))["analysis_only"] is True


def test_import_diagnostic_retained_run_recovers_coverage_from_root_tensor_cache_manifest(
    tmp_path: Path,
    fixture_path: Path,
    dataset_spec,
    training_bundle: tuple,
    reward_spec,
    evaluation_boundary,
) -> None:
    source_root, artifact, report, score, _ = _build_retained_run_root(
        tmp_path=tmp_path,
        fixture_path=fixture_path,
        dataset_spec=dataset_spec,
        training_bundle=training_bundle,
        reward_spec=reward_spec,
        evaluation_boundary=evaluation_boundary,
    )
    manifest_payload = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    manifest_payload["split_write_stats"] = {}
    (source_root / "manifest.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    (source_root / "tensor_cache_manifest.json").write_text(
        (source_root / "tensor_cache_v1" / "tensor_cache_manifest.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    shutil.rmtree(source_root / "tensor_cache_v1")
    _add_incomplete_registry_scaffold(source_root)

    output_root = tmp_path / "legacy-diagnostic-import"
    classification = import_diagnostic_retained_run(
        source_root=source_root,
        output_root=output_root,
    )
    imported_store = LocalRegistryStore(output_root / "registry")
    imported_record = imported_store.get_record(artifact.policy_id)

    assert classification["analysis_only"] is True
    assert imported_record is not None
    assert imported_record.coverage.train_sample_count > 0


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


def test_ql031_distinct_rerun_config_preserves_market_scope_and_shifts_surface(repo_root: Path) -> None:
    baseline = DatasetSpec.model_validate(load_yaml(repo_root / "configs" / "data" / "controlled-remote-day.yaml")["dataset"])
    fallback = DatasetSpec.model_validate(
        load_yaml(repo_root / "configs" / "data" / "ql031-controlled-remote-day-20260126.yaml")["dataset"]
    )

    baseline_payload = baseline.model_dump(mode="json")
    fallback_payload = fallback.model_dump(mode="json")
    for key in ("slice_id", "train_range", "validation_range", "final_untouched_test_range"):
        baseline_payload.pop(key)
        fallback_payload.pop(key)

    assert fallback_payload == baseline_payload
    assert fallback.slice_id == "controlled-remote-example-20260126"
    assert fallback.train_range.start == baseline.train_range.start + timedelta(days=1)
    assert fallback.train_range.end == baseline.train_range.end + timedelta(days=1)
    assert fallback.validation_range.start == baseline.validation_range.start + timedelta(days=1)
    assert fallback.validation_range.end == baseline.validation_range.end + timedelta(days=1)
    assert fallback.final_untouched_test_range.start == baseline.final_untouched_test_range.start + timedelta(days=1)
    assert fallback.final_untouched_test_range.end == baseline.final_untouched_test_range.end + timedelta(days=1)


def test_ql031_distinct_rerun_config_preflight_passes_controlled_remote_collision_shape(
    repo_root: Path,
    tmp_path: Path,
    trajectory_bundle,
    policy_artifact: PolicyArtifact,
    evaluation_report,
    policy_score: PolicyScore,
    training_bundle: tuple,
    reward_spec,
) -> None:
    _, _, training_config = training_bundle
    registry_root = tmp_path / "registry"
    controlled_remote_train_range = TimeRange(
        start="2026-01-25T00:00:00Z",
        end="2026-01-25T15:59:00Z",
    )
    controlled_remote_bundle = trajectory_bundle.model_copy(
        update={
            "dataset_spec": trajectory_bundle.dataset_spec.model_copy(
                update={
                    "dataset_hash": "s3-controlled-remote-v1",
                    "slice_id": "controlled-remote-example-20260125",
                    "train_range": controlled_remote_train_range,
                    "validation_range": TimeRange(
                        start="2026-01-25T16:00:00Z",
                        end="2026-01-25T19:59:00Z",
                    ),
                    "final_untouched_test_range": TimeRange(
                        start="2026-01-25T20:00:00Z",
                        end="2026-01-25T23:59:00Z",
                    ),
                }
            ),
        },
        deep=True,
    )
    controlled_remote_artifact = policy_artifact.model_copy(
        update={
            "policy_id": f"{policy_artifact.policy_id}-controlled-remote",
            "artifact_id": f"{policy_artifact.artifact_id}-controlled-remote",
            "training_run_id": f"{policy_artifact.training_run_id}-controlled-remote",
            "training_snapshot_id": "s3-controlled-remote-v1:controlled-remote-example-20260125",
            "evaluation_surface_id": "controlled-remote-example-20260125:split_v1_walkforward:reward_v1",
        },
        deep=True,
    )
    controlled_remote_report = evaluation_report.model_copy(
        update={
            "policy_id": controlled_remote_artifact.policy_id,
            "evaluation_id": f"{evaluation_report.evaluation_id}-controlled-remote",
        }
    )
    controlled_remote_score = policy_score.model_copy(
        update={
            "policy_id": controlled_remote_artifact.policy_id,
            "evaluation_id": controlled_remote_report.evaluation_id,
        }
    )
    store = LocalRegistryStore(registry_root)
    store.register_candidate(
        controlled_remote_artifact,
        controlled_remote_bundle,
        reward_config_hash=hash_payload(trajectory_bundle.reward_spec),
        training_config_hash=hash_payload(training_config),
    )
    store.append_score(controlled_remote_artifact.policy_id, controlled_remote_score, controlled_remote_report)

    fallback_dataset = DatasetSpec.model_validate(
        load_yaml(repo_root / "configs" / "data" / "ql031-controlled-remote-day-20260126.yaml")["dataset"]
    )
    preflight = preflight_distinct_surface(
        dataset_spec=fallback_dataset,
        reward_spec=reward_spec,
        registry_roots=[registry_root],
    )

    assert preflight["allowed"] is True
    assert preflight["collisions"] == []
    assert preflight["candidate"]["evaluation_surface_id"] == "controlled-remote-example-20260126:split_v1_walkforward:reward_v1"
    assert preflight["candidate"]["slice_id"] == "controlled-remote-example-20260126"
    assert preflight["candidate"]["train_window"] == "2026-01-26T00:00:00+00:00 -> 2026-01-26T15:59:00+00:00"


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
    _add_incomplete_registry_scaffold(source_root)
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
    incomplete_scaffold_candidate = next(
        candidate
        for candidate in discovery["candidates"]
        if candidate["run_root"] == str(source_root.resolve())
    )
    assert incomplete_scaffold_candidate["candidate_classification"] == "diagnostic_import_only_bundle"
    assert incomplete_scaffold_candidate["has_registry_state"] is False
    assert incomplete_scaffold_candidate["has_incomplete_registry_scaffold"] is True
    assert "registry_scaffold_incomplete_or_empty" in incomplete_scaffold_candidate["classification_reasons"]


def test_ql031_batch_script_uses_distinct_default_preflight_surface(
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
    registry_root_a = tmp_path / "retained-a" / "registry"
    _register_retained_run(
        registry_root=registry_root_a,
        artifact=artifact,
        manifest=manifest,
        report=report,
        score=score,
    )

    artifact_b = artifact.model_copy(
        update={
            "policy_id": f"{artifact.policy_id}-b",
            "artifact_id": f"{artifact.artifact_id}-b",
            "training_run_id": f"{artifact.training_run_id}-b",
        },
        deep=True,
    )
    report_b = report.model_copy(
        update={
            "policy_id": artifact_b.policy_id,
            "evaluation_id": f"{report.evaluation_id}-b",
            "total_net_return": report.total_net_return - 0.1,
        }
    )
    score_b = score.model_copy(
        update={
            "policy_id": artifact_b.policy_id,
            "evaluation_id": report_b.evaluation_id,
            "composite_rank": score.composite_rank - 0.001,
        }
    )
    registry_root_b = tmp_path / "retained-b" / "registry"
    _register_retained_run(
        registry_root=registry_root_b,
        artifact=artifact_b,
        manifest=manifest,
        report=report_b,
        score=score_b,
    )

    diagnostic_bundle_root = tmp_path / "controlled-rerun-bundle"
    diagnostic_bundle_root.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "policy.json", "evaluation.json", "score.json"):
        (diagnostic_bundle_root / name).write_text(
            (source_root / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    _add_incomplete_registry_scaffold(diagnostic_bundle_root)

    output_root = tmp_path / "analysis" / "ql031"
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_ql031_batch.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--workspace-registry-root",
            str(registry_root_a),
            "--workspace-authority-status",
            "unconfirmed",
            "--workspace-registry-root",
            str(registry_root_b),
            "--workspace-authority-status",
            "unconfirmed",
            "--diagnostic-bundle-root",
            str(diagnostic_bundle_root),
            "--external-search-root",
            str(tmp_path),
            "--output-root",
            str(output_root),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout

    workspace_inventory = json.loads((output_root / "workspace_blocker_inventory.json").read_text(encoding="utf-8"))
    workspace_plus_inventory = json.loads(
        (output_root / "workspace_plus_diagnostic_blocker_inventory.json").read_text(encoding="utf-8")
    )
    summary = json.loads((output_root / "ql031_status.json").read_text(encoding="utf-8"))

    assert workspace_inventory["source_count"] == 2
    assert workspace_plus_inventory["source_count"] == 3
    assert len(workspace_plus_inventory["grouped_by_evaluation_surface"]) == 1
    assert len(workspace_plus_inventory["grouped_by_slice"]) == 1
    assert len(workspace_plus_inventory["grouped_by_train_window"]) == 1
    assert summary["status"] == "preflight_passed_no_distinct_retained_surface"
    assert summary["preflight_selection_result"] == "preflight_passed_no_distinct_retained_surface"
    assert summary["batch_execution_result"] == "preflight_only"
    assert summary["failure_stage"] is None
    assert summary["failure_reason"] is None
    assert summary["comparison_reports"] == []
    assert summary["distinct_surface_preflight_allowed"] is True
    assert (output_root / "distinct_surface_preflight.json").exists()
    assert (output_root / "retained_root_discovery.json").exists()
    assert (output_root / "diagnostic_imports" / diagnostic_bundle_root.name / "import_classification.json").exists()


def test_ql031_batch_legacy_status_prefers_execution_failure_over_preflight_result(repo_root: Path) -> None:
    script_path = repo_root / "scripts" / "run_ql031_batch.py"
    spec = importlib.util.spec_from_file_location("run_ql031_batch", script_path)

    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    status = module._legacy_status(
        preflight_selection_result="blocked_distinct_surface_collision",
        batch_execution_result="failed_evidence_regeneration",
    )

    assert status == "failed_evidence_regeneration"


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


def _add_incomplete_registry_scaffold(run_root: Path) -> None:
    for directory in ("records", "scores", "evaluations", "artifacts", "paper_sim", "promotions"):
        (run_root / "registry" / directory).mkdir(parents=True, exist_ok=True)


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
