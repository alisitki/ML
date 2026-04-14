from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from quantlab_ml.cli.app import app
from quantlab_ml.common import load_model
from quantlab_ml.contracts import InferenceArtifactExport, PolicyArtifact, PromotionEvidence, ReproducibilityMetadata
from quantlab_ml.evaluation import EvaluationEngine
from quantlab_ml.registry import LocalRegistryStore


def test_cli_smoke(repo_root: Path, fixture_path: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    trajectories = tmp_path / "outputs" / "trajectories.json"
    policy = tmp_path / "outputs" / "policy.json"
    evaluation = tmp_path / "outputs" / "evaluation.json"
    score = tmp_path / "outputs" / "score.json"
    exported = tmp_path / "outputs" / "inference_artifact.json"
    registry_root = tmp_path / "registry"

    args_common = [
        "--data-config",
        str(repo_root / "configs" / "data" / "fixture.yaml"),
        "--training-config",
        str(repo_root / "configs" / "training" / "default.yaml"),
        "--reward-config",
        str(repo_root / "configs" / "reward" / "default.yaml"),
    ]
    result = runner.invoke(
        app,
        [
            "build-trajectories",
            "--input",
            str(fixture_path),
            "--output",
            str(trajectories),
            *args_common,
        ],
    )
    assert result.exit_code == 0, result.stdout

    result = runner.invoke(
        app,
        [
            "train",
            "--trajectories",
            str(trajectories),
            "--output",
            str(policy),
            "--training-config",
            str(repo_root / "configs" / "training" / "default.yaml"),
            "--registry-root",
            str(registry_root),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert not (tmp_path / "outputs" / "policy_search.json").exists()
    assert not (tmp_path / "outputs" / "policy_candidates").exists()

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--trajectories",
            str(trajectories),
            "--policy",
            str(policy),
            "--output",
            str(evaluation),
            "--evaluation-config",
            str(repo_root / "configs" / "evaluation" / "default.yaml"),
        ],
    )
    assert result.exit_code == 0, result.stdout

    result = runner.invoke(
        app,
        [
            "score",
            "--policy",
            str(policy),
            "--evaluation",
            str(evaluation),
            "--output",
            str(score),
            "--registry-root",
            str(registry_root),
        ],
    )
    assert result.exit_code == 0, result.stdout

    result = runner.invoke(
        app,
        [
            "export-policy",
            "--policy",
            str(policy),
            "--score",
            str(score),
            "--output",
            str(exported),
        ],
    )
    assert result.exit_code == 0, result.stdout

    exported_policy = load_model(exported, InferenceArtifactExport)
    registry = LocalRegistryStore(registry_root)
    paper_sim_report = tmp_path / "outputs" / "paper-sim.md"
    paper_sim_report.write_text("# paper sim\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "record-paper-sim",
            "--registry-root",
            str(registry_root),
            "--policy-id",
            exported_policy.policy_id,
            "--report",
            str(paper_sim_report),
        ],
    )
    assert result.exit_code == 0, result.stdout
    paper_sim_evidence = registry.get_record(exported_policy.policy_id)
    assert paper_sim_evidence is not None
    assert paper_sim_evidence.paper_sim_evidence_id is not None

    decision = registry.promote_candidate(
        exported_policy.policy_id,
        evidence=PromotionEvidence(
            preprocessing_fit_on_train_only=True,
            no_future_features=True,
            no_future_masks=True,
            no_future_reward_construction=True,
            no_cross_split_contamination=True,
            final_untouched_test_unused_for_selection=True,
            realistic_execution_assumptions=True,
            superiority_not_one_lucky_slice_only=True,
            comparison_report_id=None,
            paper_sim_evidence_id=paper_sim_evidence.paper_sim_evidence_id,
            deployment_artifact_path=str(exported),
            runtime_uses_inference_artifact_only=True,
            no_live_learning=True,
            executor_boundary_respected=True,
            selector_boundary_respected=True,
            reproducibility=ReproducibilityMetadata(
                data_snapshot_id=f"{exported_policy.runtime_metadata.target_asset}:{exported_policy.artifact_id}",
                code_commit_hash="test-commit",
                config_hash="test-config",
                seed=7,
                runtime_stack={"python": "3.12", "framework": "pytorch"},
                reproducible_within_tolerance=True,
            ),
        ),
    )
    assert exported_policy.score_summary["composite_rank"] != 0.0
    assert exported_policy.runtime_metadata.allowed_venues
    assert decision.decision in {"promote", "reject"}
    assert decision.paper_sim_evidence_id == paper_sim_evidence.paper_sim_evidence_id
    if decision.decision == "promote":
        assert registry.load_index().champion_policy_id == exported_policy.policy_id
    else:
        assert decision.failure_reasons


def test_cli_train_search_writes_manifest_and_registers_all_candidates(
    repo_root: Path,
    fixture_path: Path,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    trajectories = tmp_path / "outputs" / "trajectories.json"
    selected_policy = tmp_path / "outputs" / "search-policy.json"
    evaluation = tmp_path / "outputs" / "search-evaluation.json"
    score = tmp_path / "outputs" / "search-score.json"
    exported = tmp_path / "outputs" / "search-inference-artifact.json"
    manifest_path = tmp_path / "outputs" / "search-policy_search.json"
    candidate_dir = tmp_path / "outputs" / "search-policy_candidates"
    registry_root = tmp_path / "registry"

    result = runner.invoke(
        app,
        [
            "build-trajectories",
            "--input",
            str(fixture_path),
            "--output",
            str(trajectories),
            "--data-config",
            str(repo_root / "configs" / "data" / "fixture.yaml"),
            "--training-config",
            str(repo_root / "configs" / "training" / "search-small.yaml"),
            "--reward-config",
            str(repo_root / "configs" / "reward" / "default.yaml"),
        ],
    )
    assert result.exit_code == 0, result.stdout

    result = runner.invoke(
        app,
        [
            "train",
            "--trajectories",
            str(trajectories),
            "--output",
            str(selected_policy),
            "--training-config",
            str(repo_root / "configs" / "training" / "search-small.yaml"),
            "--registry-root",
            str(registry_root),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert selected_policy.exists()
    assert manifest_path.exists()
    assert candidate_dir.exists()

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--trajectories",
            str(trajectories),
            "--policy",
            str(selected_policy),
            "--output",
            str(evaluation),
            "--evaluation-config",
            str(repo_root / "configs" / "evaluation" / "default.yaml"),
        ],
    )
    assert result.exit_code == 0, result.stdout

    result = runner.invoke(
        app,
        [
            "score",
            "--policy",
            str(selected_policy),
            "--evaluation",
            str(evaluation),
            "--output",
            str(score),
        ],
    )
    assert result.exit_code == 0, result.stdout

    result = runner.invoke(
        app,
        [
            "export-policy",
            "--policy",
            str(selected_policy),
            "--score",
            str(score),
            "--output",
            str(exported),
        ],
    )
    assert result.exit_code == 0, result.stdout

    selected = load_model(selected_policy, PolicyArtifact)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exported_policy = load_model(exported, InferenceArtifactExport)
    registry = LocalRegistryStore(registry_root)

    assert manifest["selected_policy_id"] == selected.policy_id
    assert manifest["selected_artifact_path"] == str(selected_policy)
    assert manifest["search_budget_summary"]["total_candidate_count"] == 4
    assert len(manifest["candidates"]) == 4

    selected_candidates = [candidate for candidate in manifest["candidates"] if candidate["selected_candidate"]]
    assert len(selected_candidates) == 1
    assert selected_candidates[0]["artifact_path"] == str(selected_policy)
    assert len(list(candidate_dir.glob("*.json"))) == 3
    assert exported_policy.policy_id == selected.policy_id

    records = registry.list_records()
    assert len(records) == 4
    selected_records = [record for record in records if _tag_map(record.artifact_compatibility_tags)["search_selected"] == "true"]
    assert len(selected_records) == 1
    assert selected_records[0].policy_id == manifest["selected_policy_id"]


def test_cli_audit_continuity_reports_core_backend_status(
    repo_root: Path,
    fixture_path: Path,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    trajectories = tmp_path / "outputs" / "trajectories.json"
    policy = tmp_path / "outputs" / "policy.json"
    registry_root = tmp_path / "registry"
    audit_path = tmp_path / "outputs" / "continuity-audit.json"

    result = runner.invoke(
        app,
        [
            "build-trajectories",
            "--input",
            str(fixture_path),
            "--output",
            str(trajectories),
            "--data-config",
            str(repo_root / "configs" / "data" / "fixture.yaml"),
            "--training-config",
            str(repo_root / "configs" / "training" / "default.yaml"),
            "--reward-config",
            str(repo_root / "configs" / "reward" / "default.yaml"),
        ],
    )
    assert result.exit_code == 0, result.stdout

    result = runner.invoke(
        app,
        [
            "train",
            "--trajectories",
            str(trajectories),
            "--output",
            str(policy),
            "--training-config",
            str(repo_root / "configs" / "training" / "default.yaml"),
            "--registry-root",
            str(registry_root),
        ],
    )
    assert result.exit_code == 0, result.stdout

    result = runner.invoke(
        app,
        [
            "audit-continuity",
            "--registry-root",
            str(registry_root),
            "--output",
            str(audit_path),
        ],
    )
    assert result.exit_code == 0, result.stdout

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["record_count"] == 1
    assert audit["active_training_backend_counts"] == {"pytorch": 1}
    assert audit["active_numpy_training_backend_count"] == 0
    assert audit["ready_to_close_numpy_continuity_window"] is True


def test_cli_evaluate_directory_uses_streaming_api(
    repo_root: Path,
    fixture_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = CliRunner()
    trajectories = tmp_path / "outputs" / "trajectories"
    policy = tmp_path / "outputs" / "policy.json"
    evaluation = tmp_path / "outputs" / "evaluation.json"

    result = runner.invoke(
        app,
        [
            "build-trajectories",
            "--input",
            str(fixture_path),
            "--output",
            str(trajectories),
            "--data-config",
            str(repo_root / "configs" / "data" / "fixture.yaml"),
            "--training-config",
            str(repo_root / "configs" / "training" / "default.yaml"),
            "--reward-config",
            str(repo_root / "configs" / "reward" / "default.yaml"),
        ],
    )
    assert result.exit_code == 0, result.stdout

    result = runner.invoke(
        app,
        [
            "train",
            "--trajectories",
            str(trajectories),
            "--output",
            str(policy),
            "--training-config",
            str(repo_root / "configs" / "training" / "default.yaml"),
        ],
    )
    assert result.exit_code == 0, result.stdout

    called = {"count": 0}
    original = EvaluationEngine.evaluate_records

    def wrapped(self, dataset_spec, reward_spec, trajectories_iter, artifact):
        called["count"] += 1
        return original(self, dataset_spec, reward_spec, trajectories_iter, artifact)

    monkeypatch.setattr(EvaluationEngine, "evaluate_records", wrapped)

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--trajectories",
            str(trajectories),
            "--policy",
            str(policy),
            "--output",
            str(evaluation),
            "--evaluation-config",
            str(repo_root / "configs" / "evaluation" / "default.yaml"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert called["count"] == 1


def _tag_map(tags: list[str]) -> dict[str, str]:
    tag_map: dict[str, str] = {}
    for tag in tags:
        if ":" not in tag:
            continue
        key, value = tag.split(":", 1)
        tag_map[key] = value
    return tag_map
