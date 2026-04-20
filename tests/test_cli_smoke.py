from __future__ import annotations

import json
from pathlib import Path
import shutil

from typer.testing import CliRunner
import yaml

from quantlab_ml.cli.app import app
from quantlab_ml.common import dump_model, hash_payload, load_model
from quantlab_ml.contracts import (
    EvaluationReport,
    InferenceArtifactExport,
    PolicyArtifact,
    PromotionEvidence,
    ReproducibilityMetadata,
)
from quantlab_ml.evaluation import EvaluationEngine
from quantlab_ml.policies import PolicyRuntimeBridge
from quantlab_ml.registry import LocalRegistryStore
from quantlab_ml.trajectories import TrajectoryDirectoryStore
from quantlab_ml.trajectories.tensor_cache import tensor_cache_directory


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


def test_cli_promote_policy_promotes_scored_candidate_from_yaml_evidence(
    tmp_path: Path,
    trajectory_bundle,
    policy_artifact: PolicyArtifact,
    evaluation_report,
    policy_score,
    training_bundle: tuple,
) -> None:
    runner = CliRunner()
    _, _, training_config = training_bundle
    reward_hash = hash_payload(trajectory_bundle.reward_spec)
    training_hash = hash_payload(training_config)
    registry_root = tmp_path / "registry"
    registry = LocalRegistryStore(registry_root)
    deployment_artifact = tmp_path / "candidate-inference-artifact.json"
    deployment_artifact.write_text("{}", encoding="utf-8")
    champion_report = evaluation_report.model_copy(
        update={
            "total_net_return": 0.35,
            "average_net_return": 0.07,
        }
    )
    champion_score = policy_score.model_copy(
        update={
            "expected_return_score": 0.07,
            "composite_rank": max(policy_score.composite_rank, 0.86),
        }
    )

    registry.register_candidate(
        policy_artifact,
        trajectory_bundle,
        reward_config_hash=reward_hash,
        training_config_hash=training_hash,
    )
    registry.append_score(policy_artifact.policy_id, champion_score, champion_report)
    paper_sim_report = tmp_path / "candidate-paper-sim.md"
    paper_sim_report.write_text("# paper sim\n", encoding="utf-8")
    paper_sim_evidence = registry.record_paper_sim_evidence(policy_artifact.policy_id, paper_sim_report)

    evidence_path = tmp_path / "promotion-evidence.yaml"
    _write_promotion_evidence(
        evidence_path,
        _promotion_evidence(
            policy_artifact,
            deployment_artifact_path=str(deployment_artifact),
            paper_sim_evidence_id=paper_sim_evidence.evidence_id,
        ),
    )
    decision_path = tmp_path / "promotion-decision.json"

    result = runner.invoke(
        app,
        [
            "promote-policy",
            "--registry-root",
            str(registry_root),
            "--policy-id",
            policy_artifact.policy_id,
            "--evidence",
            str(evidence_path),
            "--output",
            str(decision_path),
        ],
    )
    assert result.exit_code == 0, result.stdout

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    record = registry.get_record(policy_artifact.policy_id)

    assert decision["decision"] == "promote"
    assert decision["failure_reasons"] == []
    assert registry.load_index().champion_policy_id == policy_artifact.policy_id
    assert record is not None
    assert record.status == "champion"


def test_cli_promote_policy_rejects_missing_paper_sim_evidence(
    tmp_path: Path,
    trajectory_bundle,
    policy_artifact: PolicyArtifact,
    evaluation_report,
    policy_score,
    training_bundle: tuple,
) -> None:
    runner = CliRunner()
    _, _, training_config = training_bundle
    reward_hash = hash_payload(trajectory_bundle.reward_spec)
    training_hash = hash_payload(training_config)
    registry_root = tmp_path / "registry"
    registry = LocalRegistryStore(registry_root)
    deployment_artifact = tmp_path / "candidate-inference-artifact.json"
    deployment_artifact.write_text("{}", encoding="utf-8")
    champion_report = evaluation_report.model_copy(
        update={
            "total_net_return": 0.35,
            "average_net_return": 0.07,
        }
    )
    champion_score = policy_score.model_copy(
        update={
            "expected_return_score": 0.07,
            "composite_rank": max(policy_score.composite_rank, 0.86),
        }
    )

    registry.register_candidate(
        policy_artifact,
        trajectory_bundle,
        reward_config_hash=reward_hash,
        training_config_hash=training_hash,
    )
    registry.append_score(policy_artifact.policy_id, champion_score, champion_report)

    evidence_path = tmp_path / "promotion-evidence.json"
    _write_promotion_evidence(
        evidence_path,
        _promotion_evidence(
            policy_artifact,
            deployment_artifact_path=str(deployment_artifact),
            paper_sim_evidence_id="paper-sim-missing",
        ),
    )
    decision_path = tmp_path / "promotion-decision.json"

    result = runner.invoke(
        app,
        [
            "promote-policy",
            "--registry-root",
            str(registry_root),
            "--policy-id",
            policy_artifact.policy_id,
            "--evidence",
            str(evidence_path),
            "--output",
            str(decision_path),
        ],
    )
    assert result.exit_code == 0, result.stdout

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["decision"] == "reject"
    assert "artifacts.paper_sim_report_exists" in decision["failure_reasons"]
    assert "artifacts.paper_sim_linked_to_evaluation" in decision["failure_reasons"]
    assert "artifacts.deployment_artifact_exists" not in decision["failure_reasons"]


def test_cli_promote_policy_rejects_missing_deployment_artifact(
    tmp_path: Path,
    trajectory_bundle,
    policy_artifact: PolicyArtifact,
    evaluation_report,
    policy_score,
    training_bundle: tuple,
) -> None:
    runner = CliRunner()
    _, _, training_config = training_bundle
    reward_hash = hash_payload(trajectory_bundle.reward_spec)
    training_hash = hash_payload(training_config)
    registry_root = tmp_path / "registry"
    registry = LocalRegistryStore(registry_root)
    champion_report = evaluation_report.model_copy(
        update={
            "total_net_return": 0.35,
            "average_net_return": 0.07,
        }
    )
    champion_score = policy_score.model_copy(
        update={
            "expected_return_score": 0.07,
            "composite_rank": max(policy_score.composite_rank, 0.86),
        }
    )

    registry.register_candidate(
        policy_artifact,
        trajectory_bundle,
        reward_config_hash=reward_hash,
        training_config_hash=training_hash,
    )
    registry.append_score(policy_artifact.policy_id, champion_score, champion_report)
    paper_sim_report = tmp_path / "candidate-paper-sim.md"
    paper_sim_report.write_text("# paper sim\n", encoding="utf-8")
    paper_sim_evidence = registry.record_paper_sim_evidence(policy_artifact.policy_id, paper_sim_report)

    evidence_path = tmp_path / "promotion-evidence.json"
    _write_promotion_evidence(
        evidence_path,
        _promotion_evidence(
            policy_artifact,
            deployment_artifact_path=str(tmp_path / "missing-inference-artifact.json"),
            paper_sim_evidence_id=paper_sim_evidence.evidence_id,
        ),
    )
    decision_path = tmp_path / "promotion-decision.json"

    result = runner.invoke(
        app,
        [
            "promote-policy",
            "--registry-root",
            str(registry_root),
            "--policy-id",
            policy_artifact.policy_id,
            "--evidence",
            str(evidence_path),
            "--output",
            str(decision_path),
        ],
    )
    assert result.exit_code == 0, result.stdout

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["decision"] == "reject"
    assert "artifacts.deployment_artifact_exists" in decision["failure_reasons"]
    assert "artifacts.paper_sim_report_exists" not in decision["failure_reasons"]


def test_cli_promote_policy_rejects_challenger_without_comparison_evidence(
    tmp_path: Path,
    trajectory_bundle,
    policy_artifact: PolicyArtifact,
    evaluation_report,
    policy_score,
    training_bundle: tuple,
) -> None:
    runner = CliRunner()
    _, _, training_config = training_bundle
    reward_hash = hash_payload(trajectory_bundle.reward_spec)
    training_hash = hash_payload(training_config)
    registry_root = tmp_path / "registry"
    registry = LocalRegistryStore(registry_root)

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
    registry.register_candidate(
        policy_artifact,
        trajectory_bundle,
        reward_config_hash=reward_hash,
        training_config_hash=training_hash,
    )
    registry.append_score(policy_artifact.policy_id, champion_score, champion_report)
    champion_export = tmp_path / "champion-inference-artifact.json"
    champion_export.write_text("{}", encoding="utf-8")
    champion_paper_sim = tmp_path / "champion-paper-sim.md"
    champion_paper_sim.write_text("# champion paper sim\n", encoding="utf-8")
    champion_evidence = registry.record_paper_sim_evidence(policy_artifact.policy_id, champion_paper_sim)
    registry.promote_candidate(
        policy_artifact.policy_id,
        evidence=_promotion_evidence(
            policy_artifact,
            deployment_artifact_path=str(champion_export),
            paper_sim_evidence_id=champion_evidence.evidence_id,
        ),
    )

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
    registry.register_candidate(
        challenger_artifact,
        trajectory_bundle,
        reward_config_hash=reward_hash,
        training_config_hash=training_hash,
    )
    registry.append_score(challenger_artifact.policy_id, challenger_score, challenger_report)

    challenger_export = tmp_path / "challenger-inference-artifact.json"
    challenger_export.write_text("{}", encoding="utf-8")
    evidence_path = tmp_path / "challenger-promotion-evidence.json"
    _write_promotion_evidence(
        evidence_path,
        _promotion_evidence(
            challenger_artifact,
            deployment_artifact_path=str(challenger_export),
            paper_sim_evidence_id=champion_evidence.evidence_id,
        ),
    )
    decision_path = tmp_path / "challenger-promotion-decision.json"

    result = runner.invoke(
        app,
        [
            "promote-policy",
            "--registry-root",
            str(registry_root),
            "--policy-id",
            challenger_artifact.policy_id,
            "--evidence",
            str(evidence_path),
            "--output",
            str(decision_path),
        ],
    )
    assert result.exit_code == 0, result.stdout

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["decision"] == "reject"
    assert "comparison.report_attached" in decision["failure_reasons"]


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


def test_cli_train_search_supports_same_root_comparison_and_linked_paper_sim_chain(
    repo_root: Path,
    fixture_path: Path,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    trajectories = tmp_path / "outputs" / "trajectories.json"
    selected_policy = tmp_path / "outputs" / "search-policy.json"
    manifest_path = tmp_path / "outputs" / "search-policy_search.json"
    registry_root = tmp_path / "registry"
    candidate_eval_dir = tmp_path / "outputs" / "candidate-evaluations"
    candidate_score_dir = tmp_path / "outputs" / "candidate-scores"
    champion_export = tmp_path / "outputs" / "champion-inference-artifact.json"
    comparison_output = tmp_path / "outputs" / "comparison-report.json"
    evidence_pack_path = tmp_path / "outputs" / "offline-evidence-pack.json"

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
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ranked_candidates = sorted(manifest["candidates"], key=lambda candidate: candidate["candidate_rank"])
    assert len(ranked_candidates) >= 2

    for index, candidate in enumerate(ranked_candidates, start=1):
        artifact_path = Path(candidate["artifact_path"])
        evaluation_path = candidate_eval_dir / f"{artifact_path.stem}-evaluation.json"
        score_path = candidate_score_dir / f"{artifact_path.stem}-score.json"

        result = runner.invoke(
            app,
            [
                "evaluate",
                "--trajectories",
                str(trajectories),
                "--policy",
                str(artifact_path),
                "--output",
                str(evaluation_path),
                "--evaluation-config",
                str(repo_root / "configs" / "evaluation" / "default.yaml"),
            ],
        )
        assert result.exit_code == 0, result.stdout

        evaluation_report = load_model(evaluation_path, EvaluationReport)
        adjusted_report = evaluation_report.model_copy(
            update={
                "total_net_return": 0.4 if candidate["candidate_rank"] == 1 else max(0.05, 0.3 - (index * 0.01)),
                "average_net_return": 0.08 if candidate["candidate_rank"] == 1 else max(0.01, 0.06 - (index * 0.005)),
            }
        )
        if candidate["candidate_rank"] == 2:
            adjusted_report = adjusted_report.model_copy(
                update={
                    "total_net_return": 0.7,
                    "average_net_return": 0.14,
                }
            )
        dump_model(evaluation_path, adjusted_report)

        result = runner.invoke(
            app,
            [
                "score",
                "--policy",
                str(artifact_path),
                "--evaluation",
                str(evaluation_path),
                "--output",
                str(score_path),
                "--registry-root",
                str(registry_root),
            ],
        )
        assert result.exit_code == 0, result.stdout

    champion_candidate = ranked_candidates[0]
    challenger_candidate = ranked_candidates[1]
    champion_score_path = candidate_score_dir / f"{Path(champion_candidate['artifact_path']).stem}-score.json"

    result = runner.invoke(
        app,
        [
            "export-policy",
            "--policy",
            champion_candidate["artifact_path"],
            "--score",
            str(champion_score_path),
            "--output",
            str(champion_export),
        ],
    )
    assert result.exit_code == 0, result.stdout

    champion_report = tmp_path / "outputs" / "champion-paper-sim.md"
    champion_report.write_text("# champion paper sim\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "record-paper-sim",
            "--registry-root",
            str(registry_root),
            "--policy-id",
            champion_candidate["policy_id"],
            "--report",
            str(champion_report),
        ],
    )
    assert result.exit_code == 0, result.stdout

    champion_artifact = load_model(Path(champion_candidate["artifact_path"]), PolicyArtifact)
    registry = LocalRegistryStore(registry_root)
    champion_record = registry.get_record(champion_artifact.policy_id)
    assert champion_record is not None
    assert champion_record.paper_sim_evidence_id is not None

    decision = registry.promote_candidate(
        champion_artifact.policy_id,
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
            paper_sim_evidence_id=champion_record.paper_sim_evidence_id,
            deployment_artifact_path=str(champion_export),
            runtime_uses_inference_artifact_only=True,
            no_live_learning=True,
            executor_boundary_respected=True,
            selector_boundary_respected=True,
            reproducibility=ReproducibilityMetadata(
                data_snapshot_id=champion_artifact.training_snapshot_id,
                code_commit_hash=champion_artifact.code_commit_hash,
                config_hash=champion_artifact.training_config_hash,
                seed=7,
                runtime_stack={"python": "3.12", "framework": "pytorch"},
                reproducible_within_tolerance=True,
            ),
        ),
    )
    assert decision.decision == "promote"
    assert registry.load_index().champion_policy_id == champion_artifact.policy_id

    comparison_report_id = None
    for candidate in ranked_candidates[1:]:
        candidate_comparison_output = comparison_output.with_name(f"{candidate['policy_id']}-comparison-report.json")
        result = runner.invoke(
            app,
            [
                "compare-policies",
                "--registry-root",
                str(registry_root),
                "--challenger-policy-id",
                candidate["policy_id"],
                "--output",
                str(candidate_comparison_output),
            ],
        )
        assert result.exit_code == 0, result.stdout
        candidate_comparison_payload = json.loads(candidate_comparison_output.read_text(encoding="utf-8"))
        candidate_comparison_report_id = candidate_comparison_payload["comparison_report_id"]

        candidate_report = tmp_path / "outputs" / f"{candidate['policy_id']}-paper-sim.md"
        candidate_report.write_text(f"# paper sim {candidate['policy_id']}\n", encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "record-paper-sim",
                "--registry-root",
                str(registry_root),
                "--policy-id",
                candidate["policy_id"],
                "--report",
                str(candidate_report),
                "--comparison-report-id",
                candidate_comparison_report_id,
            ],
        )
        assert result.exit_code == 0, result.stdout
        if candidate["policy_id"] == challenger_candidate["policy_id"]:
            comparison_report_id = candidate_comparison_report_id

    assert comparison_report_id is not None

    result = runner.invoke(
        app,
        [
            "build-offline-evidence-pack",
            "--registry-root",
            str(registry_root),
            "--inspected-evidence-kind",
            "external-retained-evidence",
            "--authority-status",
            "unconfirmed",
            "--output",
            str(evidence_pack_path),
        ],
    )
    assert result.exit_code == 0, result.stdout

    pack = json.loads(evidence_pack_path.read_text(encoding="utf-8"))
    source = pack["sources"][0]
    challenger_record = next(record for record in source["policy_records"] if record["policy_id"] == challenger_candidate["policy_id"])

    assert source["comparison_report_count"] == len(ranked_candidates) - 1
    assert source["paper_sim_evidence_count"] == len(ranked_candidates)
    assert source["missing_comparison_policy_ids"] == []
    assert source["missing_paper_sim_policy_ids"] == []
    assert challenger_record["comparison_report_id"] == comparison_report_id
    assert challenger_record["paper_sim_evidence_id"] is not None
    assert "Scored challengers still require explicit comparison and paper/sim linkage review." not in source["limitations"]
    assert registry.get_record(champion_candidate["policy_id"]).status == "champion"
    assert registry.get_record(challenger_candidate["policy_id"]).comparison_report_id == comparison_report_id


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
    assert audit["inspected_evidence_kind"] == "external_retained_evidence"
    assert audit["authority_status"] == "unconfirmed"
    assert audit["active_training_backend_counts"] == {"pytorch": 1}
    assert audit["active_numpy_training_backend_count"] == 0
    assert audit["closeout_decision_allowed"] is False
    assert audit["closeout_blockers"] == ["authoritative_scope_not_confirmed"]
    assert audit["audit_scope_verdict"] == "clear_in_inspected_scope"
    assert audit["blocking_reasons"] == []
    assert audit["ready_to_close_numpy_continuity_window"] is True
    assert audit["ready_to_retire_legacy_compat_window"] is True


def test_cli_audit_continuity_blocks_empty_registry_scope(tmp_path: Path) -> None:
    runner = CliRunner()
    registry_root = tmp_path / "registry"
    audit_path = tmp_path / "outputs" / "continuity-audit.json"

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
    assert audit["record_count"] == 0
    assert audit["active_record_count"] == 0
    assert audit["inspected_evidence_kind"] == "external_retained_evidence"
    assert audit["authority_status"] == "unconfirmed"
    assert audit["closeout_decision_allowed"] is False
    assert audit["closeout_blockers"] == [
        "no_active_records_in_registry_scope",
        "authoritative_scope_not_confirmed",
    ]
    assert audit["audit_scope_verdict"] == "blocked"
    assert audit["blocking_reasons"] == ["no_active_records_in_registry_scope"]
    assert audit["ready_to_close_numpy_continuity_window"] is False
    assert audit["ready_to_retire_legacy_compat_window"] is False


def test_cli_compare_policies_and_build_offline_evidence_pack(
    tmp_path: Path,
    trajectory_bundle,
    policy_artifact: PolicyArtifact,
    evaluation_report,
    policy_score,
    training_bundle: tuple,
) -> None:
    runner = CliRunner()
    _, _, training_config = training_bundle
    reward_hash = hash_payload(trajectory_bundle.reward_spec)
    training_hash = hash_payload(training_config)
    registry_root = tmp_path / "registry"
    registry = LocalRegistryStore(registry_root)

    champion_report = evaluation_report.model_copy(
        update={
            "total_net_return": 0.3,
            "average_net_return": 0.06,
        }
    )
    champion_score = policy_score.model_copy(
        update={
            "expected_return_score": 0.06,
            "composite_rank": max(policy_score.composite_rank, 0.84),
        }
    )
    registry.register_candidate(
        policy_artifact,
        trajectory_bundle,
        reward_config_hash=reward_hash,
        training_config_hash=training_hash,
    )
    registry.append_score(policy_artifact.policy_id, champion_score, champion_report)
    champion_export = tmp_path / "champion-inference-artifact.json"
    champion_export.write_text("{}", encoding="utf-8")
    champion_paper_sim = tmp_path / "champion-paper-sim.md"
    champion_paper_sim.write_text("# champion paper sim\n", encoding="utf-8")
    champion_evidence = registry.record_paper_sim_evidence(policy_artifact.policy_id, champion_paper_sim)
    registry.promote_candidate(
        policy_artifact.policy_id,
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
            paper_sim_evidence_id=champion_evidence.evidence_id,
            deployment_artifact_path=str(champion_export),
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
        ),
    )

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
            "total_net_return": 0.7,
            "average_net_return": 0.14,
        }
    )
    challenger_score = policy_score.model_copy(
        update={
            "policy_id": challenger_artifact.policy_id,
            "evaluation_id": challenger_report.evaluation_id,
            "expected_return_score": 0.14,
            "composite_rank": max(policy_score.composite_rank, 0.94),
        }
    )
    registry.register_candidate(
        challenger_artifact,
        trajectory_bundle,
        reward_config_hash=reward_hash,
        training_config_hash=training_hash,
    )
    registry.append_score(challenger_artifact.policy_id, challenger_score, challenger_report)

    comparison_output = tmp_path / "comparison-report.json"
    result = runner.invoke(
        app,
        [
            "compare-policies",
            "--registry-root",
            str(registry_root),
            "--challenger-policy-id",
            challenger_artifact.policy_id,
            "--output",
            str(comparison_output),
        ],
    )
    assert result.exit_code == 0, result.stdout
    comparison_payload = json.loads(comparison_output.read_text(encoding="utf-8"))
    comparison_report_id = comparison_payload["comparison_report_id"]

    challenger_paper_sim = tmp_path / "challenger-paper-sim.md"
    challenger_paper_sim.write_text("# challenger paper sim\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "record-paper-sim",
            "--registry-root",
            str(registry_root),
            "--policy-id",
            challenger_artifact.policy_id,
            "--report",
            str(challenger_paper_sim),
            "--comparison-report-id",
            comparison_report_id,
        ],
    )
    assert result.exit_code == 0, result.stdout

    second_registry_root = tmp_path / "registry-second"
    second_registry = LocalRegistryStore(second_registry_root)
    second_bundle = trajectory_bundle.model_copy(
        update={
            "dataset_spec": trajectory_bundle.dataset_spec.model_copy(update={"slice_id": "fixture-slice-second"}),
        },
        deep=True,
    )
    second_artifact = challenger_artifact.model_copy(
        update={
            "policy_id": f"{challenger_artifact.policy_id}-second",
            "artifact_id": f"{challenger_artifact.artifact_id}-second",
            "training_run_id": f"{challenger_artifact.training_run_id}-second",
        },
        deep=True,
    )
    second_report = challenger_report.model_copy(
        update={
            "policy_id": second_artifact.policy_id,
            "evaluation_id": f"{challenger_report.evaluation_id}-second",
        }
    )
    second_score = challenger_score.model_copy(
        update={
            "policy_id": second_artifact.policy_id,
            "evaluation_id": second_report.evaluation_id,
        }
    )
    second_registry.register_candidate(
        second_artifact,
        second_bundle,
        reward_config_hash=reward_hash,
        training_config_hash=training_hash,
    )
    second_registry.append_score(second_artifact.policy_id, second_score, second_report)

    evidence_pack_path = tmp_path / "offline-evidence-pack.json"
    result = runner.invoke(
        app,
        [
            "build-offline-evidence-pack",
            "--registry-root",
            str(registry_root),
            "--registry-root",
            str(second_registry_root),
            "--inspected-evidence-kind",
            "external-retained-evidence",
            "--output",
            str(evidence_pack_path),
        ],
    )
    assert result.exit_code == 0, result.stdout

    pack = json.loads(evidence_pack_path.read_text(encoding="utf-8"))
    assert pack["source_count"] == 2
    assert challenger_artifact.training_snapshot_id in pack["grouped_by_training_snapshot"]
    assert challenger_artifact.evaluation_surface_id in pack["grouped_by_evaluation_surface"]
    assert comparison_report_id in {
        source_record["comparison_report_id"]
        for source in pack["sources"]
        for source_record in source["policy_records"]
        if source_record["comparison_report_id"] is not None
    }
    assert any(
        source["comparison_report_count"] >= 1 and source["paper_sim_evidence_count"] >= 1
        for source in pack["sources"]
    )


def test_cli_evaluate_directory_uses_tensor_cache_api(
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
    original = EvaluationEngine._evaluate_tensor_cache

    def wrapped(self, *args, **kwargs):
        called["count"] += 1
        return original(self, *args, **kwargs)

    def _boom(*args, **kwargs):
        raise AssertionError("CLI directory evaluate must not iterate JSONL when tensor cache exists")

    def _bridge_boom(self, artifact, observation):
        raise AssertionError("CLI directory evaluate must not use PolicyRuntimeBridge.decide()")

    monkeypatch.setattr(EvaluationEngine, "_evaluate_tensor_cache", wrapped)
    monkeypatch.setattr(TrajectoryDirectoryStore, "iter_records", _boom)
    monkeypatch.setattr(PolicyRuntimeBridge, "decide", _bridge_boom)

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


def test_cli_evaluate_directory_requires_explicit_jsonl_fallback_when_cache_missing(
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

    shutil.rmtree(tensor_cache_directory(trajectories))

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
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)

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
            "--allow-jsonl-fallback",
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


def _write_promotion_evidence(path: Path, evidence: PromotionEvidence) -> None:
    if path.suffix.lower() in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(evidence.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
        return
    dump_model(path, evidence)
