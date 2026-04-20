from __future__ import annotations

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from quantlab_ml.cli.app import app
from quantlab_ml.common import dump_model
from quantlab_ml.contracts import EvaluationReport
from quantlab_ml.data import LocalFixtureSource
from quantlab_ml.evaluation import EvaluationEngine
from quantlab_ml.registry.bundle_integrity import normalize_retained_bundle
from quantlab_ml.training import LinearPolicyTrainer
from quantlab_ml.trajectories import TrajectoryBuilder, TrajectoryDirectoryStore


def _build_source_run_root(
    *,
    fixture_path: Path,
    tmp_path: Path,
    dataset_spec,
    training_bundle,
    reward_spec,
    evaluation_boundary,
) -> tuple[Path, object, EvaluationReport]:
    trajectory_spec, action_space, training_config = training_bundle
    source_root = tmp_path / "source-run-root"
    builder = TrajectoryBuilder(dataset_spec, trajectory_spec, action_space, reward_spec)
    events = LocalFixtureSource(fixture_path).load_events(dataset_spec)
    builder.build_to_directory(events, source_root)
    manifest = TrajectoryDirectoryStore.read_manifest(source_root)
    artifact = LinearPolicyTrainer(training_config).train_search_from_directory(manifest, source_root).selected_artifact
    report = EvaluationEngine(evaluation_boundary).evaluate_directory(
        manifest=manifest,
        directory=source_root,
        artifact=artifact,
        split_name="final_untouched_test",
    )
    return source_root, artifact, report


def _make_dangling_bundle(source_root: Path, bundle_root: Path, artifact, report: EvaluationReport) -> Path:
    trajectories_root = bundle_root / "trajectories" / "tensor_cache_v1"
    trajectories_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / "manifest.json", bundle_root / "trajectories" / "manifest.json")
    shutil.copy2(
        source_root / "tensor_cache_v1" / "tensor_cache_manifest.json",
        trajectories_root / "tensor_cache_manifest.json",
    )
    dump_model(bundle_root / "policy.json", artifact)
    dump_model(bundle_root / "evaluation.json", report.model_copy(update={"diagnostics": None}))
    (bundle_root / "bundle_manifest.json").write_text(json.dumps({"retained_bundle_kind": "legacy-slim"}), encoding="utf-8")
    return bundle_root


def test_cli_inspect_commands_surface_phase0_empirical_closure_unsupported_for_slim_bundle(
    repo_root: Path,
    fixture_path: Path,
    tmp_path: Path,
    dataset_spec,
    training_bundle,
    reward_spec,
    evaluation_boundary,
) -> None:
    runner = CliRunner()
    source_root, artifact, report = _build_source_run_root(
        fixture_path=fixture_path,
        tmp_path=tmp_path,
        dataset_spec=dataset_spec,
        training_bundle=training_bundle,
        reward_spec=reward_spec,
        evaluation_boundary=evaluation_boundary,
    )
    dangling_root = _make_dangling_bundle(source_root, tmp_path / "dangling-bundle", artifact, report)
    normalized_root, _ = normalize_retained_bundle(bundle_root=dangling_root)

    for command in (
        [
            "inspect-sparsity",
            "--trajectories",
            str(normalized_root / "trajectories"),
        ],
        [
            "inspect-policy-state",
            "--trajectories",
            str(normalized_root / "trajectories"),
        ],
        [
            "inspect-eval-diagnostics",
            "--evaluation",
            str(normalized_root / "evaluation.json"),
        ],
    ):
        result = runner.invoke(app, command)
        assert result.exit_code != 0
        error_text = "\n".join(
            part
            for part in (result.output, str(result.exception) if result.exception is not None else "")
            if part
        )
        assert "phase0_empirical_closure_unsupported" in error_text
        assert "bundle_payload_class=slim" in error_text


def test_cli_train_and_evaluate_surface_dangling_tensor_cache_manifest_for_legacy_bundle(
    repo_root: Path,
    fixture_path: Path,
    tmp_path: Path,
    dataset_spec,
    training_bundle,
    reward_spec,
    evaluation_boundary,
) -> None:
    runner = CliRunner()
    source_root, artifact, report = _build_source_run_root(
        fixture_path=fixture_path,
        tmp_path=tmp_path,
        dataset_spec=dataset_spec,
        training_bundle=training_bundle,
        reward_spec=reward_spec,
        evaluation_boundary=evaluation_boundary,
    )
    dangling_root = _make_dangling_bundle(source_root, tmp_path / "dangling-bundle", artifact, report)
    evaluation_output = tmp_path / "legacy-evaluation.json"
    policy_output = tmp_path / "legacy-policy.json"

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--trajectories",
            str(dangling_root / "trajectories"),
            "--policy",
            str(dangling_root / "policy.json"),
            "--output",
            str(evaluation_output),
            "--evaluation-config",
            str(repo_root / "configs" / "evaluation" / "default.yaml"),
        ],
    )
    assert result.exit_code != 0
    assert "dangling_tensor_cache_manifest" in result.output
    assert "bundle_payload_class=slim" in result.output

    result = runner.invoke(
        app,
        [
            "train",
            "--trajectories",
            str(dangling_root / "trajectories"),
            "--output",
            str(policy_output),
            "--training-config",
            str(repo_root / "configs" / "training" / "default.yaml"),
        ],
    )
    assert result.exit_code != 0
    assert "dangling_tensor_cache_manifest" in result.output
    assert "bundle_payload_class=slim" in result.output
