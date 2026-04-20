from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quantlab_ml.cli.app import app
from quantlab_ml.contracts import EvaluationReport
from quantlab_ml.data import LocalFixtureSource
from quantlab_ml.evaluation import EvaluationEngine
from quantlab_ml.trajectories import TrajectoryBuilder, TrajectoryDirectoryStore


def test_evaluation_report_contains_diagnostics(
    trajectory_bundle,
    policy_artifact,
    evaluation_boundary,
) -> None:
    report = EvaluationEngine(evaluation_boundary).evaluate(trajectory_bundle, policy_artifact)

    assert report.diagnostics is not None
    assert report.diagnostics.trade_rate == pytest.approx(report.realized_trade_count / report.total_steps)
    assert report.diagnostics.fee_slippage_burden >= 0.0
    assert report.diagnostics.same_side_streak_stats.count >= 0


def test_evaluation_directory_matches_record_stream_diagnostics(
    fixture_path: Path,
    tmp_path: Path,
    dataset_spec,
    training_bundle,
    reward_spec,
    policy_artifact,
    evaluation_boundary,
) -> None:
    trajectory_spec, action_space, _ = training_bundle
    events = LocalFixtureSource(fixture_path).load_events(dataset_spec)
    builder = TrajectoryBuilder(dataset_spec, trajectory_spec, action_space, reward_spec)
    builder.build_to_directory(events, tmp_path)

    manifest = TrajectoryDirectoryStore.read_manifest(tmp_path)
    cache_report = EvaluationEngine(evaluation_boundary).evaluate_directory(
        manifest=manifest,
        directory=tmp_path,
        artifact=policy_artifact,
        split_name="validation",
    )
    record_report = EvaluationEngine(evaluation_boundary).evaluate_records(
        manifest.dataset_spec,
        manifest.reward_spec,
        TrajectoryDirectoryStore.iter_records(tmp_path, "validation"),
        policy_artifact,
    )

    assert cache_report.diagnostics is not None
    assert record_report.diagnostics is not None
    assert cache_report.diagnostics.trade_rate == pytest.approx(record_report.diagnostics.trade_rate)
    assert cache_report.diagnostics.mean_dwell_steps == pytest.approx(record_report.diagnostics.mean_dwell_steps)
    assert cache_report.diagnostics.flip_rate == pytest.approx(record_report.diagnostics.flip_rate)


def test_cli_inspect_eval_diagnostics(
    repo_root: Path,
    fixture_path: Path,
    tmp_path: Path,
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

    report = EvaluationReport.model_validate_json(evaluation.read_text(encoding="utf-8"))
    assert report.diagnostics is not None

    result = runner.invoke(
        app,
        [
            "inspect-eval-diagnostics",
            "--evaluation",
            str(evaluation),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["evaluation_id"] == report.evaluation_id
    assert payload["diagnostics"]["trade_rate"] == pytest.approx(report.diagnostics.trade_rate)
