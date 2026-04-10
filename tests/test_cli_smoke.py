from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from quantlab_ml.cli.app import app
from quantlab_ml.common import load_model
from quantlab_ml.contracts import ExecutorPolicyExport
from quantlab_ml.registry import LocalRegistryStore


def test_cli_smoke(repo_root: Path, fixture_path: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    trajectories = tmp_path / "outputs" / "trajectories.json"
    policy = tmp_path / "outputs" / "policy.json"
    evaluation = tmp_path / "outputs" / "evaluation.json"
    score = tmp_path / "outputs" / "score.json"
    exported = tmp_path / "outputs" / "executor_policy.json"
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

    exported_policy = load_model(exported, ExecutorPolicyExport)
    registry = LocalRegistryStore(registry_root)
    assert exported_policy.score_summary["composite_rank"] != 0.0
    assert registry.load_index().champion_policy_id == exported_policy.policy_id
