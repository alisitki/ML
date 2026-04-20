from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from quantlab_ml.cli.app import app
from quantlab_ml.data import LocalFixtureSource
from quantlab_ml.models.features import observation_feature_segment_manifest, observation_feature_vector
from quantlab_ml.trajectories import TrajectoryBuilder, TrajectoryDirectoryStore
from quantlab_ml.trajectories.builder import read_trajectory_build_diagnostics
from quantlab_ml.trajectories.tensor_cache import read_tensor_cache_diagnostics


def test_feature_segment_manifest_matches_feature_vector(trajectory_bundle) -> None:
    observation = trajectory_bundle.splits["train"][0].steps[0].observation
    segments = observation_feature_segment_manifest(observation)

    assert sum(int(segment["length"]) for segment in segments) == len(observation_feature_vector(observation))
    assert segments[-1]["name"] == "target_asset_index"
    assert segments[-1]["length"] == 1


def test_build_to_directory_writes_sparsity_and_policy_state_diagnostics(
    fixture_path: Path,
    tmp_path: Path,
    dataset_spec,
    training_bundle,
    reward_spec,
) -> None:
    trajectory_spec, action_space, _ = training_bundle
    events = LocalFixtureSource(fixture_path).load_events(dataset_spec)
    builder = TrajectoryBuilder(dataset_spec, trajectory_spec, action_space, reward_spec)
    builder.build_to_directory(events, tmp_path)

    manifest = TrajectoryDirectoryStore.read_manifest(tmp_path)
    build_diag = read_trajectory_build_diagnostics(tmp_path)
    cache_diag = read_tensor_cache_diagnostics(tmp_path)

    assert build_diag["structural_sparsity"]["field_total"] > 0
    assert build_diag["split_window_eligibility"]["train"]["usable_steps_per_symbol"] > 0
    assert build_diag["raw_surface_mask_summary"]["train"]["observation_count"] > 0

    validation_diag = cache_diag.splits["validation"]
    assert validation_diag.empirical_sparsity.row_count == manifest.split_write_stats["validation"].step_count
    assert validation_diag.empirical_sparsity.feature_dim > 0
    assert any(segment.name == "target_asset_index" for segment in validation_diag.empirical_sparsity.segments)
    assert validation_diag.label_histograms.action_counts
    policy_state_counts = validation_diag.policy_state_histograms.previous_position_side_counts
    observed_policy_state_rows = sum(policy_state_counts.values()) + validation_diag.policy_state_histograms.missing_policy_state_count
    assert observed_policy_state_rows == validation_diag.empirical_sparsity.row_count


def test_cli_inspect_sparsity_and_policy_state(
    repo_root: Path,
    fixture_path: Path,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    trajectories = tmp_path / "outputs" / "trajectories"

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
            "inspect-sparsity",
            "--trajectories",
            str(trajectories),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert "structural_sparsity" in payload
    assert "empirical_sparsity" in payload

    result = runner.invoke(
        app,
        [
            "inspect-policy-state",
            "--trajectories",
            str(trajectories),
            "--split",
            "validation",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["split"] == "validation"
    assert "previous_position_side_counts" in payload["policy_state_histograms"]
