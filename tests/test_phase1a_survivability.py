from __future__ import annotations

import json
from pathlib import Path
import subprocess

from typer.testing import CliRunner
import yaml

from quantlab_ml.cli.app import app
from quantlab_ml.common import hash_payload, load_yaml
from quantlab_ml.trajectories import TrajectoryBuilder, TrajectoryDirectoryStore


def _build_phase1a_directory(
    *,
    tmp_path: Path,
    phase1a_dataset_spec,
    phase1a_events,
    phase1a_training_bundle,
    reward_spec,
) -> Path:
    trajectory_spec, action_space, _ = phase1a_training_bundle
    builder = TrajectoryBuilder(phase1a_dataset_spec, trajectory_spec, action_space, reward_spec)
    output = tmp_path / "trajectories"
    builder.build_to_directory(phase1a_events, output)
    return output


def _write_phase1a_search_config(repo_root: Path, tmp_path: Path, *, epochs: int = 1) -> Path:
    payload = load_yaml(repo_root / "configs" / "training" / "production-phase1a-flat-v2-search.yaml")
    payload["trainer"]["epochs"] = epochs
    payload["trainer"]["candidate_search"] = {
        "seeds": [7],
        "learning_rates": [0.05],
        "l2_weights": [0.00005],
    }
    path = tmp_path / "phase1a-search-small.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_phase1a_materialize_command_reuses_sidecar_and_writes_profile(
    repo_root: Path,
    tmp_path: Path,
    phase1a_dataset_spec,
    phase1a_events,
    phase1a_training_bundle,
    reward_spec,
) -> None:
    runner = CliRunner()
    trajectories = _build_phase1a_directory(
        tmp_path=tmp_path,
        phase1a_dataset_spec=phase1a_dataset_spec,
        phase1a_events=phase1a_events,
        phase1a_training_bundle=phase1a_training_bundle,
        reward_spec=reward_spec,
    )
    config_path = _write_phase1a_search_config(repo_root, tmp_path)
    sidecar = trajectories / "phase1a_supervision_v1"
    profile_path = tmp_path / "phase1a_profile.json"

    first = runner.invoke(
        app,
        [
            "materialize-phase1a-supervision",
            "--trajectories",
            str(trajectories),
            "--training-config",
            str(config_path),
            "--output",
            str(sidecar),
            "--profile-output",
            str(profile_path),
        ],
    )
    assert first.exit_code == 0, first.stdout
    assert "[PROGRESS] marker=materialization_completed" in first.stdout

    second = runner.invoke(
        app,
        [
            "materialize-phase1a-supervision",
            "--trajectories",
            str(trajectories),
            "--training-config",
            str(config_path),
            "--output",
            str(sidecar),
            "--profile-output",
            str(profile_path),
        ],
    )
    assert second.exit_code == 0, second.stdout
    assert "[PROGRESS] marker=materialization_completed" in second.stdout
    assert (sidecar / "manifest.json").exists()

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["profile_version"] == "phase1a_profile_v1"
    assert profile["summary"]["materialization_reused"] is True
    assert profile["summary"]["tensor_cache_used"] is True
    assert profile["summary"]["phase1a_supervision_used"] is True


def test_phase1a_materialize_supports_symlinked_trajectory_paths(
    repo_root: Path,
    tmp_path: Path,
    phase1a_dataset_spec,
    phase1a_events,
    phase1a_training_bundle,
    reward_spec,
) -> None:
    runner = CliRunner()
    trajectories = _build_phase1a_directory(
        tmp_path=tmp_path / "real-root",
        phase1a_dataset_spec=phase1a_dataset_spec,
        phase1a_events=phase1a_events,
        phase1a_training_bundle=phase1a_training_bundle,
        reward_spec=reward_spec,
    )
    linked_trajectories = tmp_path / "linked-trajectories"
    linked_trajectories.symlink_to(trajectories, target_is_directory=True)
    config_path = _write_phase1a_search_config(repo_root, tmp_path)
    profile_path = tmp_path / "phase1a_profile_symlink.json"

    result = runner.invoke(
        app,
        [
            "materialize-phase1a-supervision",
            "--trajectories",
            str(linked_trajectories),
            "--training-config",
            str(config_path),
            "--output",
            str(linked_trajectories / "phase1a_supervision_v1"),
            "--profile-output",
            str(profile_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "[PROGRESS] marker=materialization_completed" in result.stdout

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["summary"]["phase1a_supervision_used"] is True
    manifest_payload = json.loads((trajectories / "phase1a_supervision_v1" / "manifest.json").read_text(encoding="utf-8"))
    shard_path = manifest_payload["splits"]["development"]["shards"][0]["policy_state_path"]
    assert shard_path.startswith("phase1a_supervision_v1/")


def test_phase1a_train_and_evaluate_merge_profile_and_write_partial_outputs(
    repo_root: Path,
    tmp_path: Path,
    phase1a_dataset_spec,
    phase1a_events,
    phase1a_training_bundle,
    reward_spec,
) -> None:
    runner = CliRunner()
    trajectories = _build_phase1a_directory(
        tmp_path=tmp_path,
        phase1a_dataset_spec=phase1a_dataset_spec,
        phase1a_events=phase1a_events,
        phase1a_training_bundle=phase1a_training_bundle,
        reward_spec=reward_spec,
    )
    config_path = _write_phase1a_search_config(repo_root, tmp_path, epochs=1)
    sidecar = trajectories / "phase1a_supervision_v1"
    profile_path = tmp_path / "phase1a_profile.json"
    policy_path = tmp_path / "outputs" / "policy.json"
    evaluation_path = tmp_path / "outputs" / "evaluation.json"

    result = runner.invoke(
        app,
        [
            "materialize-phase1a-supervision",
            "--trajectories",
            str(trajectories),
            "--training-config",
            str(config_path),
            "--output",
            str(sidecar),
            "--profile-output",
            str(profile_path),
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
            str(policy_path),
            "--training-config",
            str(config_path),
            "--profile-output",
            str(profile_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert policy_path.exists()
    assert (tmp_path / "outputs" / "policy_search.partial.json").exists()
    assert (tmp_path / "outputs" / "policy_candidates_partial").exists()
    assert (tmp_path / "outputs" / "checkpoints" / "phase1a_search_state.json").exists()

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--trajectories",
            str(trajectories),
            "--policy",
            str(policy_path),
            "--output",
            str(evaluation_path),
            "--evaluation-config",
            str(repo_root / "configs" / "evaluation" / "default.yaml"),
            "--split",
            "validation",
            "--profile-output",
            str(profile_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert evaluation_path.exists()

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["summary"]["tensor_cache_used"] is True
    assert profile["summary"]["phase1a_supervision_used"] is True
    assert profile["summary"]["compiled_v2_eval_used"] is True
    assert profile["summary"]["jsonl_fallback_used"] is False
    assert profile["summary"]["batch_compute_wall_sec"] >= 0.0
    assert profile["summary"]["evaluation_rows_per_sec"] > 0.0
    assert profile["summary"]["joint_ce_loss"] >= 0.0
    assert profile["summary"]["aux_value_loss_raw"] >= 0.0
    assert profile["summary"]["aux_value_loss_weighted"] >= 0.0
    assert profile["summary"]["total_loss"] >= 0.0
    assert profile["summary"]["action_logit_abs_max"] >= 0.0
    assert profile["summary"]["action_entropy"] >= 0.0
    assert profile["summary"]["value_pred_abs_max"] >= 0.0
    assert profile["summary"]["value_grad_norm_pre_clip"] >= profile["summary"]["value_grad_norm_post_clip"] >= 0.0
    assert profile["summary"]["clip_applied_count"] >= 0
    assert profile["summary"]["first_nonfinite_component"] is None
    assert profile["summary"]["first_nonfinite_batch_context"] is None


def test_phase1a_resume_mismatch_fails_closed(
    repo_root: Path,
    tmp_path: Path,
    phase1a_dataset_spec,
    phase1a_events,
    phase1a_training_bundle,
    reward_spec,
) -> None:
    runner = CliRunner()
    trajectories = _build_phase1a_directory(
        tmp_path=tmp_path,
        phase1a_dataset_spec=phase1a_dataset_spec,
        phase1a_events=phase1a_events,
        phase1a_training_bundle=phase1a_training_bundle,
        reward_spec=reward_spec,
    )
    config_path = _write_phase1a_search_config(repo_root, tmp_path, epochs=1)
    sidecar = trajectories / "phase1a_supervision_v1"
    materialize = runner.invoke(
        app,
        [
            "materialize-phase1a-supervision",
            "--trajectories",
            str(trajectories),
            "--training-config",
            str(config_path),
            "--output",
            str(sidecar),
        ],
    )
    assert materialize.exit_code == 0, materialize.stdout

    output_path = tmp_path / "outputs" / "policy.json"
    checkpoints_root = tmp_path / "outputs" / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    state_path = checkpoints_root / "phase1a_search_state.json"
    state_path.write_text(
        json.dumps(
            {
                "compatibility": {
                    "tensor_cache_manifest_hash": "wrong",
                    "phase1a_supervision_manifest_hash": "wrong",
                    "training_config_hash": "wrong",
                    "phase1a_compute_dtype": "float64",
                    "action_space_version": "wrong",
                    "policy_state_feature_version": "wrong",
                },
                "selection_runs": {},
                "candidate_results": [],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "train",
            "--trajectories",
            str(trajectories),
            "--output",
            str(output_path),
            "--training-config",
            str(config_path),
            "--resume-search",
        ],
    )
    assert result.exit_code != 0
    combined_output = result.stdout
    stderr = getattr(result, "stderr", "")
    if isinstance(stderr, str):
        combined_output += stderr
    if result.exception is not None:
        combined_output += str(result.exception)
    assert "resume compatibility mismatch" in combined_output.lower()


def test_retain_remote_run_bundle_allows_incomplete_runs(
    repo_root: Path,
    tmp_path: Path,
    phase1a_dataset_spec,
    phase1a_events,
    phase1a_training_bundle,
    reward_spec,
) -> None:
    trajectories = _build_phase1a_directory(
        tmp_path=tmp_path,
        phase1a_dataset_spec=phase1a_dataset_spec,
        phase1a_events=phase1a_events,
        phase1a_training_bundle=phase1a_training_bundle,
        reward_spec=reward_spec,
    )
    bundle_root = tmp_path / "bundle"
    (bundle_root / "trajectories").mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cp",
            "-R",
            str(trajectories / "."),
            str(bundle_root / "trajectories"),
        ],
        check=True,
    )
    (bundle_root / "registry").mkdir(parents=True, exist_ok=True)
    (bundle_root / "build.log").write_text("[STARTED]\n[COMPLETED]\n", encoding="utf-8")
    (bundle_root / "build.exit").write_text("0\n", encoding="utf-8")
    (bundle_root / "phase1a_profile.json").write_text(
        json.dumps({"profile_version": "phase1a_profile_v1", "stages": {}, "summary": {}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            str(repo_root / ".venv" / "bin" / "python"),
            str(repo_root / "scripts" / "retain_remote_run_bundle.py"),
            "--allow-incomplete",
            "--bundle-root",
            str(bundle_root),
            "--source-run-root",
            "/workspace/runs/ql031-phase1a",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads((bundle_root / "bundle_manifest.json").read_text(encoding="utf-8"))
    assert payload["run_completion_state"] == "partial"
    assert payload["known_partial"] is True
