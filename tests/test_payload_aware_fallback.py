from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from quantlab_ml.data import LocalFixtureSource
from quantlab_ml.evaluation import EvaluationEngine
from quantlab_ml.registry.bundle_errors import DanglingTensorCacheManifestError
from quantlab_ml.training import LinearPolicyTrainer
from quantlab_ml.trajectories import TrajectoryBuilder, TrajectoryDirectoryStore
from quantlab_ml.trajectories.tensor_cache import tensor_cache_directory


def _build_directory(
    *,
    fixture_path: Path,
    tmp_path: Path,
    dataset_spec,
    training_bundle,
    reward_spec,
) -> Path:
    trajectory_spec, action_space, _ = training_bundle
    builder = TrajectoryBuilder(dataset_spec, trajectory_spec, action_space, reward_spec)
    events = LocalFixtureSource(fixture_path).load_events(dataset_spec)
    builder.build_to_directory(events, tmp_path)
    return tmp_path


def _remove_tensor_cache_payloads(directory: Path) -> None:
    cache_root = tensor_cache_directory(directory)
    for path in sorted(cache_root.iterdir()):
        if path.name == "tensor_cache_manifest.json":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def test_evaluation_directory_uses_jsonl_fallback_only_when_payloads_are_missing(
    fixture_path: Path,
    tmp_path: Path,
    dataset_spec,
    training_bundle,
    reward_spec,
    policy_artifact,
    evaluation_boundary,
) -> None:
    directory = _build_directory(
        fixture_path=fixture_path,
        tmp_path=tmp_path,
        dataset_spec=dataset_spec,
        training_bundle=training_bundle,
        reward_spec=reward_spec,
    )
    _remove_tensor_cache_payloads(directory)
    manifest = TrajectoryDirectoryStore.read_manifest(directory)
    engine = EvaluationEngine(evaluation_boundary)

    with pytest.raises(DanglingTensorCacheManifestError, match="dangling_tensor_cache_manifest"):
        engine.evaluate_directory(
            manifest=manifest,
            directory=directory,
            artifact=policy_artifact,
            split_name="validation",
        )

    report = engine.evaluate_directory(
        manifest=manifest,
        directory=directory,
        artifact=policy_artifact,
        split_name="validation",
        allow_jsonl_fallback=True,
    )

    assert report.total_steps > 0
    assert report.diagnostics is not None


def test_training_directory_uses_jsonl_fallback_only_when_payloads_are_missing(
    fixture_path: Path,
    tmp_path: Path,
    dataset_spec,
    training_bundle,
    reward_spec,
) -> None:
    trajectory_spec, action_space, training_config = training_bundle
    builder = TrajectoryBuilder(dataset_spec, trajectory_spec, action_space, reward_spec)
    events = LocalFixtureSource(fixture_path).load_events(dataset_spec)
    builder.build_to_directory(events, tmp_path)
    _remove_tensor_cache_payloads(tmp_path)

    manifest = TrajectoryDirectoryStore.read_manifest(tmp_path)
    trainer = LinearPolicyTrainer(training_config)

    with pytest.raises(DanglingTensorCacheManifestError, match="dangling_tensor_cache_manifest"):
        trainer.train_search_from_directory(manifest, tmp_path)

    result = trainer.train_search_from_directory(
        manifest,
        tmp_path,
        allow_jsonl_fallback=True,
    )
    summary = result.selected_artifact.training_summary

    assert summary["tensor_cache_used"] is False
    assert summary["jsonl_fallback_used"] is True
