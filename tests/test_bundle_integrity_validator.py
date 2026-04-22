from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from quantlab_ml.data import LocalFixtureSource
from quantlab_ml.registry.bundle_errors import (
    DanglingEventTokenCacheManifestError,
    DanglingTensorCacheManifestError,
)
from quantlab_ml.registry.bundle_integrity import inspect_retained_bundle, validate_retained_bundle
from quantlab_ml.trajectories import TrajectoryBuilder


def _build_source_trajectory_root(
    *,
    fixture_path: Path,
    tmp_path: Path,
    dataset_spec,
    training_bundle,
    reward_spec,
) -> Path:
    trajectory_spec, action_space, _ = training_bundle
    builder = TrajectoryBuilder(
        dataset_spec,
        trajectory_spec,
        action_space,
        reward_spec,
    )
    events = LocalFixtureSource(fixture_path).load_events(dataset_spec)
    source_root = tmp_path / "source-trajectories"
    builder.build_to_directory(events, source_root)
    return source_root


def _copy_full_retained_bundle(source_root: Path, bundle_root: Path) -> Path:
    trajectories_root = bundle_root / "trajectories"
    trajectories_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / "manifest.json", trajectories_root / "manifest.json")
    for split_path in source_root.glob("*.jsonl"):
        shutil.copy2(split_path, trajectories_root / split_path.name)
    shutil.copytree(
        source_root / "tensor_cache_v1",
        trajectories_root / "tensor_cache_v1",
        dirs_exist_ok=True,
    )
    return bundle_root


def _copy_dangling_retained_bundle(source_root: Path, bundle_root: Path) -> Path:
    trajectories_root = bundle_root / "trajectories" / "tensor_cache_v1"
    trajectories_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / "manifest.json", bundle_root / "trajectories" / "manifest.json")
    shutil.copy2(
        source_root / "tensor_cache_v1" / "tensor_cache_manifest.json",
        trajectories_root / "tensor_cache_manifest.json",
    )
    return bundle_root


def _copy_dangling_event_token_bundle(source_root: Path, bundle_root: Path) -> Path:
    trajectories_root = bundle_root / "trajectories" / "event_token_cache_v1"
    trajectories_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / "manifest.json", bundle_root / "trajectories" / "manifest.json")
    shutil.copy2(
        source_root / "event_token_cache_v1" / "event_token_cache_manifest.json",
        trajectories_root / "event_token_cache_manifest.json",
    )
    return bundle_root


def test_validate_retained_full_bundle_passes(
    fixture_path: Path,
    tmp_path: Path,
    dataset_spec,
    training_bundle,
    reward_spec,
) -> None:
    source_root = _build_source_trajectory_root(
        fixture_path=fixture_path,
        tmp_path=tmp_path,
        dataset_spec=dataset_spec,
        training_bundle=training_bundle,
        reward_spec=reward_spec,
    )
    bundle_root = _copy_full_retained_bundle(source_root, tmp_path / "full-bundle")

    report = validate_retained_bundle(bundle_root)

    assert report is not None
    assert report.bundle_payload_class == "full"
    assert report.replayable is True
    assert report.supports_phase0_empirical_closure is True
    assert report.blocking_reasons == []


def test_validate_retained_slim_bundle_rejects_dangling_tensor_cache_manifest(
    fixture_path: Path,
    tmp_path: Path,
    dataset_spec,
    training_bundle,
    reward_spec,
) -> None:
    source_root = _build_source_trajectory_root(
        fixture_path=fixture_path,
        tmp_path=tmp_path,
        dataset_spec=dataset_spec,
        training_bundle=training_bundle,
        reward_spec=reward_spec,
    )
    bundle_root = _copy_dangling_retained_bundle(source_root, tmp_path / "dangling-bundle")

    report = inspect_retained_bundle(bundle_root)
    assert report is not None
    assert report.bundle_payload_class == "slim"
    assert "dangling_tensor_cache_manifest" in report.blocking_reasons

    with pytest.raises(DanglingTensorCacheManifestError, match="dangling_tensor_cache_manifest"):
        validate_retained_bundle(bundle_root)


def test_validate_retained_bundle_rejects_dangling_event_token_cache_manifest(
    fixture_path: Path,
    tmp_path: Path,
    dataset_spec,
    training_bundle,
    reward_spec,
) -> None:
    source_root = _build_source_trajectory_root(
        fixture_path=fixture_path,
        tmp_path=tmp_path,
        dataset_spec=dataset_spec,
        training_bundle=training_bundle,
        reward_spec=reward_spec,
    )
    bundle_root = _copy_dangling_event_token_bundle(source_root, tmp_path / "dangling-event-bundle")

    report = inspect_retained_bundle(bundle_root)
    assert report is not None
    assert report.bundle_payload_class == "slim"
    assert "dangling_event_token_cache_manifest" in report.blocking_reasons

    with pytest.raises(DanglingEventTokenCacheManifestError, match="dangling_event_token_cache_manifest"):
        validate_retained_bundle(bundle_root)
