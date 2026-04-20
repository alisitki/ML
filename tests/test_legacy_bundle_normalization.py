from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from quantlab_ml.common import dump_json_data
from quantlab_ml.data import LocalFixtureSource
from quantlab_ml.registry.retention import write_bundle_sha256sums
from quantlab_ml.trajectories import TrajectoryBuilder


def _build_dangling_bundle(
    *,
    fixture_path: Path,
    tmp_path: Path,
    dataset_spec,
    training_bundle,
    reward_spec,
) -> Path:
    trajectory_spec, action_space, _ = training_bundle
    source_root = tmp_path / "source-trajectories"
    builder = TrajectoryBuilder(dataset_spec, trajectory_spec, action_space, reward_spec)
    events = LocalFixtureSource(fixture_path).load_events(dataset_spec)
    builder.build_to_directory(events, source_root)

    bundle_root = tmp_path / "legacy-retained-bundle"
    (bundle_root / "trajectories" / "tensor_cache_v1").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / "manifest.json", bundle_root / "trajectories" / "manifest.json")
    shutil.copy2(
        source_root / "tensor_cache_v1" / "tensor_cache_manifest.json",
        bundle_root / "trajectories" / "tensor_cache_v1" / "tensor_cache_manifest.json",
    )
    dump_json_data(
        bundle_root / "bundle_manifest.json",
        {
            "retained_bundle_kind": "legacy-slim",
            "source_run_root": "/workspace/runs/legacy-retained-bundle",
        },
    )
    write_bundle_sha256sums(bundle_root)
    return bundle_root


def test_normalize_retained_bundle_preserves_original_and_writes_receipt(
    repo_root: Path,
    fixture_path: Path,
    tmp_path: Path,
    dataset_spec,
    training_bundle,
    reward_spec,
) -> None:
    bundle_root = _build_dangling_bundle(
        fixture_path=fixture_path,
        tmp_path=tmp_path,
        dataset_spec=dataset_spec,
        training_bundle=training_bundle,
        reward_spec=reward_spec,
    )
    script_path = repo_root / "scripts" / "normalize_retained_bundle.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--bundle-root",
            str(bundle_root),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout

    normalized_root = bundle_root.with_name(f"{bundle_root.name}-normalized")
    receipt_path = normalized_root / "normalization_receipt.json"
    summary_path = normalized_root / "trajectories" / "tensor_cache_v1" / "tensor_cache_manifest.summary.json"
    manifest_path = normalized_root / "bundle_manifest.json"

    assert (bundle_root / "trajectories" / "tensor_cache_v1" / "tensor_cache_manifest.json").exists()
    assert normalized_root.exists()
    assert not (normalized_root / "trajectories" / "tensor_cache_v1" / "tensor_cache_manifest.json").exists()
    assert summary_path.exists()
    assert receipt_path.exists()
    assert (normalized_root / "SHA256SUMS").exists()

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["original_bundle_path"] == str(bundle_root.resolve())
    assert receipt["normalization_mode"] == "sibling_copy"
    assert receipt["removed_dangling_files"] == ["trajectories/tensor_cache_v1/tensor_cache_manifest.json"]
    assert receipt["replacement_summary_artifacts"] == ["trajectories/tensor_cache_v1/tensor_cache_manifest.summary.json"]
    assert receipt["original_sha256_path"].endswith("SHA256SUMS")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["bundle_payload_class"] == "slim"
    assert manifest["replayable"] is False
    assert manifest["supports_phase0_empirical_closure"] is False
    assert manifest["normalization_receipt_path"] == "normalization_receipt.json"
