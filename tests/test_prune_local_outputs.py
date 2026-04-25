from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _load_prune_module(repo_root: Path):
    script_path = repo_root / "scripts" / "prune_local_outputs.py"
    spec = importlib.util.spec_from_file_location("prune_local_outputs", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_receipt(root: Path, *, verified: bool = True) -> Path:
    receipt = {
        "receipt_version": "archive_receipt_v1",
        "source_root": str(root.resolve()),
        "archive_destination_prefix": "s3://quantlab-archive/quantlab/local-outputs/run/",
        "verification_status": "verified" if verified else "pending",
        "verified_at": "2026-04-23T00:00:00Z" if verified else None,
        "retained_class": "full",
        "replayable": True,
    }
    path = root / "archive_receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return path


def test_prune_requires_verified_archive_receipt(repo_root: Path, tmp_path: Path) -> None:
    prune = _load_prune_module(repo_root)
    source_root = tmp_path / "outputs" / "run"
    source_root.mkdir(parents=True)
    (source_root / "policy.json").write_text("{}\n", encoding="utf-8")
    receipt = _write_receipt(source_root, verified=False)

    with pytest.raises(ValueError, match="not verified"):
        prune.build_prune_plan(source_root, receipt_path=receipt, repo_root=tmp_path)


def test_prune_plan_blocks_denylisted_entries(repo_root: Path, tmp_path: Path) -> None:
    prune = _load_prune_module(repo_root)
    source_root = tmp_path / "outputs" / "run"
    source_root.mkdir(parents=True)
    (source_root / ".venv").mkdir()
    (source_root / "validation_report.md").write_text("report\n", encoding="utf-8")
    receipt = _write_receipt(source_root)

    plan = prune.build_prune_plan(source_root, receipt_path=receipt, repo_root=tmp_path)

    assert plan.blocked is True
    assert ".venv" in plan.denied_entries


def test_prune_plan_blocks_repo_tracked_files(repo_root: Path, tmp_path: Path) -> None:
    prune = _load_prune_module(repo_root)
    source_root = tmp_path / "outputs" / "run"
    source_root.mkdir(parents=True)
    tracked = source_root / "tracked.txt"
    tracked.write_text("tracked\n", encoding="utf-8")
    _write_receipt(source_root)

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "outputs/run/tracked.txt"], cwd=tmp_path, check=True)

    plan = prune.build_prune_plan(source_root, repo_root=tmp_path)

    assert plan.blocked is True
    assert plan.tracked_entries == ("outputs/run/tracked.txt",)


def test_prune_execute_writes_thin_mirror_and_cache_summary(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    prune = _load_prune_module(repo_root)
    source_root = tmp_path / "outputs" / "run"
    cache_root = source_root / "trajectories" / "tensor_cache_v1" / "train"
    cache_root.mkdir(parents=True)
    manifest = source_root / "trajectories" / "tensor_cache_v1" / "tensor_cache_manifest.json"
    manifest.write_text('{"format_version": "tensor_cache_v1"}\n', encoding="utf-8")
    shard = cache_root / "shard_00000_X.pt"
    shard.write_bytes(b"x" * 1024)
    report = source_root / "validation_report.md"
    report.write_text("keep\n", encoding="utf-8")
    receipt = _write_receipt(source_root)

    plan = prune.build_prune_plan(source_root, receipt_path=receipt, repo_root=tmp_path)

    assert plan.blocked is False
    assert "trajectories/tensor_cache_v1/tensor_cache_manifest.summary.json" in (
        plan.generated_summary_files
    )
    assert report.relative_to(source_root).as_posix() in plan.keep_files
    assert shard.relative_to(source_root).as_posix() in plan.prune_files

    result = prune.execute_prune_plan(plan)

    assert result["pruned_file_count"] >= 2
    assert not shard.exists()
    assert not manifest.exists()
    assert report.exists()
    assert (source_root / "trajectories" / "tensor_cache_v1" / "tensor_cache_manifest.summary.json").exists()
    assert (source_root / "local_prune_receipt.json").exists()
    thin_manifest_path = source_root / "post_prune_thin_mirror_manifest.json"
    thin_manifest_sha_path = source_root / "post_prune_thin_mirror_manifest.sha256"
    assert thin_manifest_path.exists()
    assert thin_manifest_sha_path.exists()


def test_post_prune_thin_manifest_hashes_retained_files_only(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    prune = _load_prune_module(repo_root)
    source_root = tmp_path / "outputs" / "run"
    cache_root = source_root / "trajectories" / "tensor_cache_v1" / "train"
    cache_root.mkdir(parents=True)
    (source_root / "SHA256SUMS").write_text("old-full-archive-checksums\n", encoding="utf-8")
    (source_root / "validation_report.md").write_text("keep\n", encoding="utf-8")
    (source_root / "trajectories" / "tensor_cache_v1" / "tensor_cache_manifest.json").write_text(
        '{"format_version": "tensor_cache_v1"}\n',
        encoding="utf-8",
    )
    heavy_shard = cache_root / "shard_00000_X.pt"
    heavy_shard.write_bytes(b"x" * 1024)
    receipt = _write_receipt(source_root)

    plan = prune.build_prune_plan(source_root, receipt_path=receipt, repo_root=tmp_path)
    result = prune.execute_prune_plan(plan)

    assert not heavy_shard.exists()
    thin_manifest = json.loads(
        (source_root / "post_prune_thin_mirror_manifest.json").read_text(encoding="utf-8")
    )
    retained_paths = {record["path"] for record in thin_manifest["file_inventory"]}
    assert "trajectories/tensor_cache_v1/train/shard_00000_X.pt" not in retained_paths
    assert "validation_report.md" in retained_paths
    assert "SHA256SUMS" in retained_paths
    assert thin_manifest["self_included_in_file_inventory"] is False
    assert "post_prune_thin_mirror_manifest.json" not in retained_paths
    assert "post_prune_thin_mirror_manifest.sha256" not in retained_paths
    assert thin_manifest["heavy_artifact_class_check"]["passed"] is True
    assert thin_manifest["pre_prune_archive_evidence"]["checksum_manifest"] == "SHA256SUMS"
    assert result["post_prune_thin_mirror_manifest"]["sha256"]
