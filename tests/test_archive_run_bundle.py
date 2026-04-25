from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_archive_module(repo_root: Path):
    script_path = repo_root / "scripts" / "archive_run_bundle.py"
    spec = importlib.util.spec_from_file_location("archive_run_bundle", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_archive_plan_blocks_denylisted_material(repo_root: Path, tmp_path: Path) -> None:
    archive = _load_archive_module(repo_root)
    source_root = tmp_path / "outputs" / "run-with-secret"
    source_root.mkdir(parents=True)
    (source_root / "policy.json").write_text('{"ok": true}\n', encoding="utf-8")
    (source_root / ".env").write_text("SECRET=value\n", encoding="utf-8")
    (source_root / "id_quantlab").write_text("private-key\n", encoding="utf-8")

    plan = archive.build_archive_plan(source_root=source_root, repo_root=tmp_path)

    assert plan.blocked is True
    assert ".env" in plan.denied_entries
    assert "id_quantlab" in plan.denied_entries
    assert plan.as_dict()["classification"] == "blocked"


def test_archive_plan_rejects_roots_outside_allowlist(repo_root: Path, tmp_path: Path) -> None:
    archive = _load_archive_module(repo_root)
    source_root = tmp_path / "not-outputs" / "run"
    source_root.mkdir(parents=True)
    (source_root / "report.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside archive allowlist"):
        archive.build_archive_plan(source_root=source_root, repo_root=tmp_path)


def test_archive_destination_uses_source_path_layout(repo_root: Path, tmp_path: Path) -> None:
    archive = _load_archive_module(repo_root)
    source_root = tmp_path / "outputs" / "analysis" / "ql031"
    source_root.mkdir(parents=True)
    (source_root / "report.json").write_text("{}\n", encoding="utf-8")

    plan = archive.build_archive_plan(
        source_root=source_root,
        archive_base_uri="s3://quantlab-archive/quantlab",
        repo_root=tmp_path,
    )

    assert plan.destination_prefix == "s3://quantlab-archive/quantlab/local-outputs/analysis/ql031/"


def test_archive_plan_keeps_thin_files_and_marks_heavy_payloads(repo_root: Path, tmp_path: Path) -> None:
    archive = _load_archive_module(repo_root)
    source_root = tmp_path / "outputs" / "proof"
    cache_root = source_root / "trajectories" / "tensor_cache_v1" / "train"
    cache_root.mkdir(parents=True)
    (source_root / "validation_report.md").write_text("report\n", encoding="utf-8")
    (source_root / "trajectories" / "manifest.json").write_text("{}\n", encoding="utf-8")
    (cache_root / "shard_00000_X.pt").write_bytes(b"x" * 1024)

    plan = archive.build_archive_plan(source_root=source_root, repo_root=tmp_path)

    assert plan.replayable is True
    assert plan.retained_class == "full"
    assert "validation_report.md" in plan.thin_keep_files
    assert "trajectories/manifest.json" in plan.thin_keep_files
    assert "trajectories/tensor_cache_v1/train/shard_00000_X.pt" in plan.prune_candidate_files


def test_archive_plan_keeps_post_prune_thin_mirror_evidence(repo_root: Path, tmp_path: Path) -> None:
    archive = _load_archive_module(repo_root)
    source_root = tmp_path / "outputs" / "proof"
    source_root.mkdir(parents=True)
    (source_root / "post_prune_thin_mirror_manifest.json").write_text("{}\n", encoding="utf-8")
    (source_root / "post_prune_thin_mirror_manifest.sha256").write_text(
        "abc  post_prune_thin_mirror_manifest.json\n",
        encoding="utf-8",
    )

    plan = archive.build_archive_plan(source_root=source_root, repo_root=tmp_path)

    assert "post_prune_thin_mirror_manifest.json" in plan.thin_keep_files
    assert "post_prune_thin_mirror_manifest.sha256" in plan.thin_keep_files
