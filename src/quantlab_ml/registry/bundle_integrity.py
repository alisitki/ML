from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

from pydantic import Field

from quantlab_ml.common import current_code_commit_hash, dump_json_data, utcnow
from quantlab_ml.contracts import EventTokenCacheManifest
from quantlab_ml.contracts.common import QuantBaseModel
from quantlab_ml.registry.bundle_errors import (
    DanglingEventTokenCacheManifestError,
    DanglingTensorCacheManifestError,
    Phase0EmpiricalClosureUnsupportedError,
)
from quantlab_ml.trajectories.streaming_store import TrajectoryDirectoryStore
from quantlab_ml.trajectories.event_token_cache import (
    event_token_cache_manifest_path,
    event_token_cache_payload_status,
    read_event_token_cache_manifest,
)
from quantlab_ml.trajectories.tensor_cache import (
    TensorCacheManifest,
    read_tensor_cache_manifest,
    tensor_cache_manifest_path,
    tensor_cache_payload_status,
)

BundlePayloadClass = Literal["full", "slim"]
_BUNDLE_MANIFEST_FILENAME = "bundle_manifest.json"
_NORMALIZATION_RECEIPT_FILENAME = "normalization_receipt.json"
_TENSOR_CACHE_SUMMARY_FILENAME = "tensor_cache_manifest.summary.json"
_EVENT_TOKEN_CACHE_SUMMARY_FILENAME = "event_token_cache_manifest.summary.json"


class TrajectoryDirectoryIntegrityReport(QuantBaseModel):
    bundle_payload_class: BundlePayloadClass
    replayable: bool
    supports_phase0_empirical_closure: bool
    split_jsonl_presence: dict[str, bool] = Field(default_factory=dict)
    tensor_cache_status: dict[str, object] = Field(default_factory=dict)
    event_token_cache_status: dict[str, object] = Field(default_factory=dict)
    blocking_reasons: list[str] = Field(default_factory=list)


class RetainedBundleNormalizationReceipt(QuantBaseModel):
    original_bundle_path: str
    normalized_bundle_path: str
    original_bundle_class: BundlePayloadClass
    original_sha256_path: str | None = None
    original_file_inventory: list[dict[str, str]] = Field(default_factory=list)
    removed_dangling_files: list[str] = Field(default_factory=list)
    replacement_summary_artifacts: list[str] = Field(default_factory=list)
    normalization_mode: Literal["sibling_copy", "in_place"]
    normalization_timestamp: str
    tool_version: str


def inspect_trajectory_directory(directory: Path) -> TrajectoryDirectoryIntegrityReport:
    resolved_directory = directory.expanduser().resolve()
    split_presence: dict[str, bool] = {}
    if TrajectoryDirectoryStore.is_trajectory_directory(resolved_directory):
        manifest = TrajectoryDirectoryStore.read_manifest(resolved_directory)
        split_presence = {
            split_name: TrajectoryDirectoryStore.split_exists(resolved_directory, split_name)
            for split_name in manifest.split_names
        }
    cache_status = tensor_cache_payload_status(resolved_directory)
    event_cache_status = event_token_cache_payload_status(resolved_directory)
    has_any_split_jsonl = any(split_presence.values())
    has_complete_tensor_cache = bool(cache_status.payload_complete)
    replayable = has_any_split_jsonl or has_complete_tensor_cache
    blocking_reasons: list[str] = []
    if cache_status.manifest_present and not cache_status.payload_complete:
        blocking_reasons.append("dangling_tensor_cache_manifest")
    if event_cache_status.manifest_present and not event_cache_status.payload_complete:
        blocking_reasons.append("dangling_event_token_cache_manifest")
    if not replayable:
        blocking_reasons.append("phase0_empirical_closure_unsupported")
    return TrajectoryDirectoryIntegrityReport(
        bundle_payload_class="full" if replayable else "slim",
        replayable=replayable,
        supports_phase0_empirical_closure=replayable,
        split_jsonl_presence=split_presence,
        tensor_cache_status=cache_status.model_dump(mode="json"),
        event_token_cache_status=event_cache_status.model_dump(mode="json"),
        blocking_reasons=blocking_reasons,
    )


def inspect_retained_bundle(bundle_root: Path) -> TrajectoryDirectoryIntegrityReport | None:
    resolved_bundle_root = bundle_root.expanduser().resolve()
    trajectories_root = resolved_bundle_root / "trajectories"
    if not trajectories_root.exists():
        return None
    return inspect_trajectory_directory(trajectories_root)


def validate_retained_bundle(bundle_root: Path) -> TrajectoryDirectoryIntegrityReport | None:
    report = inspect_retained_bundle(bundle_root)
    if report is None:
        return None
    if "dangling_tensor_cache_manifest" in report.blocking_reasons:
        raise DanglingTensorCacheManifestError(
            detail="retained bundle contains tensor_cache_manifest.json references without readable shard payloads",
            bundle_payload_class=report.bundle_payload_class,
            bundle_root=bundle_root,
        )
    if "dangling_event_token_cache_manifest" in report.blocking_reasons:
        raise DanglingEventTokenCacheManifestError(
            detail="retained bundle contains event_token_cache_manifest.json references without readable shard payloads",
            bundle_payload_class=report.bundle_payload_class,
            bundle_root=bundle_root,
        )
    if report.bundle_payload_class == "full" and report.split_jsonl_presence and not all(report.split_jsonl_presence.values()):
        missing_splits = sorted(
            split_name
            for split_name, present in report.split_jsonl_presence.items()
            if not present
        )
        raise ValueError(
            "retained bundle classified full but missing split JSONL payloads: "
            + ", ".join(missing_splits)
        )
    return report


def infer_bundle_payload_error_for_directory(
    directory: Path,
) -> DanglingTensorCacheManifestError | DanglingEventTokenCacheManifestError | Phase0EmpiricalClosureUnsupportedError | None:
    report = inspect_trajectory_directory(directory)
    resolved_directory = directory.expanduser().resolve()
    if "dangling_tensor_cache_manifest" in report.blocking_reasons:
        return DanglingTensorCacheManifestError(
            detail="trajectory directory contains tensor_cache_manifest.json references without readable shard payloads",
            bundle_payload_class=report.bundle_payload_class,
            bundle_root=resolved_directory,
        )
    if "dangling_event_token_cache_manifest" in report.blocking_reasons:
        return DanglingEventTokenCacheManifestError(
            detail="trajectory directory contains event_token_cache_manifest.json references without readable shard payloads",
            bundle_payload_class=report.bundle_payload_class,
            bundle_root=resolved_directory,
        )
    if not report.replayable:
        return Phase0EmpiricalClosureUnsupportedError(
            detail="trajectory directory is non-replayable; Phase 0 empirical closure is unsupported",
            bundle_payload_class=report.bundle_payload_class,
            bundle_root=resolved_directory,
        )
    return None


def infer_bundle_payload_error_for_evaluation(
    evaluation_path: Path,
) -> Phase0EmpiricalClosureUnsupportedError | DanglingTensorCacheManifestError | DanglingEventTokenCacheManifestError | None:
    resolved_path = evaluation_path.expanduser().resolve()
    bundle_root = resolved_path.parent
    if not _has_retained_bundle_markers(bundle_root):
        return None
    report = inspect_retained_bundle(bundle_root)
    if report is not None and "dangling_tensor_cache_manifest" in report.blocking_reasons:
        return DanglingTensorCacheManifestError(
            detail="evaluation report belongs to a retained bundle with a dangling tensor cache manifest",
            bundle_payload_class=report.bundle_payload_class,
            bundle_root=bundle_root,
        )
    if report is not None and "dangling_event_token_cache_manifest" in report.blocking_reasons:
        return DanglingEventTokenCacheManifestError(
            detail="evaluation report belongs to a retained bundle with a dangling event token cache manifest",
            bundle_payload_class=report.bundle_payload_class,
            bundle_root=bundle_root,
        )
    bundle_payload_class = report.bundle_payload_class if report is not None else "slim"
    return Phase0EmpiricalClosureUnsupportedError(
        detail="evaluation report does not contain diagnostics and the retained bundle is non-replayable",
        bundle_payload_class=bundle_payload_class,
        bundle_root=bundle_root,
    )


def normalize_retained_bundle(
    *,
    bundle_root: Path,
    output_root: Path | None = None,
    in_place: bool = False,
) -> tuple[Path, RetainedBundleNormalizationReceipt]:
    resolved_bundle_root = bundle_root.expanduser().resolve()
    if not resolved_bundle_root.exists():
        raise FileNotFoundError(f"bundle root does not exist: {resolved_bundle_root}")
    if in_place and output_root is not None:
        raise ValueError("output_root must not be set when normalizing in-place")
    if in_place:
        normalized_root = resolved_bundle_root
        normalization_mode: Literal["sibling_copy", "in_place"] = "in_place"
    else:
        normalized_root = (
            output_root.expanduser().resolve()
            if output_root is not None
            else resolved_bundle_root.with_name(f"{resolved_bundle_root.name}-normalized")
        )
        if normalized_root.exists():
            shutil.rmtree(normalized_root)
        shutil.copytree(resolved_bundle_root, normalized_root)
        normalization_mode = "sibling_copy"

    original_report = inspect_retained_bundle(resolved_bundle_root)
    original_bundle_class: BundlePayloadClass = "slim" if original_report is None else original_report.bundle_payload_class
    removed_dangling_files: list[str] = []
    replacement_summary_artifacts: list[str] = []
    normalized_trajectories_root = normalized_root / "trajectories"
    if normalized_trajectories_root.exists():
        normalized_report = inspect_trajectory_directory(normalized_trajectories_root)
        if "dangling_tensor_cache_manifest" in normalized_report.blocking_reasons:
            manifest_path = tensor_cache_manifest_path(normalized_trajectories_root)
            cache_manifest = read_tensor_cache_manifest(normalized_trajectories_root)
            summary_path = manifest_path.with_name(_TENSOR_CACHE_SUMMARY_FILENAME)
            dump_json_data(summary_path, _tensor_cache_summary(cache_manifest))
            replacement_summary_artifacts.append(summary_path.relative_to(normalized_root).as_posix())
            removed_dangling_files.append(manifest_path.relative_to(normalized_root).as_posix())
            manifest_path.unlink()
        if "dangling_event_token_cache_manifest" in normalized_report.blocking_reasons:
            manifest_path = event_token_cache_manifest_path(normalized_trajectories_root)
            cache_manifest = read_event_token_cache_manifest(normalized_trajectories_root)
            summary_path = manifest_path.with_name(_EVENT_TOKEN_CACHE_SUMMARY_FILENAME)
            dump_json_data(summary_path, _event_token_cache_summary(cache_manifest))
            replacement_summary_artifacts.append(summary_path.relative_to(normalized_root).as_posix())
            removed_dangling_files.append(manifest_path.relative_to(normalized_root).as_posix())
            manifest_path.unlink()

    receipt = RetainedBundleNormalizationReceipt(
        original_bundle_path=str(resolved_bundle_root),
        normalized_bundle_path=str(normalized_root),
        original_bundle_class=original_bundle_class,
        original_sha256_path=_existing_sha256_path(resolved_bundle_root),
        original_file_inventory=_file_inventory_with_hashes(resolved_bundle_root),
        removed_dangling_files=removed_dangling_files,
        replacement_summary_artifacts=replacement_summary_artifacts,
        normalization_mode=normalization_mode,
        normalization_timestamp=utcnow().replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        tool_version=current_code_commit_hash(),
    )
    receipt_path = normalized_root / _NORMALIZATION_RECEIPT_FILENAME
    dump_json_data(receipt_path, receipt.model_dump(mode="json"))
    return normalized_root, receipt


def _existing_sha256_path(bundle_root: Path) -> str | None:
    sha_path = bundle_root / "SHA256SUMS"
    if not sha_path.exists():
        return None
    return str(sha_path)


def _file_inventory_with_hashes(bundle_root: Path) -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file():
            continue
        inventory.append(
            {
                "path": path.relative_to(bundle_root).as_posix(),
                "sha256": _sha256(path),
            }
        )
    return inventory


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_cache_summary(cache_manifest: TensorCacheManifest) -> dict[str, object]:
    return {
        "format_version": cache_manifest.format_version,
        "feature_dtype": cache_manifest.feature_dtype,
        "feature_dim": cache_manifest.feature_dim,
        "shard_target_bytes": cache_manifest.shard_target_bytes,
        "split_shard_counts": {
            split_name: split_manifest.shard_count
            for split_name, split_manifest in cache_manifest.splits.items()
        },
        "split_row_counts": {
            split_name: split_manifest.row_count
            for split_name, split_manifest in cache_manifest.splits.items()
        },
    }


def _event_token_cache_summary(cache_manifest: EventTokenCacheManifest) -> dict[str, object]:
    return {
        "format_version": cache_manifest.format_version,
        "event_window_contract_version": cache_manifest.event_window_contract_version,
        "tokenizer_version": cache_manifest.tokenizer_version,
        "lookback_seconds": cache_manifest.lookback_seconds,
        "token_cap": cache_manifest.token_cap,
        "split_shard_counts": {
            split_name: split_manifest.shard_count
            for split_name, split_manifest in cache_manifest.splits.items()
        },
        "split_row_counts": {
            split_name: split_manifest.row_count
            for split_name, split_manifest in cache_manifest.splits.items()
        },
        "split_token_counts": {
            split_name: split_manifest.token_count
            for split_name, split_manifest in cache_manifest.splits.items()
        },
    }


def _has_retained_bundle_markers(bundle_root: Path) -> bool:
    return (bundle_root / "trajectories").exists() or (bundle_root / _BUNDLE_MANIFEST_FILENAME).exists()


__all__ = [
    "RetainedBundleNormalizationReceipt",
    "TrajectoryDirectoryIntegrityReport",
    "infer_bundle_payload_error_for_directory",
    "infer_bundle_payload_error_for_evaluation",
    "inspect_retained_bundle",
    "inspect_trajectory_directory",
    "normalize_retained_bundle",
    "validate_retained_bundle",
]
