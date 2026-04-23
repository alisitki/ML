from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from quantlab_ml.common import current_code_commit_hash, dump_json_data, hash_payload, load_model, load_yaml, utcnow
from quantlab_ml.contracts import (
    ComparisonReport,
    EvaluationReport,
    InferenceArtifactExport,
    PaperSimEvidenceRecord,
    PolicyArtifact,
    PolicyScore,
    PromotionDecisionRecord,
)
from quantlab_ml.registry.bundle_integrity import inspect_retained_bundle
from quantlab_ml.registry.store import LocalRegistryStore
from quantlab_ml.trajectories.event_token_cache import (
    event_token_cache_retention_receipt_path,
    read_event_token_cache_manifest,
    read_event_token_cache_retention_receipt,
)
from quantlab_ml.trajectories.tensor_cache import read_tensor_cache_manifest

_MANIFEST_FILENAME = "bundle_manifest.json"
_SHA256SUMS_FILENAME = "SHA256SUMS"
_TENSOR_CACHE_SUMMARY_FILENAME = "tensor_cache_manifest.summary.json"
_EVENT_TOKEN_CACHE_SUMMARY_FILENAME = "event_token_cache_manifest.summary.json"
_DEFAULT_AUTHORITY_NOTE = (
    "derived from a controlled remote run; retained copy remains external-retained-evidence "
    "and is not relabeled authoritative evidence"
)
_INSTANCE_REQUIRED_KEYS = (
    "gpu_model",
    "vcpu_count",
    "ram_gb",
    "disk_gb",
    "storage_note",
    "offer_id",
)


def build_retained_bundle_manifest(
    *,
    bundle_root: Path,
    source_run_root: str,
    retained_bundle_kind: str = "repo_local_retained_bundle",
    retained_bundle_authority_note: str = _DEFAULT_AUTHORITY_NOTE,
    source_registry_root: str | None = None,
    source_repo_commit_sha: str | None = None,
    instance_metadata_path: Path | None = None,
    ql031_status_path: Path | None = None,
    config_copies: list[tuple[Path, str]] | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    resolved_bundle_root = bundle_root.expanduser().resolve()
    if not resolved_bundle_root.exists() or not resolved_bundle_root.is_dir():
        raise FileNotFoundError(f"bundle root does not exist: {resolved_bundle_root}")

    copied_configs = _copy_config_files(bundle_root=resolved_bundle_root, config_copies=config_copies or [])
    registry_root = resolved_bundle_root / "registry"
    if not registry_root.exists() and not allow_incomplete:
        raise FileNotFoundError(f"bundle registry root does not exist: {registry_root}")
    store = LocalRegistryStore(registry_root) if registry_root.exists() else None

    policy = _maybe_load_model(resolved_bundle_root / "policy.json", PolicyArtifact)
    evaluation = _maybe_load_model(resolved_bundle_root / "evaluation.json", EvaluationReport)
    score = _maybe_load_model(resolved_bundle_root / "score.json", PolicyScore)
    inference_artifact = _maybe_load_model(resolved_bundle_root / "inference_artifact.json", InferenceArtifactExport)

    trajectory_summary = _trajectory_summary(resolved_bundle_root / "trajectories")
    manifest = {
        "retained_bundle_kind": retained_bundle_kind,
        "retained_bundle_authority_note": retained_bundle_authority_note,
        "generated_at_local": utcnow().replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_repo_commit_sha": source_repo_commit_sha or current_code_commit_hash(),
        "source_run_root": source_run_root,
        "source_registry_root": source_registry_root or f"{source_run_root.rstrip('/')}/registry",
        "retained_bundle_path": str(resolved_bundle_root),
        "run_completion_state": "partial" if allow_incomplete else "complete",
        "known_partial": allow_incomplete,
        "bundle_disk_usage_human": _bundle_disk_usage_human(resolved_bundle_root),
        "bundle_unique_bytes": _bundle_unique_bytes(resolved_bundle_root),
        "run_started_at_logline": _first_non_empty_line(
            resolved_bundle_root / "build.log",
            fallback_paths=[resolved_bundle_root / "train.log"],
        ),
        "run_export_message": _last_non_empty_line(resolved_bundle_root / "export.log"),
        "stage_exit_codes": _stage_exit_codes(resolved_bundle_root),
        "training_summary": dict(policy.training_summary) if policy is not None else None,
        "policy_summary": _policy_summary(policy),
        "evaluation_summary": _evaluation_summary(evaluation),
        "score_summary": _score_summary(score),
        "inference_artifact_summary": _inference_artifact_summary(inference_artifact),
        "inspect_s3_summary": _inspect_s3_summary(resolved_bundle_root / "inspect_s3.json"),
        "trajectory_summary": trajectory_summary,
        "registry_summary": _registry_summary(store) if store is not None else None,
        "same_root_proof_summary": _same_root_proof_summary(store) if store is not None else None,
        "instance_metadata": _instance_metadata(instance_metadata_path),
        "ql031_integration_summary": _ql031_integration_summary(ql031_status_path),
        "config_copies": copied_configs,
        "copied_files": _copied_files(resolved_bundle_root),
        "hardlink_map": _hardlink_map(resolved_bundle_root),
    }
    if trajectory_summary is not None:
        if "event_token_cache_manifest_hash" in trajectory_summary:
            manifest["event_token_cache_manifest_hash"] = trajectory_summary["event_token_cache_manifest_hash"]
        if "tensor_cache_manifest_hash" in trajectory_summary:
            manifest["tensor_cache_manifest_hash"] = trajectory_summary["tensor_cache_manifest_hash"]
    manifest = _attach_bundle_integrity_metadata(manifest, resolved_bundle_root)
    return {key: value for key, value in manifest.items() if value not in (None, [], {})}


def write_retained_bundle_manifest(
    *,
    bundle_root: Path,
    source_run_root: str,
    retained_bundle_kind: str = "repo_local_retained_bundle",
    retained_bundle_authority_note: str = _DEFAULT_AUTHORITY_NOTE,
    source_registry_root: str | None = None,
    source_repo_commit_sha: str | None = None,
    instance_metadata_path: Path | None = None,
    ql031_status_path: Path | None = None,
    config_copies: list[tuple[Path, str]] | None = None,
    allow_incomplete: bool = False,
) -> Path:
    manifest = build_retained_bundle_manifest(
        bundle_root=bundle_root,
        source_run_root=source_run_root,
        retained_bundle_kind=retained_bundle_kind,
        retained_bundle_authority_note=retained_bundle_authority_note,
        source_registry_root=source_registry_root,
        source_repo_commit_sha=source_repo_commit_sha,
        instance_metadata_path=instance_metadata_path,
        ql031_status_path=ql031_status_path,
        config_copies=config_copies,
        allow_incomplete=allow_incomplete,
    )
    output_path = bundle_root.expanduser().resolve() / _MANIFEST_FILENAME
    dump_json_data(output_path, manifest)
    return output_path


def write_bundle_sha256sums(bundle_root: Path) -> Path:
    resolved_bundle_root = bundle_root.expanduser().resolve()
    output_path = resolved_bundle_root / _SHA256SUMS_FILENAME
    lines = []
    for path in sorted(_iter_bundle_files(resolved_bundle_root)):
        if path.name == _SHA256SUMS_FILENAME:
            continue
        lines.append(f"{_sha256(path)}  {path.relative_to(resolved_bundle_root).as_posix()}")
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path


def refresh_existing_retained_bundle_manifest(
    bundle_root: Path,
    *,
    extra_updates: dict[str, Any] | None = None,
) -> Path | None:
    resolved_bundle_root = bundle_root.expanduser().resolve()
    manifest_path = resolved_bundle_root / _MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["trajectory_summary"] = _trajectory_summary(resolved_bundle_root / "trajectories")
    if extra_updates:
        payload.update(extra_updates)
    payload = _attach_bundle_integrity_metadata(payload, resolved_bundle_root)
    dump_json_data(manifest_path, {key: value for key, value in payload.items() if value not in (None, [], {})})
    return manifest_path


def _copy_config_files(*, bundle_root: Path, config_copies: list[tuple[Path, str]]) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for source_path, bundle_relative_path in config_copies:
        resolved_source = source_path.expanduser().resolve()
        if not resolved_source.exists() or not resolved_source.is_file():
            raise FileNotFoundError(f"config source path does not exist: {resolved_source}")
        destination = bundle_root / bundle_relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved_source, destination)
        copied.append(
            {
                "bundle_relative_path": bundle_relative_path,
                "repo_source_path": str(resolved_source),
                "sha256": _sha256(destination),
            }
        )
    return copied


def _maybe_load_model(path: Path, model_type: type[Any]) -> Any | None:
    if not path.exists():
        return None
    return load_model(path, model_type)


def _policy_summary(policy: PolicyArtifact | None) -> dict[str, Any] | None:
    if policy is None:
        return None
    summary = {
        "policy_id": policy.policy_id,
        "artifact_id": policy.artifact_id,
        "training_snapshot_id": policy.training_snapshot_id,
        "evaluation_surface_id": policy.evaluation_surface_id,
        "reward_version": policy.reward_version,
        "created_at": policy.created_at.isoformat().replace("+00:00", "Z"),
        "target_asset": policy.target_asset,
    }
    return summary


def _evaluation_summary(evaluation: EvaluationReport | None) -> dict[str, Any] | None:
    if evaluation is None:
        return None
    return {
        "evaluation_id": evaluation.evaluation_id,
        "policy_id": evaluation.policy_id,
        "total_net_return": evaluation.total_net_return,
        "total_steps": evaluation.total_steps,
        "realized_trade_count": evaluation.realized_trade_count,
        "active_date_range": evaluation.active_date_range.model_dump(mode="json"),
    }


def _score_summary(score: PolicyScore | None) -> dict[str, Any] | None:
    if score is None:
        return None
    return score.model_dump(mode="json")


def _inference_artifact_summary(inference_artifact: InferenceArtifactExport | None) -> dict[str, Any] | None:
    if inference_artifact is None:
        return None
    return {
        "policy_id": inference_artifact.policy_id,
        "artifact_id": inference_artifact.artifact_id,
        "runtime_adapter": inference_artifact.runtime_adapter,
        "created_at": inference_artifact.created_at.isoformat().replace("+00:00", "Z"),
    }


def _inspect_s3_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed_keys = (
        "matched_partition_count",
        "successful_day_count",
        "successful_days_sample_first",
        "exchange_counts",
        "stream_counts",
    )
    return {key: payload[key] for key in allowed_keys if key in payload}


def _trajectory_summary(trajectories_root: Path) -> dict[str, Any] | None:
    manifest_path = trajectories_root / "manifest.json"
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "split_write_stats": payload.get("split_write_stats", {}),
    }
    cache_manifest_path = trajectories_root / "tensor_cache_v1" / "tensor_cache_manifest.json"
    cache_summary_path = trajectories_root / "tensor_cache_v1" / _TENSOR_CACHE_SUMMARY_FILENAME
    event_cache_manifest_path = trajectories_root / "event_token_cache_v1" / "event_token_cache_manifest.json"
    event_cache_summary_path = trajectories_root / "event_token_cache_v1" / _EVENT_TOKEN_CACHE_SUMMARY_FILENAME
    event_cache_receipt_path = event_token_cache_retention_receipt_path(trajectories_root)
    if cache_manifest_path.exists():
        cache_manifest_payload = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        cache_manifest = read_tensor_cache_manifest(trajectories_root)
        summary["tensor_cache_manifest_hash"] = hash_payload(cache_manifest_payload)
        summary["tensor_cache_feature_dtype"] = cache_manifest.feature_dtype
        summary["tensor_cache_feature_dim"] = cache_manifest.feature_dim
        summary["tensor_cache_split_shard_counts"] = {
            split_name: split_manifest.shard_count for split_name, split_manifest in cache_manifest.splits.items()
        }
        summary["tensor_cache_split_row_counts"] = {
            split_name: split_manifest.row_count for split_name, split_manifest in cache_manifest.splits.items()
        }
    elif cache_summary_path.exists():
        cache_summary = json.loads(cache_summary_path.read_text(encoding="utf-8"))
        summary["tensor_cache_feature_dtype"] = cache_summary.get("feature_dtype")
        summary["tensor_cache_feature_dim"] = cache_summary.get("feature_dim")
        summary["tensor_cache_split_shard_counts"] = cache_summary.get("split_shard_counts", {})
        summary["tensor_cache_split_row_counts"] = cache_summary.get("split_row_counts", {})
    if event_cache_manifest_path.exists():
        event_cache_manifest_payload = json.loads(event_cache_manifest_path.read_text(encoding="utf-8"))
        event_cache_manifest = read_event_token_cache_manifest(trajectories_root)
        summary["event_token_cache_manifest_hash"] = hash_payload(event_cache_manifest_payload)
        summary["event_token_cache_contract_version"] = event_cache_manifest.event_window_contract_version
        summary["event_token_cache_tokenizer_version"] = event_cache_manifest.tokenizer_version
        summary["event_token_cache_selection_policy_id"] = event_cache_manifest.selection_policy_id
        summary["event_token_cache_selector_params_hash"] = event_cache_manifest.selector_params_hash
        summary["event_token_cache_selection_hyperparameters"] = (
            event_cache_manifest.selection_hyperparameters.model_dump(mode="json")
        )
        summary["event_token_cache_token_cap"] = event_cache_manifest.token_cap
        summary["event_token_cache_lookback_seconds"] = event_cache_manifest.lookback_seconds
        summary["event_token_cache_split_shard_counts"] = {
            split_name: split_manifest.shard_count
            for split_name, split_manifest in event_cache_manifest.splits.items()
        }
        summary["event_token_cache_split_row_counts"] = {
            split_name: split_manifest.row_count
            for split_name, split_manifest in event_cache_manifest.splits.items()
        }
        summary["event_token_cache_split_token_counts"] = {
            split_name: split_manifest.token_count
            for split_name, split_manifest in event_cache_manifest.splits.items()
        }
    elif event_cache_summary_path.exists():
        event_cache_summary = json.loads(event_cache_summary_path.read_text(encoding="utf-8"))
        summary["event_token_cache_contract_version"] = event_cache_summary.get("event_window_contract_version")
        summary["event_token_cache_tokenizer_version"] = event_cache_summary.get("tokenizer_version")
        summary["event_token_cache_selection_policy_id"] = event_cache_summary.get("selection_policy_id")
        summary["event_token_cache_selector_params_hash"] = event_cache_summary.get("selector_params_hash")
        summary["event_token_cache_selection_hyperparameters"] = event_cache_summary.get("selection_hyperparameters")
        summary["event_token_cache_token_cap"] = event_cache_summary.get("token_cap")
        summary["event_token_cache_lookback_seconds"] = event_cache_summary.get("lookback_seconds")
        summary["event_token_cache_split_shard_counts"] = event_cache_summary.get("split_shard_counts", {})
        summary["event_token_cache_split_row_counts"] = event_cache_summary.get("split_row_counts", {})
        summary["event_token_cache_split_token_counts"] = event_cache_summary.get("split_token_counts", {})
    if event_cache_receipt_path.exists():
        receipt = read_event_token_cache_retention_receipt(trajectories_root)
        summary["event_token_cache_retention_receipt_path"] = (
            event_cache_receipt_path.relative_to(trajectories_root).as_posix()
        )
        summary["event_token_cache_retained_shard_count"] = receipt.retained_shard_count
        summary["event_token_cache_missing_shard_count"] = receipt.missing_shard_count
        summary["event_token_cache_retained_payload_count"] = receipt.retained_payload_count
        summary["event_token_cache_missing_payload_count"] = receipt.missing_payload_count
    return summary


def _attach_bundle_integrity_metadata(manifest: dict[str, Any], bundle_root: Path) -> dict[str, Any]:
    report = inspect_retained_bundle(bundle_root)
    if report is None:
        return manifest
    manifest["bundle_payload_class"] = report.bundle_payload_class
    manifest["replayable"] = report.replayable
    manifest["supports_phase0_empirical_closure"] = report.supports_phase0_empirical_closure
    manifest["known_partial"] = not report.replayable
    manifest["non_replayable"] = not report.replayable
    manifest["bundle_integrity"] = report.model_dump(mode="json")
    return manifest


def _registry_summary(store: LocalRegistryStore) -> dict[str, Any]:
    index = store.load_index()
    records = store.list_records()
    comparisons = store.list_comparison_reports()
    paper_sim = store.list_paper_sim_evidence()
    promotions = _list_promotion_decisions(store)
    return {
        "champion_policy_id": index.champion_policy_id,
        "challenger_policy_ids": index.challenger_policy_ids,
        "record_count": len(records),
        "comparison_report_count": len(comparisons),
        "paper_sim_evidence_count": len(paper_sim),
        "promotion_decision_count": len(promotions),
    }


def _same_root_proof_summary(store: LocalRegistryStore) -> dict[str, Any] | None:
    comparisons = store.list_comparison_reports()
    paper_sim = store.list_paper_sim_evidence()
    promotions = _list_promotion_decisions(store)
    if not comparisons and not paper_sim and not promotions:
        return None
    records = store.list_records()
    linked_challengers = sorted(
        record.policy_id
        for record in records
        if record.status == "challenger"
        and record.comparison_report_id is not None
        and record.paper_sim_evidence_id is not None
    )
    return {
        "comparison_report_ids": [report.comparison_report_id for report in comparisons],
        "paper_sim_evidence_ids": [record.evidence_id for record in paper_sim],
        "promotion_decision_ids": [decision.decision_id for decision in promotions],
        "linked_challenger_policy_ids": linked_challengers,
        "compared_champion_policy_ids": sorted({report.champion_policy_id for report in comparisons}),
        "compared_challenger_policy_ids": sorted({report.challenger_policy_id for report in comparisons}),
    }


def _list_promotion_decisions(store: LocalRegistryStore) -> list[PromotionDecisionRecord]:
    return [
        load_model(path, PromotionDecisionRecord)
        for path in sorted(store.promotions_dir.glob("*.json"))
    ]


def _instance_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(f"instance metadata path does not exist: {resolved_path}")
    if resolved_path.suffix.lower() == ".json":
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    else:
        payload = load_yaml(resolved_path)
    missing = [key for key in _INSTANCE_REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(f"instance metadata is missing required keys: {', '.join(missing)}")
    return payload


def _ql031_integration_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(f"ql031 status path does not exist: {resolved_path}")
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    summary = {
        "ql031_status": payload.get("status"),
        "ql031_output_root": payload.get("output_root"),
        "external_search_roots": payload.get("external_search_roots", []),
        "distinct_retained_candidate_count": len(payload.get("distinct_retained_candidates", [])),
        "comparison_report_count": len(payload.get("comparison_reports", [])),
    }
    if payload.get("distinct_retained_candidates"):
        summary["distinct_surface_identity"] = payload["distinct_retained_candidates"][0].get("surface_identity")
    return summary


def _stage_exit_codes(bundle_root: Path) -> dict[str, int | None]:
    stage_exit_codes: dict[str, int | None] = {}
    for stage in ("build", "train", "evaluate", "score", "export"):
        path = bundle_root / f"{stage}.exit"
        if not path.exists():
            stage_exit_codes[stage] = None
            continue
        text = path.read_text(encoding="utf-8").strip()
        stage_exit_codes[stage] = int(text) if text else None
    return stage_exit_codes


def _first_non_empty_line(path: Path, *, fallback_paths: list[Path] | None = None) -> str | None:
    for candidate in [path, *(fallback_paths or [])]:
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
    return None


def _last_non_empty_line(path: Path) -> str | None:
    if not path.exists():
        return None
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _bundle_disk_usage_human(bundle_root: Path) -> str:
    total_bytes = sum(path.stat().st_size for path in _iter_bundle_files(bundle_root))
    return _human_size(total_bytes)


def _bundle_unique_bytes(bundle_root: Path) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for path in _iter_bundle_files(bundle_root):
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino)
        if key in seen:
            continue
        seen.add(key)
        total += stat.st_size
    return total


def _human_size(size_bytes: int) -> str:
    units = ["B", "K", "M", "G", "T"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)}{unit}"
            return f"{size:.0f}{unit}"
        size /= 1024
    return f"{size_bytes}B"


def _copied_files(bundle_root: Path) -> list[str]:
    return [
        path.relative_to(bundle_root).as_posix()
        for path in _iter_bundle_files(bundle_root)
        if path.name not in {_MANIFEST_FILENAME, _SHA256SUMS_FILENAME}
    ]


def _hardlink_map(bundle_root: Path) -> dict[str, str]:
    inode_map: dict[tuple[int, int], list[Path]] = {}
    for path in _iter_bundle_files(bundle_root):
        stat = path.stat()
        inode_map.setdefault((stat.st_dev, stat.st_ino), []).append(path)
    hardlinks: dict[str, str] = {}
    for paths in inode_map.values():
        if len(paths) < 2:
            continue
        sorted_paths = sorted(paths)
        source = sorted_paths[0].relative_to(bundle_root).as_posix()
        for target in sorted_paths[1:]:
            hardlinks[target.relative_to(bundle_root).as_posix()] = source
    return hardlinks


def _iter_bundle_files(bundle_root: Path) -> list[Path]:
    return [path for path in sorted(bundle_root.rglob("*")) if path.is_file()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
