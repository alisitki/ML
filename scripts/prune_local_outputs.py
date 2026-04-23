#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from archive_run_bundle import (  # noqa: E402
    DEFAULT_MAX_THIN_FILE_BYTES,
    find_denylisted_entries,
    find_tracked_entries,
    human_size,
    iter_files,
    sha256_file,
    should_keep_in_thin_mirror,
    source_root_allowed,
    utc_now,
)

CACHE_MANIFESTS = {
    "trajectories/tensor_cache_v1/tensor_cache_manifest.json": (
        "trajectories/tensor_cache_v1/tensor_cache_manifest.summary.json"
    ),
    "trajectories/event_token_cache_v1/event_token_cache_manifest.json": (
        "trajectories/event_token_cache_v1/event_token_cache_manifest.summary.json"
    ),
}
EVENT_RETENTION_RECEIPT = (
    "trajectories/event_token_cache_v1/event_token_cache_retention_receipt.json"
)


@dataclass(frozen=True)
class PrunePlan:
    source_root: Path
    receipt_path: Path
    archive_destination_prefix: str
    retained_class: str
    replayable: bool
    keep_files: tuple[str, ...]
    prune_files: tuple[str, ...]
    prune_bytes: int
    generated_summary_files: tuple[str, ...]
    denied_entries: tuple[str, ...]
    tracked_entries: tuple[str, ...]

    @property
    def blocked(self) -> bool:
        return bool(self.denied_entries or self.tracked_entries)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_root": str(self.source_root),
            "receipt_path": str(self.receipt_path),
            "archive_destination_prefix": self.archive_destination_prefix,
            "retained_class": self.retained_class,
            "replayable": self.replayable,
            "blocked": self.blocked,
            "blocked_denylisted_entries": list(self.denied_entries),
            "blocked_tracked_entries": list(self.tracked_entries),
            "thin_local_mirror": {
                "file_count": len(self.keep_files) + len(self.generated_summary_files),
                "sample": list((self.keep_files + self.generated_summary_files)[:30]),
            },
            "proposed_prune": {
                "file_count": len(self.prune_files),
                "size_bytes": self.prune_bytes,
                "size_human": human_size(self.prune_bytes),
                "sample": list(self.prune_files[:30]),
            },
            "generated_summary_files": list(self.generated_summary_files),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prune local QuantLab outputs only after a verified archive receipt exists. "
            "The default mode is dry-run; --execute is required before deletion."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Local output or remote run root to thin after archive verification.",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Verified archive receipt. Defaults to <source-root>/archive_receipt.json.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Delete prune candidates and write local_prune_receipt.json.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional report path. Not written unless provided.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_prune_plan(args.source_root, receipt_path=args.receipt, repo_root=REPO_ROOT)
    report: dict[str, Any] = {
        "mode": "execute" if args.execute else "dry_run",
        "plan": plan.as_dict(),
    }
    if not args.execute:
        emit_report(report, args.output_json)
        return 0
    if plan.blocked:
        report["execution_error"] = "refusing prune because source root is blocked"
        emit_report(report, args.output_json)
        return 2
    result = execute_prune_plan(plan)
    report["execution_result"] = result
    emit_report(report, args.output_json)
    return 0


def build_prune_plan(
    source_root: Path,
    *,
    receipt_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> PrunePlan:
    resolved = source_root.expanduser().resolve()
    if not source_root_allowed(resolved, repo_root=repo_root):
        raise ValueError(f"source root is outside prune allowlist: {resolved}")
    receipt = (receipt_path or (resolved / "archive_receipt.json")).expanduser().resolve()
    payload = load_verified_receipt(receipt, expected_source_root=resolved)
    denied_entries = tuple(find_denylisted_entries(resolved))
    tracked_entries = tuple(find_tracked_entries(resolved, repo_root=repo_root))
    files = iter_files(resolved)
    generated_summary_files = tuple(summary_paths_for_plan(resolved, files))
    keep_files = tuple(
        path.relative_to(resolved).as_posix()
        for path in files
        if should_keep_after_prune(path, resolved)
    )
    keep_set = set(keep_files)
    generated_set = set(generated_summary_files)
    prune_files = tuple(
        path.relative_to(resolved).as_posix()
        for path in files
        if path.relative_to(resolved).as_posix() not in keep_set
        and path.relative_to(resolved).as_posix() not in generated_set
    )
    prune_bytes = sum((resolved / path).stat().st_size for path in prune_files)
    retained_class = str(payload.get("retained_class") or payload.get("retained_class_after_prune") or "partial")
    return PrunePlan(
        source_root=resolved,
        receipt_path=receipt,
        archive_destination_prefix=str(payload["archive_destination_prefix"]),
        retained_class=retained_class,
        replayable=bool(payload.get("replayable", False)),
        keep_files=keep_files,
        prune_files=prune_files,
        prune_bytes=prune_bytes,
        generated_summary_files=generated_summary_files,
        denied_entries=denied_entries,
        tracked_entries=tracked_entries,
    )


def load_verified_receipt(receipt_path: Path, *, expected_source_root: Path) -> dict[str, Any]:
    if not receipt_path.exists():
        raise FileNotFoundError(f"archive receipt does not exist: {receipt_path}")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if payload.get("verification_status") != "verified" or not payload.get("verified_at"):
        raise ValueError("archive receipt is not verified; refusing prune")
    receipt_source = Path(str(payload.get("source_root", ""))).expanduser().resolve()
    if receipt_source != expected_source_root:
        raise ValueError(
            f"archive receipt source root mismatch: {receipt_source} != {expected_source_root}"
        )
    if not payload.get("archive_destination_prefix"):
        raise ValueError("archive receipt missing archive_destination_prefix")
    return payload


def should_keep_after_prune(path: Path, source_root: Path) -> bool:
    relative = path.relative_to(source_root).as_posix()
    if relative in CACHE_MANIFESTS:
        return False
    if path.name in {"policy.json", "inference_artifact.json"}:
        return path.stat().st_size <= DEFAULT_MAX_THIN_FILE_BYTES
    if relative.startswith("registry/artifacts/"):
        return path.stat().st_size <= DEFAULT_MAX_THIN_FILE_BYTES
    return should_keep_in_thin_mirror(path, source_root)


def summary_paths_for_plan(source_root: Path, files: list[Path]) -> list[str]:
    existing = {path.relative_to(source_root).as_posix() for path in files}
    summaries: list[str] = []
    for manifest_relative, summary_relative in CACHE_MANIFESTS.items():
        if manifest_relative in existing:
            summaries.append(summary_relative)
            if manifest_relative.startswith("trajectories/event_token_cache_v1/"):
                summaries.append(EVENT_RETENTION_RECEIPT)
    return summaries


def execute_prune_plan(plan: PrunePlan) -> dict[str, Any]:
    write_cache_summaries(plan)
    pruned: list[dict[str, Any]] = []
    for relative_path in plan.prune_files:
        path = plan.source_root / relative_path
        if not path.exists():
            continue
        pruned.append(
            {
                "path": relative_path,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        path.unlink()
    remove_empty_dirs(plan.source_root)
    receipt = {
        "receipt_version": "local_prune_receipt_v1",
        "source_root": str(plan.source_root),
        "archive_receipt_path": str(plan.receipt_path),
        "archive_destination_prefix": plan.archive_destination_prefix,
        "timestamp": utc_now(),
        "retained_class": "slim",
        "replayable": False,
        "what_was_kept_locally": sorted(
            path.relative_to(plan.source_root).as_posix()
            for path in iter_files(plan.source_root)
        ),
        "what_was_pruned_locally": pruned,
        "what_was_pruned_remotely": [],
    }
    prune_receipt_path = plan.source_root / "local_prune_receipt.json"
    prune_receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "source_root": str(plan.source_root),
        "pruned_file_count": len(pruned),
        "pruned_bytes": sum(item["size_bytes"] for item in pruned),
        "pruned_human": human_size(sum(item["size_bytes"] for item in pruned)),
        "local_prune_receipt": str(prune_receipt_path),
    }


def write_cache_summaries(plan: PrunePlan) -> None:
    for manifest_relative, summary_relative in CACHE_MANIFESTS.items():
        manifest_path = plan.source_root / manifest_relative
        if not manifest_path.exists():
            continue
        summary_path = plan.source_root / summary_relative
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary_version": "pruned_cache_manifest_summary_v1",
            "original_manifest_path": manifest_relative,
            "original_manifest_sha256": sha256_file(manifest_path),
            "original_manifest_size_bytes": manifest_path.stat().st_size,
            "created_at": utc_now(),
            "archive_receipt_path": str(plan.receipt_path),
            "archive_destination_prefix": plan.archive_destination_prefix,
            "local_payloads_pruned": True,
        }
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if manifest_relative.startswith("trajectories/event_token_cache_v1/"):
            receipt_path = plan.source_root / EVENT_RETENTION_RECEIPT
            receipt = {
                "receipt_version": "event_token_cache_local_prune_receipt_v1",
                "created_at": utc_now(),
                "archive_receipt_path": str(plan.receipt_path),
                "archive_destination_prefix": plan.archive_destination_prefix,
                "local_payloads_pruned": True,
            }
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def remove_empty_dirs(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def emit_report(report: dict[str, Any], output_json: Path | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    raise SystemExit(main())
