#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from quantlab_ml.registry.retention import write_bundle_sha256sums, write_retained_bundle_manifest  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build bundle_manifest.json and SHA256SUMS for a local retained remote-run bundle, "
            "optionally copying exact repo config files and attaching Vast instance metadata."
        )
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        required=True,
        help="Local retained bundle root that already contains the copied run outputs.",
    )
    parser.add_argument(
        "--source-run-root",
        required=True,
        help="Original remote run root, recorded as metadata only.",
    )
    parser.add_argument(
        "--source-registry-root",
        default=None,
        help="Optional original remote registry root. Defaults to <source-run-root>/registry.",
    )
    parser.add_argument(
        "--retained-bundle-kind",
        default="repo_local_retained_bundle",
        help="Retained bundle kind label written into bundle_manifest.json.",
    )
    parser.add_argument(
        "--retained-bundle-authority-note",
        default=(
            "derived from a controlled remote run; retained copy remains external-retained-evidence "
            "and is not relabeled authoritative evidence"
        ),
        help="Authority note written into bundle_manifest.json.",
    )
    parser.add_argument(
        "--source-repo-commit-sha",
        default=None,
        help="Optional explicit source repo commit SHA. Defaults to the local repo HEAD when available.",
    )
    parser.add_argument(
        "--instance-metadata",
        type=Path,
        default=None,
        help="Optional JSON/YAML file describing the exact Vast.ai instance used for the run.",
    )
    parser.add_argument(
        "--ql031-status-path",
        type=Path,
        default=None,
        help="Optional ql031_status.json path to embed QL-031 integration summary.",
    )
    parser.add_argument(
        "--config-copy",
        action="append",
        default=[],
        help=(
            "Copy an exact repo config into the bundle and record its SHA-256. "
            "Format: <repo_source_path>:<bundle_relative_path>. Repeat as needed."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_copies = [_parse_config_copy(value) for value in args.config_copy]
    manifest_path = write_retained_bundle_manifest(
        bundle_root=args.bundle_root,
        source_run_root=args.source_run_root,
        retained_bundle_kind=args.retained_bundle_kind,
        retained_bundle_authority_note=args.retained_bundle_authority_note,
        source_registry_root=args.source_registry_root,
        source_repo_commit_sha=args.source_repo_commit_sha,
        instance_metadata_path=args.instance_metadata,
        ql031_status_path=args.ql031_status_path,
        config_copies=config_copies,
    )
    sha256sums_path = write_bundle_sha256sums(args.bundle_root)
    print(f"wrote retained bundle manifest to {manifest_path}")
    print(f"wrote retained bundle checksums to {sha256sums_path}")
    return 0


def _parse_config_copy(value: str) -> tuple[Path, str]:
    if ":" not in value:
        raise ValueError("--config-copy must use <repo_source_path>:<bundle_relative_path>")
    source_path, bundle_relative_path = value.split(":", 1)
    if not source_path or not bundle_relative_path:
        raise ValueError("--config-copy must provide both source and bundle-relative paths")
    return Path(source_path), bundle_relative_path


if __name__ == "__main__":
    raise SystemExit(main())
