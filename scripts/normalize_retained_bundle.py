#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from quantlab_ml.registry.bundle_integrity import normalize_retained_bundle, validate_retained_bundle  # noqa: E402
from quantlab_ml.registry.retention import refresh_existing_retained_bundle_manifest, write_bundle_sha256sums  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize a retained bundle into an explicitly classified replayable/full or "
            "non-replayable/slim form while preserving provenance via a normalization receipt."
        )
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        required=True,
        help="Existing retained bundle root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional normalized sibling output root. Omit to use <bundle-root>-normalized.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Normalize in place. Default behavior is sibling-copy normalization.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    normalized_root, receipt = normalize_retained_bundle(
        bundle_root=args.bundle_root,
        output_root=args.output_root,
        in_place=args.in_place,
    )
    refresh_existing_retained_bundle_manifest(
        normalized_root,
        extra_updates={
            "normalization_receipt_path": "normalization_receipt.json",
        },
    )
    write_bundle_sha256sums(normalized_root)
    report = validate_retained_bundle(normalized_root)
    print(f"normalized retained bundle to {normalized_root}")
    print(
        "wrote normalization receipt "
        f"mode={receipt.normalization_mode} "
        f"removed_dangling_files={len(receipt.removed_dangling_files)} "
        f"replacement_summary_artifacts={len(receipt.replacement_summary_artifacts)}"
    )
    if report is not None:
        print(
            "validated normalized bundle "
            f"bundle_payload_class={report.bundle_payload_class} "
            f"replayable={str(report.replayable).lower()} "
            f"supports_phase0_empirical_closure={str(report.supports_phase0_empirical_closure).lower()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
