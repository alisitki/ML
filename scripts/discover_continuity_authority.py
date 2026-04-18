#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from quantlab_ml.registry.authority_discovery import discover_continuity_authority  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover continuity authority candidates without granting authority or "
            "running an authoritative continuity rerun."
        )
    )
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="Optional search root to inspect before the default /workspace/runs and /root/runs locations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. If omitted, the summary is written to stdout only.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = discover_continuity_authority(search_roots=args.search_root)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
