#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROVENANCE_VERSION = "remote_proof_provenance_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record and validate current-code provenance before a remote proof run. "
            "A current local/remote HEAD mismatch fails unless an explicit override is recorded."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-local-head", required=True)
    parser.add_argument("--remote-head", default=None)
    parser.add_argument("--baseline-import-proof-commit", required=True)
    parser.add_argument("--runner-script", type=Path, required=True)
    parser.add_argument("--validator-script", type=Path, required=True)
    parser.add_argument("--allow-head-mismatch", action="store_true")
    parser.add_argument("--override-reason", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_provenance_report(
        repo_root=args.repo_root,
        expected_local_head=args.expected_local_head,
        remote_head=args.remote_head,
        baseline_import_proof_commit=args.baseline_import_proof_commit,
        runner_script=args.runner_script,
        validator_script=args.validator_script,
        allow_head_mismatch=args.allow_head_mismatch,
        override_reason=args.override_reason,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


def build_provenance_report(
    *,
    repo_root: Path,
    expected_local_head: str,
    remote_head: str | None,
    baseline_import_proof_commit: str,
    runner_script: Path,
    validator_script: Path,
    allow_head_mismatch: bool = False,
    override_reason: str | None = None,
) -> dict[str, Any]:
    resolved_repo = repo_root.expanduser().resolve()
    current_remote_head = remote_head or _git_output(resolved_repo, ["rev-parse", "HEAD"])
    git_status_short = _git_output(resolved_repo, ["status", "--short"], allow_failure=True)
    dirty_marker = "unknown" if git_status_short is None else ("dirty" if git_status_short.strip() else "clean")
    runner_record = _script_record(runner_script)
    validator_record = _script_record(validator_script)

    mismatch = expected_local_head != current_remote_head
    override_recorded = bool(allow_head_mismatch and override_reason and override_reason.strip())
    blocking_reasons: list[str] = []
    if mismatch and not override_recorded:
        blocking_reasons.append("current_local_remote_head_mismatch")
    if not runner_record["exists"]:
        blocking_reasons.append("missing_runner_script")
    if not validator_record["exists"]:
        blocking_reasons.append("missing_validator_script")
    if not baseline_import_proof_commit.strip():
        blocking_reasons.append("missing_baseline_import_proof_commit")

    return {
        "provenance_version": PROVENANCE_VERSION,
        "generated_at": utc_now(),
        "status": "fail" if blocking_reasons else "pass",
        "blocking_reasons": blocking_reasons,
        "current_local_head_expected": expected_local_head,
        "current_remote_head_actual": current_remote_head,
        "current_head_match": not mismatch,
        "baseline_import_proof_commit": baseline_import_proof_commit,
        "baseline_import_proof_is_current_code_provenance": False,
        "dirty_marker": dirty_marker,
        "git_status_short": git_status_short,
        "runner_script": runner_record,
        "validator_script": validator_record,
        "override": {
            "head_mismatch_allowed": allow_head_mismatch,
            "override_recorded": override_recorded,
            "override_reason": override_reason,
        },
        "failure_behavior": (
            "fail_before_expensive_run unless current local and remote HEAD match "
            "or an explicit override reason is recorded"
        ),
    }


def _script_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        return {
            "path": str(resolved),
            "exists": False,
            "sha256": None,
            "size_bytes": None,
        }
    return {
        "path": str(resolved),
        "exists": True,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _git_output(repo_root: Path, args: list[str], *, allow_failure: bool = False) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        if allow_failure:
            return None
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "git command failed")
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
