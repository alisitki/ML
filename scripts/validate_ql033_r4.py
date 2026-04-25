from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


THRESHOLDS = {
    "build_time_multiplier": 2.25,
    "artifact_size_multiplier": 1.15,
    "truncation_rate": 0.40,
    "weighted_target_symbol_retained_rate": 0.70,
    "weighted_burst_retention_rate": 0.70,
    "cross_venue_ordered_adjacency_rate": 0.80,
    "trade_to_bbo_ordered_adjacency_rate": 0.80,
    "per_symbol_starvation_rate": 0.05,
    "symbol_with_zero_retained_tokens_count_p95": 0.0,
}

NONDETERMINISTIC_HASH_EXCLUDES = [
    "event_token_cache_diagnostics.json",
    "*_partial_selector_profile.json",
]

PASS = "PASS"
FAIL = "FAIL"
UNKNOWN_BLOCKING = "UNKNOWN_BLOCKING"

EVENT_TOKEN_SHARD_PAYLOAD_FIELDS = [
    "row_offsets_path",
    "event_time_path",
    "lag_ms_path",
    "exchange_id_path",
    "symbol_id_path",
    "stream_id_path",
    "source_label_id_path",
    "source_event_index_path",
    "payload_schema_id_path",
    "payload_row_index_path",
    "replay_path",
    "window_stats_path",
    "trade_payload_values_path",
    "trade_payload_presence_path",
    "bbo_payload_values_path",
    "bbo_payload_presence_path",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _directory_size(path: Path) -> dict[str, int]:
    total = 0
    count = 0
    if not path.exists():
        return {"bytes": 0, "file_count": 0}
    for root, _, files in os.walk(path):
        for filename in files:
            file_path = Path(root) / filename
            total += file_path.stat().st_size
            count += 1
    return {"bytes": total, "file_count": count}


def _event_payload_status(run_root: Path, run_name: str) -> dict[str, Any]:
    trajectories = run_root / run_name / "trajectories"
    manifest_path = trajectories / "event_token_cache_v1" / "event_token_cache_manifest.json"
    if not manifest_path.exists():
        return {
            "manifest_present": False,
            "shard_count": 0,
            "missing_payload_count": 0,
            "missing_payloads": [],
            "payload_complete": False,
        }
    manifest = _read_json(manifest_path)
    shard_count = 0
    missing_payloads: list[str] = []
    for split in (manifest.get("splits") or {}).values():
        for shard in split.get("shards") or []:
            shard_count += 1
            for field_name in EVENT_TOKEN_SHARD_PAYLOAD_FIELDS:
                relative_path = shard.get(field_name)
                if not relative_path or not (trajectories / relative_path).exists():
                    missing_payloads.append(str(relative_path or field_name))
    return {
        "manifest_present": True,
        "shard_count": shard_count,
        "missing_payload_count": len(missing_payloads),
        "missing_payloads": sorted(set(missing_payloads)),
        "payload_complete": shard_count > 0 and not missing_payloads,
    }


def _metric_values(analyses: list[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for analysis in analyses:
        for split in (analysis.get("splits") or {}).values():
            diagnostics = split.get("diagnostics") or {}
            value = diagnostics.get(metric)
            if value is not None:
                values.append(float(value))
    return values


def _worst_high(analyses: list[dict[str, Any]], metric: str) -> float | None:
    values = _metric_values(analyses, metric)
    return max(values) if values else None


def _worst_low(analyses: list[dict[str, Any]], metric: str) -> float | None:
    values = _metric_values(analyses, metric)
    return min(values) if values else None


def _max_per_symbol_starvation(analyses: list[dict[str, Any]]) -> float | None:
    max_value: float | None = None
    for analysis in analyses:
        for split in (analysis.get("splits") or {}).values():
            diagnostics = split.get("diagnostics") or {}
            rates = diagnostics.get("per_symbol_starvation_rate") or {}
            for value in rates.values():
                max_value = float(value) if max_value is None else max(max_value, float(value))
    return max_value


def _analysis_evaluable_row_count(analysis: dict[str, Any]) -> int:
    row_count = 0
    for split in (analysis.get("splits") or {}).values():
        diagnostics = split.get("diagnostics") or {}
        candidates = (
            split.get("event_token_rows"),
            split.get("trajectory_rows"),
            split.get("tensor_cache_rows"),
            split.get("row_count"),
            diagnostics.get("row_count"),
        )
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                row_count += max(int(candidate), 0)
            except (TypeError, ValueError):
                pass
            break
    return row_count


def _analysis_status(path: Path, analysis: dict[str, Any] | None) -> dict[str, Any]:
    if analysis is None:
        return {
            "present": False,
            "evaluable": False,
            "evaluable_row_count": 0,
            "reason": "missing_analysis_json",
            "path": str(path),
        }
    splits = analysis.get("splits") or {}
    row_count = _analysis_evaluable_row_count(analysis)
    if not splits:
        return {
            "present": True,
            "evaluable": False,
            "evaluable_row_count": 0,
            "reason": "empty_analysis_splits",
            "path": str(path),
        }
    if row_count <= 0:
        return {
            "present": True,
            "evaluable": False,
            "evaluable_row_count": row_count,
            "reason": "no_evaluable_rows",
            "path": str(path),
        }
    return {
        "present": True,
        "evaluable": True,
        "evaluable_row_count": row_count,
        "reason": None,
        "path": str(path),
    }


def _all_analyses_evaluable(analysis_statuses: dict[str, dict[str, Any]]) -> bool:
    return bool(analysis_statuses) and all(
        bool(status["evaluable"]) for status in analysis_statuses.values()
    )


def _check(
    name: str,
    passed: bool,
    *,
    value: Any = None,
    threshold: Any = None,
    status: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    resolved_status = status or (PASS if passed else FAIL)
    return {
        "passed": bool(passed) and resolved_status == PASS,
        "status": resolved_status,
        "value": value,
        "threshold": threshold,
        "reason": reason,
    }


def _unknown_check(name: str, *, value: Any = None, threshold: Any = None, reason: str) -> dict[str, Any]:
    return _check(
        name,
        False,
        value=value,
        threshold=threshold,
        status=UNKNOWN_BLOCKING,
        reason=reason,
    )


def _metric_check(
    name: str,
    *,
    analyses_ready: bool,
    value: Any,
    threshold: Any,
    predicate: bool,
) -> dict[str, Any]:
    if not analyses_ready:
        return _unknown_check(
            name,
            value=value,
            threshold=threshold,
            reason="missing_or_empty_analysis",
        )
    if value is None:
        return _unknown_check(
            name,
            value=value,
            threshold=threshold,
            reason="missing_metric_or_denominator",
        )
    return _check(name, predicate, value=value, threshold=threshold)


def _archive_verified(run_root: Path, archive_prefix: str | None) -> tuple[bool, dict[str, Any] | None]:
    receipt = _maybe_json(run_root / "archive_receipt.json")
    if receipt is None:
        return False, None
    verified = receipt.get("verification_status") == "verified" and bool(receipt.get("verified_at"))
    if archive_prefix is not None:
        verified = verified and receipt.get("archive_destination_prefix") == archive_prefix
    return verified, receipt


def build_report(
    *,
    run_root: Path,
    require_archive: bool = False,
    archive_prefix: str | None = None,
) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    baseline_time = _maybe_json(run_root / "baseline_head" / "build.time.json") or {}
    run_a_time = _maybe_json(run_root / "run_a" / "build.time.json") or {}
    run_b_time = _maybe_json(run_root / "run_b" / "build.time.json") or {}
    run_a_analysis_path = run_root / "run_a_event_token_analysis.json"
    run_b_analysis_path = run_root / "run_b_event_token_analysis.json"
    run_a_analysis = _maybe_json(run_a_analysis_path)
    run_b_analysis = _maybe_json(run_b_analysis_path)
    run_a_hashes = _maybe_json(run_root / "run_a_hashes.json") or {}
    run_b_hashes = _maybe_json(run_root / "run_b_hashes.json") or {}
    run_a_event_payload_status = _event_payload_status(run_root, "run_a")
    run_b_event_payload_status = _event_payload_status(run_root, "run_b")
    analysis_statuses = {
        "run_a": _analysis_status(run_a_analysis_path, run_a_analysis),
        "run_b": _analysis_status(run_b_analysis_path, run_b_analysis),
    }
    analyses_ready = _all_analyses_evaluable(analysis_statuses)
    analyses = [
        analysis
        for run_name, analysis in (("run_a", run_a_analysis), ("run_b", run_b_analysis))
        if analysis is not None and analysis_statuses[run_name]["evaluable"]
    ]

    baseline_real = float(baseline_time.get("real_seconds") or 0.0)
    run_a_real = float(run_a_time.get("real_seconds") or 0.0)
    run_b_real = float(run_b_time.get("real_seconds") or 0.0)
    max_run_real = max(run_a_real, run_b_real)
    build_time_multiplier = (max_run_real / baseline_real) if baseline_real else None

    baseline_size = _directory_size(run_root / "baseline_head" / "trajectories")
    run_a_size = _directory_size(run_root / "run_a" / "trajectories")
    run_b_size = _directory_size(run_root / "run_b" / "trajectories")
    max_run_bytes = max(run_a_size["bytes"], run_b_size["bytes"])
    artifact_size_multiplier = (
        max_run_bytes / baseline_size["bytes"] if baseline_size["bytes"] else None
    )

    hashes_match = (
        run_a_hashes.get("status") == "ok"
        and run_b_hashes.get("status") == "ok"
        and run_a_hashes.get("tree_sha256") == run_b_hashes.get("tree_sha256")
        and run_a_hashes.get("file_count") == run_b_hashes.get("file_count")
    )
    archive_ok, archive_receipt = _archive_verified(run_root, archive_prefix)
    run_a_complete = bool(run_a_event_payload_status["payload_complete"])
    run_b_complete = bool(run_b_event_payload_status["payload_complete"])
    both_runs_complete = run_a_complete and run_b_complete
    run_b_not_started = (run_root / "run_b" / "not_started.exit").exists()
    run_b_started = (run_root / "run_b" / "build.time.json").exists() and not run_b_not_started

    metrics = {
        "build_time_multiplier": build_time_multiplier,
        "artifact_size_multiplier": artifact_size_multiplier,
        "max_truncation_rate": _worst_high(analyses, "truncation_rate"),
        "min_weighted_target_symbol_retained_rate": _worst_low(
            analyses,
            "weighted_target_symbol_retained_rate",
        ),
        "min_weighted_burst_retention_rate": _worst_low(
            analyses,
            "weighted_burst_retention_rate",
        ),
        "min_cross_venue_ordered_adjacency_rate": _worst_low(
            analyses,
            "cross_venue_ordered_adjacency_rate",
        ),
        "min_trade_to_bbo_ordered_adjacency_rate": _worst_low(
            analyses,
            "trade_to_bbo_ordered_adjacency_rate",
        ),
        "max_per_symbol_starvation_rate": _max_per_symbol_starvation(analyses),
        "max_symbol_with_zero_retained_tokens_count_p95": _worst_high(
            analyses,
            "symbol_with_zero_retained_tokens_count_p95",
        ),
    }

    checks = {
        "run_a_analysis_evaluable": _check(
            "run_a_analysis_evaluable",
            bool(analysis_statuses["run_a"]["evaluable"]),
            value=analysis_statuses["run_a"],
            status=PASS if analysis_statuses["run_a"]["evaluable"] else UNKNOWN_BLOCKING,
            reason=analysis_statuses["run_a"]["reason"],
        ),
        "run_b_analysis_evaluable": _check(
            "run_b_analysis_evaluable",
            bool(analysis_statuses["run_b"]["evaluable"]),
            value=analysis_statuses["run_b"],
            status=PASS if analysis_statuses["run_b"]["evaluable"] else UNKNOWN_BLOCKING,
            reason=analysis_statuses["run_b"]["reason"],
        ),
        "run_a_complete": _check(
            "run_a_complete",
            run_a_complete,
            value=run_a_event_payload_status,
        ),
        "run_b_complete": _check(
            "run_b_complete",
            run_b_complete,
            value=run_b_event_payload_status,
        ),
        "run_b_started": _check(
            "run_b_started",
            run_b_started,
            value={
                "build_time_present": (run_root / "run_b" / "build.time.json").exists(),
                "not_started_exit_present": run_b_not_started,
            },
        ),
        "row_alignment": _check(
            "row_alignment",
            analyses_ready
            and bool(run_a_analysis and run_a_analysis.get("row_alignment_ok"))
            and bool(run_b_analysis and run_b_analysis.get("row_alignment_ok")),
            value={
                "run_a_row_alignment_ok": (
                    run_a_analysis.get("row_alignment_ok") if run_a_analysis is not None else None
                ),
                "run_b_row_alignment_ok": (
                    run_b_analysis.get("row_alignment_ok") if run_b_analysis is not None else None
                ),
            },
            status=UNKNOWN_BLOCKING if not analyses_ready else None,
            reason=None if analyses_ready else "missing_or_empty_analysis",
        ),
        "determinism": _check(
            "determinism",
            hashes_match,
            value={
                "run_a_tree_sha256": run_a_hashes.get("tree_sha256"),
                "run_b_tree_sha256": run_b_hashes.get("tree_sha256"),
                "excluded_nondeterministic_files": NONDETERMINISTIC_HASH_EXCLUDES,
            },
        ),
        "build_time_multiplier": _check(
            "build_time_multiplier",
            both_runs_complete
            and build_time_multiplier is not None
            and build_time_multiplier <= THRESHOLDS["build_time_multiplier"],
            value=build_time_multiplier,
            threshold=f"<={THRESHOLDS['build_time_multiplier']}",
            status=UNKNOWN_BLOCKING if not both_runs_complete else None,
            reason=None if both_runs_complete else "incomplete_run_artifacts",
        ),
        "artifact_size_multiplier": _check(
            "artifact_size_multiplier",
            both_runs_complete
            and artifact_size_multiplier is not None
            and artifact_size_multiplier <= THRESHOLDS["artifact_size_multiplier"],
            value=artifact_size_multiplier,
            threshold=f"<={THRESHOLDS['artifact_size_multiplier']}",
            status=UNKNOWN_BLOCKING if not both_runs_complete else None,
            reason=None if both_runs_complete else "incomplete_run_artifacts",
        ),
        "truncation_rate": _metric_check(
            "truncation_rate",
            analyses_ready=analyses_ready,
            value=metrics["max_truncation_rate"],
            threshold=f"<={THRESHOLDS['truncation_rate']}",
            predicate=(
                metrics["max_truncation_rate"] is not None
                and metrics["max_truncation_rate"] <= THRESHOLDS["truncation_rate"]
            ),
        ),
        "weighted_target_symbol_retained_rate": _metric_check(
            "weighted_target_symbol_retained_rate",
            analyses_ready=analyses_ready,
            value=metrics["min_weighted_target_symbol_retained_rate"],
            threshold=f">={THRESHOLDS['weighted_target_symbol_retained_rate']}",
            predicate=(
                metrics["min_weighted_target_symbol_retained_rate"] is not None
                and metrics["min_weighted_target_symbol_retained_rate"]
                >= THRESHOLDS["weighted_target_symbol_retained_rate"]
            ),
        ),
        "weighted_burst_retention_rate": _metric_check(
            "weighted_burst_retention_rate",
            analyses_ready=analyses_ready,
            value=metrics["min_weighted_burst_retention_rate"],
            threshold=f">={THRESHOLDS['weighted_burst_retention_rate']}",
            predicate=(
                metrics["min_weighted_burst_retention_rate"] is not None
                and metrics["min_weighted_burst_retention_rate"]
                >= THRESHOLDS["weighted_burst_retention_rate"]
            ),
        ),
        "cross_venue_ordered_adjacency_rate": _metric_check(
            "cross_venue_ordered_adjacency_rate",
            analyses_ready=analyses_ready,
            value=metrics["min_cross_venue_ordered_adjacency_rate"],
            threshold=f">={THRESHOLDS['cross_venue_ordered_adjacency_rate']}",
            predicate=(
                metrics["min_cross_venue_ordered_adjacency_rate"] is not None
                and metrics["min_cross_venue_ordered_adjacency_rate"]
                >= THRESHOLDS["cross_venue_ordered_adjacency_rate"]
            ),
        ),
        "trade_to_bbo_ordered_adjacency_rate": _metric_check(
            "trade_to_bbo_ordered_adjacency_rate",
            analyses_ready=analyses_ready,
            value=metrics["min_trade_to_bbo_ordered_adjacency_rate"],
            threshold=f">={THRESHOLDS['trade_to_bbo_ordered_adjacency_rate']}",
            predicate=(
                metrics["min_trade_to_bbo_ordered_adjacency_rate"] is not None
                and metrics["min_trade_to_bbo_ordered_adjacency_rate"]
                >= THRESHOLDS["trade_to_bbo_ordered_adjacency_rate"]
            ),
        ),
        "per_symbol_starvation_rate": _metric_check(
            "per_symbol_starvation_rate",
            analyses_ready=analyses_ready,
            value=metrics["max_per_symbol_starvation_rate"],
            threshold=f"<={THRESHOLDS['per_symbol_starvation_rate']}",
            predicate=(
                metrics["max_per_symbol_starvation_rate"] is not None
                and metrics["max_per_symbol_starvation_rate"] <= THRESHOLDS["per_symbol_starvation_rate"]
            ),
        ),
        "symbol_with_zero_retained_tokens_count_p95": _metric_check(
            "symbol_with_zero_retained_tokens_count_p95",
            analyses_ready=analyses_ready,
            value=metrics["max_symbol_with_zero_retained_tokens_count_p95"],
            threshold=THRESHOLDS["symbol_with_zero_retained_tokens_count_p95"],
            predicate=(
                metrics["max_symbol_with_zero_retained_tokens_count_p95"]
                == THRESHOLDS["symbol_with_zero_retained_tokens_count_p95"]
            ),
        ),
    }

    if require_archive:
        checks["archive_verified"] = _check(
            "archive_verified",
            archive_ok,
            value=archive_receipt.get("archive_destination_prefix") if archive_receipt else None,
            threshold=archive_prefix or "verified archive receipt",
        )

    blocked_by = [name for name, result in checks.items() if not result["passed"]]
    unknown_blocking = [
        name for name, result in checks.items() if result.get("status") == UNKNOWN_BLOCKING
    ]
    passed = not blocked_by
    return {
        "generated_at": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "proof_root": str(run_root),
        "scope": {
            "ql_item": "QL-033",
            "phase": "Phase A/B proof work only",
            "phase_c1_started": False,
            "model_or_encoder_work_started": False,
        },
        "thresholds": THRESHOLDS,
        "timing": {
            "baseline_build_time": baseline_time,
            "run_a_build_time": run_a_time,
            "run_b_build_time": run_b_time,
        },
        "sizes": {
            "baseline": baseline_size,
            "run_a": run_a_size,
            "run_b": run_b_size,
        },
        "metrics": metrics,
        "checks": checks,
        "analysis_statuses": analysis_statuses,
        "archive_receipt": archive_receipt,
        "executive_verdict": {
            "classification": "pass" if passed else "fail",
            "proceed": passed,
            "blocked_by": blocked_by,
            "unknown_blocking": unknown_blocking,
            "phase_c1_status": "closed_until_r4_passes" if not passed else "eligible_for_separate_review",
            "failure_handling": "archive partial roots before prune; classify next blocker from validator/profiling evidence",
        },
    }


def write_report_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report_markdown(path: Path, report: dict[str, Any]) -> None:
    verdict = report["executive_verdict"]
    lines = [
        "# QL-033 R4 Validation",
        "",
        f"- proof root: `{report['proof_root']}`",
        f"- verdict: `{verdict['classification']}`",
        f"- proceed: `{verdict['proceed']}`",
        f"- Phase C1 status: `{verdict['phase_c1_status']}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Passed | Value | Threshold | Reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, result in report["checks"].items():
        lines.append(
            f"| `{name}` | `{result.get('status')}` | `{result['passed']}` | "
            f"`{result.get('value')}` | `{result.get('threshold')}` | `{result.get('reason')}` |"
        )
    if verdict.get("unknown_blocking"):
        lines.extend(["", "## Unknown Blocking", ""])
        lines.extend(f"- `{name}`" for name in verdict["unknown_blocking"])
    if verdict["blocked_by"]:
        lines.extend(["", "## Blocked By", ""])
        lines.extend(f"- `{name}`" for name in verdict["blocked_by"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate QL-033 R4 proof-slice acceptance.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-md", type=Path)
    parser.add_argument("--archive-prefix")
    parser.add_argument("--require-archive", action="store_true")
    args = parser.parse_args()

    report = build_report(
        run_root=args.run_root,
        require_archive=args.require_archive,
        archive_prefix=args.archive_prefix,
    )
    if args.report_json is not None:
        write_report_json(args.report_json, report)
    if args.report_md is not None:
        write_report_markdown(args.report_md, report)
    if args.report_json is None and args.report_md is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["executive_verdict"]["proceed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
