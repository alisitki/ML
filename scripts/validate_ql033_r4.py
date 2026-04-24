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


def _max_per_symbol_starvation(analyses: list[dict[str, Any]]) -> float:
    max_value = 0.0
    for analysis in analyses:
        for split in (analysis.get("splits") or {}).values():
            diagnostics = split.get("diagnostics") or {}
            rates = diagnostics.get("per_symbol_starvation_rate") or {}
            for value in rates.values():
                max_value = max(max_value, float(value))
    return max_value


def _check(name: str, passed: bool, *, value: Any = None, threshold: Any = None) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "value": value,
        "threshold": threshold,
    }


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
    run_a_analysis = _maybe_json(run_root / "run_a_event_token_analysis.json") or {}
    run_b_analysis = _maybe_json(run_root / "run_b_event_token_analysis.json") or {}
    run_a_hashes = _maybe_json(run_root / "run_a_hashes.json") or {}
    run_b_hashes = _maybe_json(run_root / "run_b_hashes.json") or {}
    run_a_event_payload_status = _event_payload_status(run_root, "run_a")
    run_b_event_payload_status = _event_payload_status(run_root, "run_b")
    analyses = [analysis for analysis in (run_a_analysis, run_b_analysis) if analysis]

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
        "run_a_complete": _check(
            "run_a_complete",
            bool(run_a_event_payload_status["payload_complete"]),
            value=run_a_event_payload_status,
        ),
        "run_b_complete": _check(
            "run_b_complete",
            bool(run_b_event_payload_status["payload_complete"]),
            value=run_b_event_payload_status,
        ),
        "row_alignment": _check(
            "row_alignment",
            bool(run_a_analysis.get("row_alignment_ok")) and bool(run_b_analysis.get("row_alignment_ok")),
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
            build_time_multiplier is not None
            and build_time_multiplier <= THRESHOLDS["build_time_multiplier"],
            value=build_time_multiplier,
            threshold=f"<={THRESHOLDS['build_time_multiplier']}",
        ),
        "artifact_size_multiplier": _check(
            "artifact_size_multiplier",
            artifact_size_multiplier is not None
            and artifact_size_multiplier <= THRESHOLDS["artifact_size_multiplier"],
            value=artifact_size_multiplier,
            threshold=f"<={THRESHOLDS['artifact_size_multiplier']}",
        ),
        "truncation_rate": _check(
            "truncation_rate",
            metrics["max_truncation_rate"] is not None
            and metrics["max_truncation_rate"] <= THRESHOLDS["truncation_rate"],
            value=metrics["max_truncation_rate"],
            threshold=f"<={THRESHOLDS['truncation_rate']}",
        ),
        "weighted_target_symbol_retained_rate": _check(
            "weighted_target_symbol_retained_rate",
            metrics["min_weighted_target_symbol_retained_rate"] is not None
            and metrics["min_weighted_target_symbol_retained_rate"]
            >= THRESHOLDS["weighted_target_symbol_retained_rate"],
            value=metrics["min_weighted_target_symbol_retained_rate"],
            threshold=f">={THRESHOLDS['weighted_target_symbol_retained_rate']}",
        ),
        "weighted_burst_retention_rate": _check(
            "weighted_burst_retention_rate",
            metrics["min_weighted_burst_retention_rate"] is not None
            and metrics["min_weighted_burst_retention_rate"]
            >= THRESHOLDS["weighted_burst_retention_rate"],
            value=metrics["min_weighted_burst_retention_rate"],
            threshold=f">={THRESHOLDS['weighted_burst_retention_rate']}",
        ),
        "cross_venue_ordered_adjacency_rate": _check(
            "cross_venue_ordered_adjacency_rate",
            metrics["min_cross_venue_ordered_adjacency_rate"] is not None
            and metrics["min_cross_venue_ordered_adjacency_rate"]
            >= THRESHOLDS["cross_venue_ordered_adjacency_rate"],
            value=metrics["min_cross_venue_ordered_adjacency_rate"],
            threshold=f">={THRESHOLDS['cross_venue_ordered_adjacency_rate']}",
        ),
        "trade_to_bbo_ordered_adjacency_rate": _check(
            "trade_to_bbo_ordered_adjacency_rate",
            metrics["min_trade_to_bbo_ordered_adjacency_rate"] is not None
            and metrics["min_trade_to_bbo_ordered_adjacency_rate"]
            >= THRESHOLDS["trade_to_bbo_ordered_adjacency_rate"],
            value=metrics["min_trade_to_bbo_ordered_adjacency_rate"],
            threshold=f">={THRESHOLDS['trade_to_bbo_ordered_adjacency_rate']}",
        ),
        "per_symbol_starvation_rate": _check(
            "per_symbol_starvation_rate",
            metrics["max_per_symbol_starvation_rate"] <= THRESHOLDS["per_symbol_starvation_rate"],
            value=metrics["max_per_symbol_starvation_rate"],
            threshold=f"<={THRESHOLDS['per_symbol_starvation_rate']}",
        ),
        "symbol_with_zero_retained_tokens_count_p95": _check(
            "symbol_with_zero_retained_tokens_count_p95",
            metrics["max_symbol_with_zero_retained_tokens_count_p95"] == THRESHOLDS[
                "symbol_with_zero_retained_tokens_count_p95"
            ],
            value=metrics["max_symbol_with_zero_retained_tokens_count_p95"],
            threshold=THRESHOLDS["symbol_with_zero_retained_tokens_count_p95"],
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
        "archive_receipt": archive_receipt,
        "executive_verdict": {
            "classification": "pass" if passed else "fail",
            "proceed": passed,
            "blocked_by": blocked_by,
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
        "| Check | Passed | Value | Threshold |",
        "| --- | --- | --- | --- |",
    ]
    for name, result in report["checks"].items():
        lines.append(
            f"| `{name}` | `{result['passed']}` | `{result.get('value')}` | `{result.get('threshold')}` |"
        )
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
