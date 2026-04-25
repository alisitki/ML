#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROFILE_VERSION = "ql033_r6_window_base_micro_profile_v1"
MINIMUM_FIRST_PASS_MISS_ROWS = 359
BUILD_GATE_MULTIPLIER = 2.25
REQUIRED_SUBPHASE_TIMINGS = (
    "lane_range_extraction_wall_sec",
    "raw_candidate_assembly_wall_sec",
    "deterministic_ordering_wall_sec",
    "dedupe_wall_sec",
    "bbo_tuple_extraction_wall_sec",
    "bbo_burst_significance_wall_sec",
    "t4_resolution_wall_sec",
    "quota_fill_wall_sec",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that a QL-033 R6 micro-profile is representative enough for review."
    )
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--minimum-miss-rows", type=int, default=MINIMUM_FIRST_PASS_MISS_ROWS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = json.loads(args.profile_json.read_text(encoding="utf-8"))
    report = build_report(profile, minimum_miss_rows=args.minimum_miss_rows)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["classification"] == "eligible_for_review" else 2


def build_report(profile: dict[str, Any], *, minimum_miss_rows: int = MINIMUM_FIRST_PASS_MISS_ROWS) -> dict[str, Any]:
    coverage = profile.get("coverage") or {}
    modes = profile.get("modes") or {}
    semantic_equivalence = profile.get("semantic_equivalence") or {}
    projection = profile.get("projection") or {}
    checks = {
        "profile_version": _check(profile.get("profile_version") == PROFILE_VERSION, profile.get("profile_version")),
        "same_rows": _check(bool(profile.get("same_rows")), profile.get("same_rows")),
        "reference_mode_present": _check("reference" in modes, sorted(modes)),
        "candidate_mode_present": _check("candidate" in modes, sorted(modes)),
        "minimum_first_pass_miss_rows": _check(
            int(coverage.get("first_pass_miss_rows") or 0) >= minimum_miss_rows,
            coverage.get("first_pass_miss_rows"),
            threshold=f">={minimum_miss_rows}",
        ),
        "first_pass_miss_region_included": _check(
            bool(coverage.get("includes_first_pass_miss_only_region")),
            coverage.get("includes_first_pass_miss_only_region"),
        ),
        "cache_reuse_transition": _cache_transition_check(coverage),
        "last_decision_timestamp": _check(bool(coverage.get("last_decision_timestamp")), coverage.get("last_decision_timestamp")),
        "ordered_semantic_equivalence": _check(
            bool(semantic_equivalence.get("ordered_reference_candidate_match")),
            semantic_equivalence.get("ordered_reference_candidate_match"),
        ),
        "projected_gate_multiplier": _check(
            projection.get("projected_build_time_multiplier") is not None
            and float(projection["projected_build_time_multiplier"]) <= BUILD_GATE_MULTIPLIER,
            projection.get("projected_build_time_multiplier"),
            threshold=f"<={BUILD_GATE_MULTIPLIER}",
        ),
    }
    for mode_name in ("reference", "candidate"):
        timings = (modes.get(mode_name) or {}).get("subphase_timings") or {}
        for timing_name in REQUIRED_SUBPHASE_TIMINGS:
            checks[f"{mode_name}_{timing_name}"] = _check(
                timings.get(timing_name) is not None,
                timings.get(timing_name),
            )
    blocked_by = [name for name, check in checks.items() if not check["passed"]]
    return {
        "report_version": "ql033_r6_micro_profile_validation_v1",
        "generated_at": utc_now(),
        "classification": "eligible_for_review" if not blocked_by else "insufficient_evidence",
        "proceed_to_full_proof": False,
        "blocked_by": blocked_by,
        "checks": checks,
        "note": "A green micro-profile only allows operator review; it never proves R6 PASS.",
    }


def _cache_transition_check(coverage: dict[str, Any]) -> dict[str, Any]:
    feasible = bool(coverage.get("cache_reuse_transition_feasible", True))
    included = bool(coverage.get("includes_cache_reuse_transition"))
    if not feasible:
        return _check(
            bool(coverage.get("cache_reuse_transition_justification")),
            coverage.get("cache_reuse_transition_justification"),
            threshold="explicit justification when transition is infeasible",
        )
    return _check(included, included)


def _check(passed: bool, value: Any, threshold: Any = None) -> dict[str, Any]:
    return {"passed": bool(passed), "value": value, "threshold": threshold}


def utc_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
