from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_micro_profile_module(repo_root: Path):
    script_path = repo_root / "scripts" / "validate_ql033_r6_micro_profile.py"
    spec = importlib.util.spec_from_file_location("validate_ql033_r6_micro_profile", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _profile() -> dict:
    timings = {
        "lane_range_extraction_wall_sec": 1.0,
        "raw_candidate_assembly_wall_sec": 1.0,
        "deterministic_ordering_wall_sec": 1.0,
        "dedupe_wall_sec": 1.0,
        "bbo_tuple_extraction_wall_sec": 1.0,
        "bbo_burst_significance_wall_sec": 1.0,
        "t4_resolution_wall_sec": 1.0,
        "quota_fill_wall_sec": 1.0,
    }
    return {
        "profile_version": "ql033_r6_window_base_micro_profile_v1",
        "same_rows": True,
        "coverage": {
            "rows_processed": 420,
            "first_pass_miss_rows": 359,
            "includes_first_pass_miss_only_region": True,
            "cache_reuse_transition_feasible": True,
            "includes_cache_reuse_transition": True,
            "last_decision_timestamp": "2026-01-25T21:59:00+00:00",
        },
        "modes": {
            "reference": {"subphase_timings": timings},
            "candidate": {"subphase_timings": timings},
        },
        "semantic_equivalence": {"ordered_reference_candidate_match": True},
        "projection": {"projected_build_time_multiplier": 2.0},
    }


def test_r6_micro_profile_validator_accepts_representative_profile(repo_root: Path) -> None:
    validator = _load_micro_profile_module(repo_root)

    report = validator.build_report(_profile())

    assert report["classification"] == "eligible_for_review"
    assert report["proceed_to_full_proof"] is False


def test_r6_micro_profile_validator_rejects_short_partial_profile(repo_root: Path) -> None:
    validator = _load_micro_profile_module(repo_root)
    profile = _profile()
    profile["coverage"]["rows_processed"] = 19
    profile["coverage"]["first_pass_miss_rows"] = 19
    profile["coverage"]["includes_cache_reuse_transition"] = False

    report = validator.build_report(profile)

    assert report["classification"] == "insufficient_evidence"
    assert "minimum_first_pass_miss_rows" in report["blocked_by"]
    assert "cache_reuse_transition" in report["blocked_by"]
