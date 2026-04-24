from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_validator_module(repo_root: Path):
    script_path = repo_root / "scripts" / "validate_ql033_r4.py"
    spec = importlib.util.spec_from_file_location("validate_ql033_r4", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


_SHARD_PAYLOAD_FIELDS = [
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


def _write_event_manifest(
    root: Path,
    run_name: str,
    *,
    omit_first_payload: bool = False,
) -> None:
    trajectories = root / run_name / "trajectories"
    cache_dir = trajectories / "event_token_cache_v1"
    payload_dir = cache_dir / "development"
    shard_payload = {
        field_name: f"event_token_cache_v1/development/shard_00000_{field_name}.pt"
        for field_name in _SHARD_PAYLOAD_FIELDS
    }
    for index, relative_path in enumerate(shard_payload.values()):
        if omit_first_payload and index == 0:
            continue
        payload_path = trajectories / relative_path
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_bytes(b"x")
    _write_json(
        cache_dir / "event_token_cache_manifest.json",
        {
            "splits": {
                "development": {
                    "shards": [shard_payload],
                }
            }
        },
    )
    payload_dir.mkdir(parents=True, exist_ok=True)


def _write_r4_fixture(
    root: Path,
    *,
    include_run_b_manifest: bool = True,
    omit_run_b_payload: bool = False,
) -> None:
    _write_json(root / "baseline_head" / "build.time.json", {"real_seconds": 100.0})
    _write_json(root / "run_a" / "build.time.json", {"real_seconds": 210.0})
    _write_json(root / "run_b" / "build.time.json", {"real_seconds": 211.0})
    (root / "baseline_head" / "trajectories" / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "baseline_head" / "trajectories" / "manifest.json").write_text("{}\n", encoding="utf-8")
    (root / "baseline_head" / "trajectories" / "payload.bin").write_bytes(b"x" * 10_000)
    for run_name in ("run_a", "run_b"):
        if run_name != "run_b" or include_run_b_manifest:
            _write_event_manifest(root, run_name, omit_first_payload=run_name == "run_b" and omit_run_b_payload)
        (root / run_name / "trajectories" / "payload.bin").parent.mkdir(parents=True, exist_ok=True)
        (root / run_name / "trajectories" / "payload.bin").write_bytes(b"x" * 1100)
        _write_json(
            root / f"{run_name}_event_token_analysis.json",
            {
                "row_alignment_ok": True,
                "splits": {
                    "development": {
                        "diagnostics": {
                            "truncation_rate": 0.20,
                            "weighted_target_symbol_retained_rate": 0.80,
                            "weighted_burst_retention_rate": 0.75,
                            "cross_venue_ordered_adjacency_rate": 0.90,
                            "trade_to_bbo_ordered_adjacency_rate": 0.85,
                            "per_symbol_starvation_rate": {
                                "BTCUSDT": 0.0,
                                "ETHUSDT": 0.01,
                            },
                            "symbol_with_zero_retained_tokens_count_p95": 0.0,
                        }
                    }
                },
            },
        )
    semantic_hash = {
        "status": "ok",
        "file_count": 2,
        "tree_sha256": "same-tree",
        "excluded_nondeterministic_files": [
            "event_token_cache_diagnostics.json",
            "*_partial_selector_profile.json",
        ],
    }
    _write_json(root / "run_a_hashes.json", semantic_hash)
    _write_json(root / "run_b_hashes.json", semantic_hash)
    _write_json(
        root / "archive_receipt.json",
        {
            "archive_destination_prefix": "s3://quantlab-archive/quantlab/remote-runs/r4/",
            "verification_status": "verified",
            "verified_at": "2026-04-24T00:00:00Z",
        },
    )
    _write_json(root / "remote_prune_execute_report.json", {"status": "ok"})


def test_validate_ql033_r4_passes_complete_fixture(repo_root: Path, tmp_path: Path) -> None:
    validator = _load_validator_module(repo_root)
    run_root = tmp_path / "r4"
    _write_r4_fixture(run_root)

    report = validator.build_report(run_root=run_root, require_archive=True)

    assert report["executive_verdict"]["classification"] == "pass"
    assert report["executive_verdict"]["proceed"] is True
    assert report["checks"]["determinism"]["passed"] is True
    assert report["checks"]["archive_verified"]["passed"] is True


def test_validate_ql033_r4_fails_missing_manifest(repo_root: Path, tmp_path: Path) -> None:
    validator = _load_validator_module(repo_root)
    run_root = tmp_path / "r4"
    _write_r4_fixture(run_root, include_run_b_manifest=False)

    report = validator.build_report(run_root=run_root, require_archive=True)

    assert report["executive_verdict"]["classification"] == "fail"
    assert report["executive_verdict"]["proceed"] is False
    assert report["checks"]["run_b_complete"]["passed"] is False
    assert "run_b_complete" in report["executive_verdict"]["blocked_by"]


def test_validate_ql033_r4_fails_missing_shard_payload(repo_root: Path, tmp_path: Path) -> None:
    validator = _load_validator_module(repo_root)
    run_root = tmp_path / "r4"
    _write_r4_fixture(run_root, omit_run_b_payload=True)

    report = validator.build_report(run_root=run_root, require_archive=True)

    assert report["executive_verdict"]["classification"] == "fail"
    assert report["checks"]["run_b_complete"]["passed"] is False
    assert report["checks"]["run_b_complete"]["value"]["manifest_present"] is True
    assert report["checks"]["run_b_complete"]["value"]["missing_payload_count"] == 1
