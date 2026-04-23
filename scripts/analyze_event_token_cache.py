from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quantlab_ml.common import dump_json_data
from quantlab_ml.trajectories.event_token_cache import (
    read_event_token_cache_diagnostics,
    read_event_token_cache_manifest,
)
from quantlab_ml.trajectories.streaming_store import TrajectoryDirectoryStore
from quantlab_ml.trajectories.tensor_cache import read_tensor_cache_manifest


def _trajectory_step_counts(directory: Path) -> dict[str, int]:
    manifest = TrajectoryDirectoryStore.read_manifest(directory)
    counts: dict[str, int] = {}
    for split_name in manifest.split_names:
        counts[split_name] = sum(
            len(record.steps) for record in TrajectoryDirectoryStore.iter_records(directory, split_name)
        )
    return counts


def build_report(trajectories_root: Path) -> dict[str, Any]:
    event_manifest = read_event_token_cache_manifest(trajectories_root)
    event_diagnostics = read_event_token_cache_diagnostics(trajectories_root)
    tensor_manifest = read_tensor_cache_manifest(trajectories_root)
    trajectory_steps = _trajectory_step_counts(trajectories_root)

    split_reports: dict[str, Any] = {}
    row_alignment_ok = True
    for split_name, event_split in event_manifest.splits.items():
        tensor_rows = tensor_manifest.splits[split_name].row_count
        trajectory_rows = trajectory_steps[split_name]
        aligned = event_split.row_count == tensor_rows == trajectory_rows
        row_alignment_ok = row_alignment_ok and aligned
        split_reports[split_name] = {
            "row_alignment_ok": aligned,
            "trajectory_rows": trajectory_rows,
            "tensor_cache_rows": tensor_rows,
            "event_token_rows": event_split.row_count,
            "event_token_count": event_split.token_count,
            "diagnostics": event_diagnostics.splits[split_name].model_dump(mode="json"),
        }

    return {
        "event_token_cache_format_version": event_manifest.format_version,
        "event_window_contract_version": event_manifest.event_window_contract_version,
        "tokenizer_version": event_manifest.tokenizer_version,
        "selection_policy_id": event_manifest.selection_policy_id,
        "selection_hyperparameters": event_manifest.selection_hyperparameters.model_dump(mode="json"),
        "selector_params_hash": event_manifest.selector_params_hash,
        "token_cap": event_manifest.token_cap,
        "lookback_seconds": event_manifest.lookback_seconds,
        "row_alignment_ok": row_alignment_ok,
        "audit_artifact_relative_path": event_diagnostics.audit_artifact_relative_path,
        "splits": split_reports,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Event Token Cache Analysis",
        "",
        f"- selection policy: `{report['selection_policy_id']}`",
        f"- selector params hash: `{report['selector_params_hash']}`",
        f"- event window contract: `{report['event_window_contract_version']}`",
        f"- tokenizer version: `{report['tokenizer_version']}`",
        f"- token cap: `{report['token_cap']}`",
        f"- lookback seconds: `{report['lookback_seconds']}`",
        f"- row alignment ok: `{report['row_alignment_ok']}`",
        "",
    ]
    for split_name, split_report in report["splits"].items():
        diagnostics = split_report["diagnostics"]
        lines.extend(
            [
                f"## {split_name}",
                "",
                f"- row alignment ok: `{split_report['row_alignment_ok']}`",
                f"- rows: trajectory={split_report['trajectory_rows']} tensor={split_report['tensor_cache_rows']} event={split_report['event_token_rows']}",
                f"- tokens: `{split_report['event_token_count']}`",
                f"- truncation rate: `{diagnostics['truncation_rate']}`",
                f"- weighted target retained rate: `{diagnostics.get('weighted_target_symbol_retained_rate')}`",
                f"- weighted raw target retained rate: `{diagnostics.get('weighted_raw_target_symbol_retained_rate')}`",
                f"- weighted burst retention rate: `{diagnostics.get('weighted_burst_retention_rate')}`",
                f"- cross-venue adjacency rate: `{diagnostics.get('cross_venue_ordered_adjacency_rate')}`",
                f"- trade-to-bbo adjacency rate: `{diagnostics.get('trade_to_bbo_ordered_adjacency_rate')}`",
                f"- significant BBO preservation rate: `{diagnostics.get('significant_bbo_preservation_rate')}`",
                f"- informative candidates by tier: `{diagnostics.get('informative_candidate_by_tier')}`",
                f"- T4 candidates: `{diagnostics.get('t4_candidate_total')}`",
                f"- T4 anchors: `{diagnostics.get('t4_anchor_total')}`",
                f"- T4 resolution wall seconds: `{diagnostics.get('t4_resolution_wall_sec')}`",
                f"- BBO significance wall seconds: `{diagnostics.get('bbo_significance_wall_sec')}`",
                f"- quota fill wall seconds: `{diagnostics.get('quota_fill_wall_sec')}`",
                f"- diagnostics serialization wall seconds: `{diagnostics.get('diagnostics_serialization_wall_sec')}`",
                f"- total selector wall seconds: `{diagnostics.get('total_selector_wall_sec')}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze event_token_cache_v1 diagnostics and manifest state.")
    parser.add_argument("--trajectories-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, help="Optional JSON output path.")
    parser.add_argument("--report-md", type=Path, help="Optional Markdown output path.")
    args = parser.parse_args()

    trajectories_root = args.trajectories_root.expanduser().resolve()
    report = build_report(trajectories_root)
    if args.report_json is not None:
        dump_json_data(args.report_json.expanduser().resolve(), report)
    if args.report_md is not None:
        _write_markdown(args.report_md.expanduser().resolve(), report)
    if args.report_json is None and args.report_md is None:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
