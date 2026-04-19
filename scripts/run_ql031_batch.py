#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from quantlab_ml.common import dump_json_data, dump_model, load_yaml  # noqa: E402
from quantlab_ml.contracts import DatasetSpec, RewardEventSpec  # noqa: E402
from quantlab_ml.registry.analysis import (  # noqa: E402
    build_blocker_inventory,
    discover_retained_roots,
    import_diagnostic_retained_run,
    preflight_distinct_surface,
    render_blocker_inventory_markdown,
)
from quantlab_ml.registry.store import LocalRegistryStore  # noqa: E402

DEFAULT_WORKSPACE_REGISTRY_ROOTS = [
    Path("outputs/ql016-ql004-authoritative-minimum-20260418/registry"),
    Path("outputs/ql021-acceptance-proof-20260417-no-trpro7995wx/registry"),
]
DEFAULT_DIAGNOSTIC_BUNDLE_ROOTS = [
    Path("outputs/ql021-controlled-remote-rerun-20260417-build-fresh"),
]
DEFAULT_OUTPUT_ROOT = Path("outputs/analysis/ql031")
DEFAULT_RERUN_DATA_CONFIG = Path("configs/data/controlled-remote-day.yaml")
DEFAULT_RERUN_REWARD_CONFIG = Path("configs/reward/default.yaml")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the internal QL-031 blocker-inventory batch with diagnostic imports, "
            "retained-root discovery, and guarded rerun distinctness preflight."
        )
    )
    parser.add_argument(
        "--workspace-registry-root",
        action="append",
        default=[],
        help="Workspace-visible retained registry root. Repeat to override the built-in defaults.",
    )
    parser.add_argument(
        "--workspace-authority-status",
        action="append",
        default=[],
        help="Authority status aligned one-for-one with --workspace-registry-root.",
    )
    parser.add_argument(
        "--diagnostic-bundle-root",
        action="append",
        default=[],
        help="Bundle-complete retained root to normalize into an analysis-only diagnostic registry.",
    )
    parser.add_argument(
        "--external-search-root",
        action="append",
        default=[],
        help="Operator-supplied retained-root search root to classify before any rerun fallback.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Deterministic output directory for QL-031 analysis artifacts.",
    )
    parser.add_argument(
        "--rerun-data-config",
        type=Path,
        default=DEFAULT_RERUN_DATA_CONFIG,
        help="Candidate rerun data config used only for distinct-surface preflight.",
    )
    parser.add_argument(
        "--reward-config",
        type=Path,
        default=DEFAULT_RERUN_REWARD_CONFIG,
        help="Reward config used only for distinct-surface preflight.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    workspace_registry_roots = _normalize_paths(args.workspace_registry_root, DEFAULT_WORKSPACE_REGISTRY_ROOTS)
    diagnostic_bundle_roots = _normalize_paths(args.diagnostic_bundle_root, DEFAULT_DIAGNOSTIC_BUNDLE_ROOTS)
    external_search_roots = _normalize_paths(args.external_search_root, [])
    workspace_authority_statuses = _resolve_workspace_authority_statuses(
        roots=workspace_registry_roots,
        raw_statuses=list(args.workspace_authority_status),
    )
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    workspace_inventory = build_blocker_inventory(
        registry_roots=workspace_registry_roots,
        inspected_evidence_kinds=["external-retained-evidence"] * len(workspace_registry_roots),
        authority_statuses=workspace_authority_statuses,
    )
    workspace_inventory_paths = _write_inventory(
        inventory=workspace_inventory,
        output_root=output_root,
        stem="workspace_blocker_inventory",
    )

    diagnostic_imports: list[dict[str, Any]] = []
    diagnostic_registry_roots: list[Path] = []
    for bundle_root in diagnostic_bundle_roots:
        destination = output_root / "diagnostic_imports" / bundle_root.name
        if destination.exists():
            shutil.rmtree(destination)
        classification = import_diagnostic_retained_run(
            source_root=bundle_root,
            output_root=destination,
        )
        diagnostic_imports.append(classification)
        diagnostic_registry_roots.append(Path(classification["registry_root"]))

    workspace_plus_diagnostic_inventory = build_blocker_inventory(
        registry_roots=workspace_registry_roots + diagnostic_registry_roots,
        inspected_evidence_kinds=["external-retained-evidence"]
        * (len(workspace_registry_roots) + len(diagnostic_registry_roots)),
        authority_statuses=workspace_authority_statuses + ["unconfirmed"] * len(diagnostic_registry_roots),
    )
    workspace_plus_diagnostic_inventory_paths = _write_inventory(
        inventory=workspace_plus_diagnostic_inventory,
        output_root=output_root,
        stem="workspace_plus_diagnostic_blocker_inventory",
    )

    discovery_runs: list[dict[str, Any]] = []
    discovery_candidates: list[dict[str, Any]] = []
    for search_root in external_search_roots:
        discovery = discover_retained_roots(search_root=search_root)
        filtered_candidates = [
            candidate
            for candidate in discovery["candidates"]
            if not _is_under(Path(candidate["run_root"]), output_root)
        ]
        discovery["candidates"] = filtered_candidates
        discovery["candidate_count"] = len(filtered_candidates)
        discovery_runs.append(discovery)
        discovery_candidates.extend(filtered_candidates)
    discovery_path = output_root / "retained_root_discovery.json"
    dump_json_data(
        discovery_path,
        {
            "search_roots": [str(path) for path in external_search_roots],
            "results": discovery_runs,
            "candidates": discovery_candidates,
        },
    )

    comparison_reports: list[dict[str, str]] = []
    for candidate in _dedupe_candidates_by_registry_root(discovery_candidates):
        if candidate["candidate_classification"] != "usable_registry_root":
            continue
        preflight = candidate["comparison_preflight"]
        if not preflight["allowed"]:
            continue
        registry_root = Path(candidate["registry_root"])
        report = LocalRegistryStore(registry_root).record_comparison_report(
            preflight["challenger_policy_id"]
        )
        report_path = output_root / "comparison_reports" / f"{report.comparison_report_id}.json"
        dump_model(report_path, report)
        comparison_reports.append(
            {
                "registry_root": str(registry_root),
                "comparison_report_id": report.comparison_report_id,
                "challenger_policy_id": preflight["challenger_policy_id"] or "",
                "champion_policy_id": preflight["champion_policy_id"] or "",
                "output_path": str(report_path),
            }
        )

    baseline_surface_keys = _inventory_surface_keys(workspace_plus_diagnostic_inventory)
    distinct_retained_candidates = [
        {
            "run_root": candidate["run_root"],
            "registry_root": candidate["registry_root"],
            "candidate_classification": candidate["candidate_classification"],
            "surface_identity": surface_identity,
        }
        for candidate in discovery_candidates
        for surface_identity in candidate.get("surface_identities", [])
        if _surface_identity_key(surface_identity) not in baseline_surface_keys
    ]

    preflight_path: str | None = None
    rerun_preflight: dict[str, Any] | None = None
    checked_registry_roots = _dedupe_paths(
        workspace_registry_roots
        + diagnostic_registry_roots
        + [
            Path(candidate["registry_root"])
            for candidate in discovery_candidates
            if candidate["candidate_classification"] == "usable_registry_root" and candidate["registry_root"]
        ]
    )
    if not distinct_retained_candidates:
        dataset_spec = DatasetSpec.model_validate(load_yaml(args.rerun_data_config)["dataset"])
        reward_spec = RewardEventSpec.model_validate(load_yaml(args.reward_config)["reward"])
        rerun_preflight = preflight_distinct_surface(
            dataset_spec=dataset_spec,
            reward_spec=reward_spec,
            registry_roots=checked_registry_roots,
        )
        preflight_output = output_root / "distinct_surface_preflight.json"
        dump_json_data(preflight_output, rerun_preflight)
        preflight_path = str(preflight_output)

    status = _status_for_batch(
        distinct_retained_candidates=distinct_retained_candidates,
        rerun_preflight=rerun_preflight,
    )
    summary = {
        "status": status,
        "output_root": str(output_root),
        "workspace_registry_roots": [str(path) for path in workspace_registry_roots],
        "diagnostic_bundle_roots": [str(path) for path in diagnostic_bundle_roots],
        "external_search_roots": [str(path) for path in external_search_roots],
        "workspace_blocker_inventory": workspace_inventory_paths,
        "workspace_plus_diagnostic_blocker_inventory": workspace_plus_diagnostic_inventory_paths,
        "diagnostic_imports": diagnostic_imports,
        "retained_root_discovery_path": str(discovery_path),
        "discovered_candidate_count": len(discovery_candidates),
        "distinct_retained_candidates": distinct_retained_candidates,
        "comparison_reports": comparison_reports,
        "distinct_surface_preflight_path": preflight_path,
        "distinct_surface_preflight_allowed": None if rerun_preflight is None else rerun_preflight["allowed"],
    }
    summary_path = output_root / "ql031_status.json"
    dump_json_data(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _normalize_paths(raw_paths: list[str], defaults: list[Path]) -> list[Path]:
    if not raw_paths:
        return [path.expanduser().resolve() for path in defaults]
    return [Path(path).expanduser().resolve() for path in raw_paths]


def _resolve_workspace_authority_statuses(*, roots: list[Path], raw_statuses: list[str]) -> list[str]:
    if raw_statuses:
        if len(raw_statuses) != len(roots):
            raise ValueError("--workspace-authority-status must align one-for-one with --workspace-registry-root")
        return raw_statuses
    return [_default_authority_status(root) for root in roots]


def _default_authority_status(registry_root: Path) -> str:
    if "ql016-ql004-authoritative-minimum" in registry_root.as_posix():
        return "confirmed"
    return "unconfirmed"


def _write_inventory(*, inventory: dict[str, Any], output_root: Path, stem: str) -> dict[str, str]:
    json_path = output_root / f"{stem}.json"
    markdown_path = output_root / f"{stem}.md"
    dump_json_data(json_path, inventory)
    markdown_path.write_text(render_blocker_inventory_markdown(inventory), encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }


def _dedupe_candidates_by_registry_root(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        registry_root = candidate.get("registry_root")
        if not registry_root:
            continue
        deduped.setdefault(str(registry_root), candidate)
    return [deduped[key] for key in sorted(deduped)]


def _inventory_surface_keys(inventory: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (
            record["evaluation_surface_id"],
            record["slice_id"],
            record["train_window"],
        )
        for source in inventory["sources"]
        for record in source["policy_records"]
    }


def _surface_identity_key(surface_identity: dict[str, str]) -> tuple[str, str, str]:
    return (
        surface_identity["evaluation_surface_id"],
        surface_identity["slice_id"],
        surface_identity["train_window"],
    )


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: dict[Path, None] = {}
    for path in paths:
        deduped[path.expanduser().resolve()] = None
    return sorted(deduped)


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _status_for_batch(
    *,
    distinct_retained_candidates: list[dict[str, Any]],
    rerun_preflight: dict[str, Any] | None,
) -> str:
    if distinct_retained_candidates:
        return "distinct_retained_surface_found"
    if rerun_preflight is None:
        return "no_external_candidates_and_no_rerun_preflight"
    if not rerun_preflight["allowed"]:
        return "blocked_distinct_surface_collision"
    return "preflight_passed_no_distinct_retained_surface"


if __name__ == "__main__":
    raise SystemExit(main())
