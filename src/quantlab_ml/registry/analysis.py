from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import shutil
from typing import Any

from quantlab_ml.common import dump_json_data, hash_payload, load_model, utcnow
from quantlab_ml.contracts import (
    DatasetSpec,
    EvaluationReport,
    PolicyArtifact,
    PolicyScore,
    RewardEventSpec,
    TrajectoryManifest,
)
from quantlab_ml.contracts.learning_surface import _CANONICAL_SPLIT_VERSION
from quantlab_ml.contracts.policies import build_evaluation_surface_id
from quantlab_ml.registry.audit import audit_registry_continuity
from quantlab_ml.registry.store import LocalRegistryStore

_ACTIVE_STATUSES = {"candidate", "challenger", "champion"}
_DIAGNOSTIC_CLASSIFICATION_FILENAME = "import_classification.json"
_REQUIRED_BUNDLE_FILES = ("manifest.json", "policy.json", "evaluation.json", "score.json")
_RETAINED_REGISTRY_JSON_DIRS = (
    "records",
    "scores",
    "evaluations",
    "artifacts",
    "comparisons",
    "paper_sim",
    "promotions",
)


def build_blocker_inventory(
    *,
    registry_roots: list[Path],
    inspected_evidence_kinds: list[str],
    authority_statuses: list[str | None],
    repo_root: Path | None = None,
) -> dict[str, Any]:
    if not registry_roots:
        raise ValueError("blocker inventory requires at least one registry_root")
    if not (
        len(registry_roots) == len(inspected_evidence_kinds) == len(authority_statuses)
    ):
        raise ValueError("blocker inventory inputs must align one-for-one by source")

    resolved_repo_root = _resolve_repo_root(repo_root)
    grouped_by_snapshot: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"policy_ids": [], "registry_roots": []})
    grouped_by_surface: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"policy_ids": [], "registry_roots": []})
    grouped_by_window: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"policy_ids": [], "registry_roots": []})
    grouped_by_slice: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"policy_ids": [], "registry_roots": []})
    sources: list[dict[str, Any]] = []

    for registry_root, inspected_evidence_kind, authority_status in zip(
        registry_roots,
        inspected_evidence_kinds,
        authority_statuses,
        strict=True,
    ):
        resolved_root = _require_registry_root(registry_root)
        classification = _resolve_source_classification(
            resolved_root,
            inspected_evidence_kind=inspected_evidence_kind,
            authority_status=authority_status,
            repo_root=resolved_repo_root,
        )
        store = LocalRegistryStore(resolved_root)
        audit = audit_registry_continuity(
            store,
            inspected_evidence_kind=classification["inspected_evidence_kind"],
            authority_status=classification["authority_status"],
        )
        records = store.list_records()
        active_records = [record for record in records if record.status in _ACTIVE_STATUSES]
        comparison_reports = store.list_comparison_reports()
        paper_sim_records = store.list_paper_sim_evidence()
        comparison_preflight = preflight_same_root_comparison(
            registry_root=resolved_root,
            analysis_only=classification["analysis_only"],
        )
        source_limitations = _source_limitations(
            audit=audit,
            comparison_report_count=len(comparison_reports),
            paper_sim_evidence_count=len(paper_sim_records),
            analysis_only=classification["analysis_only"],
            comparison_preflight=comparison_preflight,
            missing_comparison_policy_ids=_missing_comparison_policy_ids(active_records),
            missing_paper_sim_policy_ids=_missing_paper_sim_policy_ids(active_records),
        )
        source_summary = {
            "registry_root": str(resolved_root),
            "source_root": classification["source_root"],
            "source_kind": classification["source_kind"],
            "inspected_evidence_kind": audit["inspected_evidence_kind"],
            "authority_status": audit["authority_status"],
            "analysis_only": classification["analysis_only"],
            "promotion_eligible": classification["promotion_eligible"],
            "comparison_eligible": comparison_preflight["allowed"],
            "classification_reasons": classification["classification_reasons"],
            "comparison_preflight": comparison_preflight,
            "audit_scope_verdict": audit["audit_scope_verdict"],
            "closeout_decision_allowed": audit["closeout_decision_allowed"],
            "closeout_blockers": audit["closeout_blockers"],
            "record_count": len(records),
            "active_record_count": len(active_records),
            "scored_record_count": sum(1 for record in active_records if record.score_history),
            "comparison_report_count": len(comparison_reports),
            "paper_sim_evidence_count": len(paper_sim_records),
            "training_snapshot_ids": sorted({record.training_snapshot_id for record in active_records}),
            "evaluation_surface_ids": sorted({record.evaluation_surface_id for record in active_records}),
            "slice_ids": sorted({record.slice_id for record in active_records}),
            "train_windows": sorted({_format_range(record.train_window) for record in active_records}),
            "eval_windows": sorted(
                {_format_range(record.eval_window) for record in active_records if record.eval_window is not None}
            ),
            "missing_comparison_policy_ids": _missing_comparison_policy_ids(active_records),
            "missing_paper_sim_policy_ids": _missing_paper_sim_policy_ids(active_records),
            "limitations": source_limitations,
            "policy_records": [
                {
                    "policy_id": record.policy_id,
                    "status": record.status,
                    "training_snapshot_id": record.training_snapshot_id,
                    "evaluation_surface_id": record.evaluation_surface_id,
                    "slice_id": record.slice_id,
                    "train_window": _format_range(record.train_window),
                    "eval_window": _format_range(record.eval_window) if record.eval_window is not None else None,
                    "comparison_report_id": record.comparison_report_id,
                    "paper_sim_evidence_id": record.paper_sim_evidence_id,
                }
                for record in active_records
            ],
        }
        sources.append(source_summary)

        for record in active_records:
            _append_group_entry(grouped_by_snapshot[record.training_snapshot_id], record.policy_id, resolved_root)
            _append_group_entry(grouped_by_surface[record.evaluation_surface_id], record.policy_id, resolved_root)
            _append_group_entry(grouped_by_window[_format_range(record.train_window)], record.policy_id, resolved_root)
            _append_group_entry(grouped_by_slice[record.slice_id], record.policy_id, resolved_root)

    overall_limitations = sorted(
        {
            limitation
            for source in sources
            for limitation in source["limitations"]
        }
    )
    return {
        "generated_at": utcnow().isoformat(),
        "inventory_id": f"blocker-inventory-{hash_payload(sources)[:12]}",
        "source_count": len(sources),
        "sources": sources,
        "grouped_by_training_snapshot": _sorted_group_map(grouped_by_snapshot),
        "grouped_by_evaluation_surface": _sorted_group_map(grouped_by_surface),
        "grouped_by_train_window": _sorted_group_map(grouped_by_window),
        "grouped_by_slice": _sorted_group_map(grouped_by_slice),
        "overall_limitations": overall_limitations,
    }


def render_blocker_inventory_markdown(inventory: dict[str, Any]) -> str:
    lines = [
        "# QL-031 Blocker Inventory",
        "",
        f"- Generated at: `{inventory['generated_at']}`",
        f"- Inventory ID: `{inventory['inventory_id']}`",
        f"- Source count: `{inventory['source_count']}`",
        "",
        "## Sources",
        "",
        "| Registry Root | Evidence Class | Authority | Analysis Only | Promotion Eligible | Comparison Eligible | Audit Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for source in inventory["sources"]:
        lines.append(
            "| "
            f"`{source['registry_root']}` | "
            f"`{source['inspected_evidence_kind']}` | "
            f"`{source['authority_status']}` | "
            f"`{str(source['analysis_only']).lower()}` | "
            f"`{str(source['promotion_eligible']).lower()}` | "
            f"`{str(source['comparison_eligible']).lower()}` | "
            f"`{source['audit_scope_verdict']}` |"
        )
    lines.extend(["", "## Grouped By Evaluation Surface", ""])
    lines.extend(_render_group_section(inventory["grouped_by_evaluation_surface"]))
    lines.extend(["", "## Grouped By Training Snapshot", ""])
    lines.extend(_render_group_section(inventory["grouped_by_training_snapshot"]))
    lines.extend(["", "## Grouped By Train Window", ""])
    lines.extend(_render_group_section(inventory["grouped_by_train_window"]))
    lines.extend(["", "## Grouped By Slice", ""])
    lines.extend(_render_group_section(inventory["grouped_by_slice"]))
    lines.extend(["", "## Limitations", ""])
    if inventory["overall_limitations"]:
        for limitation in inventory["overall_limitations"]:
            lines.append(f"- {limitation}")
    else:
        lines.append("- No additional inventory-level limitations were detected.")
    return "\n".join(lines) + "\n"


def import_diagnostic_retained_run(
    *,
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    resolved_source_root = source_root.expanduser().resolve()
    resolved_output_root = output_root.expanduser().resolve()
    _validate_diagnostic_source_root(resolved_source_root)
    _validate_diagnostic_output_root(resolved_output_root)
    registry_state = _registry_state_for_run(resolved_source_root)

    manifest = load_model(resolved_source_root / "manifest.json", TrajectoryManifest)
    artifact = load_model(resolved_source_root / "policy.json", PolicyArtifact)
    evaluation = load_model(resolved_source_root / "evaluation.json", EvaluationReport)
    score = load_model(resolved_source_root / "score.json", PolicyScore)
    expected_surface_id = build_evaluation_surface_id(
        slice_id=manifest.dataset_spec.slice_id,
        split_version=manifest.split_artifact.split_version,
        reward_version=manifest.reward_spec.reward_version,
    )
    if artifact.evaluation_surface_id != expected_surface_id:
        raise ValueError(
            "diagnostic import source artifact evaluation_surface_id does not match manifest-derived surface identity"
        )

    registry_root = resolved_output_root / "registry"
    store = LocalRegistryStore(registry_root)
    trajectory_directory = _resolve_diagnostic_trajectory_directory(
        source_root=resolved_source_root,
        output_root=resolved_output_root,
    )
    store.register_candidate_from_manifest(
        artifact,
        manifest,
        reward_config_hash=hash_payload(manifest.reward_spec),
        training_config_hash=artifact.training_config_hash,
        trajectory_directory=trajectory_directory,
    )
    store.append_score(artifact.policy_id, score, evaluation)

    classification = {
        "kind": "diagnostic_registry_import_v1",
        "source_root": str(resolved_source_root),
        "registry_root": str(registry_root),
        "inspected_evidence_kind": "external_retained_evidence",
        "authority_status": "unconfirmed",
        "analysis_only": True,
        "promotion_eligible": False,
        "comparison_eligible": False,
        "classification_reasons": [
            (
                "source_root_had_incomplete_registry_scaffold"
                if registry_state == "incomplete_registry_scaffold"
                else "source_root_had_no_usable_registry_state"
            ),
            "diagnostic_import_created_for_blocker_inventory_only",
            "promotion_is_forbidden_on_diagnostic_imports",
            "comparison_is_forbidden_on_diagnostic_imports",
            "cross_root_stitching_is_forbidden",
        ],
    }
    dump_json_data(resolved_output_root / _DIAGNOSTIC_CLASSIFICATION_FILENAME, classification)
    return classification


def discover_retained_roots(
    *,
    search_root: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    resolved_repo_root = _resolve_repo_root(repo_root)
    resolved_search_root = search_root.expanduser().resolve()
    candidate_run_roots = _discover_candidate_run_roots(resolved_search_root)
    candidates = [
        _summarize_retained_root(candidate_root, repo_root=resolved_repo_root)
        for candidate_root in candidate_run_roots
    ]
    return {
        "generated_at": utcnow().isoformat(),
        "search_root": str(resolved_search_root),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def preflight_same_root_comparison(
    *,
    registry_root: Path,
    challenger_policy_id: str | None = None,
    analysis_only: bool = False,
) -> dict[str, Any]:
    resolved_registry_root = registry_root.expanduser().resolve()
    if analysis_only:
        return {
            "registry_root": str(resolved_registry_root),
            "allowed": False,
            "champion_policy_id": None,
            "challenger_policy_id": None,
            "evaluation_surface_id": None,
            "eligible_challenger_policy_ids": [],
            "blocking_reasons": ["analysis_only_registry_root"],
        }
    if not resolved_registry_root.exists() or not resolved_registry_root.is_dir():
        return {
            "registry_root": str(resolved_registry_root),
            "allowed": False,
            "champion_policy_id": None,
            "challenger_policy_id": challenger_policy_id,
            "evaluation_surface_id": None,
            "eligible_challenger_policy_ids": [],
            "blocking_reasons": ["missing_registry_root"],
        }

    store = LocalRegistryStore(resolved_registry_root)
    index = store.load_index()
    champion_policy_id = index.champion_policy_id
    if champion_policy_id is None:
        return {
            "registry_root": str(resolved_registry_root),
            "allowed": False,
            "champion_policy_id": None,
            "challenger_policy_id": challenger_policy_id,
            "evaluation_surface_id": None,
            "eligible_challenger_policy_ids": [],
            "blocking_reasons": ["no_current_champion"],
        }

    champion = store.get_record(champion_policy_id)
    if champion is None:
        return {
            "registry_root": str(resolved_registry_root),
            "allowed": False,
            "champion_policy_id": champion_policy_id,
            "challenger_policy_id": challenger_policy_id,
            "evaluation_surface_id": None,
            "eligible_challenger_policy_ids": [],
            "blocking_reasons": ["current_champion_record_missing"],
        }
    if champion.status != "champion":
        return {
            "registry_root": str(resolved_registry_root),
            "allowed": False,
            "champion_policy_id": champion.policy_id,
            "challenger_policy_id": challenger_policy_id,
            "evaluation_surface_id": champion.evaluation_surface_id,
            "eligible_challenger_policy_ids": [],
            "blocking_reasons": ["current_champion_status_invalid"],
        }
    if not champion.score_history:
        return {
            "registry_root": str(resolved_registry_root),
            "allowed": False,
            "champion_policy_id": champion.policy_id,
            "challenger_policy_id": challenger_policy_id,
            "evaluation_surface_id": champion.evaluation_surface_id,
            "eligible_challenger_policy_ids": [],
            "blocking_reasons": ["current_champion_unscored"],
        }

    if challenger_policy_id is not None:
        challenger = store.get_record(challenger_policy_id)
        if challenger is None:
            return {
                "registry_root": str(resolved_registry_root),
                "allowed": False,
                "champion_policy_id": champion.policy_id,
                "challenger_policy_id": challenger_policy_id,
                "evaluation_surface_id": champion.evaluation_surface_id,
                "eligible_challenger_policy_ids": [],
                "blocking_reasons": ["requested_challenger_missing"],
            }
        if challenger.policy_id == champion.policy_id:
            return {
                "registry_root": str(resolved_registry_root),
                "allowed": False,
                "champion_policy_id": champion.policy_id,
                "challenger_policy_id": challenger.policy_id,
                "evaluation_surface_id": champion.evaluation_surface_id,
                "eligible_challenger_policy_ids": [],
                "blocking_reasons": ["requested_challenger_matches_champion"],
            }
        if not challenger.score_history:
            return {
                "registry_root": str(resolved_registry_root),
                "allowed": False,
                "champion_policy_id": champion.policy_id,
                "challenger_policy_id": challenger.policy_id,
                "evaluation_surface_id": champion.evaluation_surface_id,
                "eligible_challenger_policy_ids": [],
                "blocking_reasons": ["requested_challenger_unscored"],
            }
        if challenger.evaluation_surface_id != champion.evaluation_surface_id:
            return {
                "registry_root": str(resolved_registry_root),
                "allowed": False,
                "champion_policy_id": champion.policy_id,
                "challenger_policy_id": challenger.policy_id,
                "evaluation_surface_id": champion.evaluation_surface_id,
                "eligible_challenger_policy_ids": [],
                "blocking_reasons": ["requested_challenger_surface_mismatch"],
            }
        return {
            "registry_root": str(resolved_registry_root),
            "allowed": True,
            "champion_policy_id": champion.policy_id,
            "challenger_policy_id": challenger.policy_id,
            "evaluation_surface_id": champion.evaluation_surface_id,
            "eligible_challenger_policy_ids": [challenger.policy_id],
            "blocking_reasons": [],
        }

    eligible_challengers = sorted(
        record.policy_id
        for record in store.list_records()
        if record.policy_id != champion.policy_id
        and bool(record.score_history)
        and record.evaluation_surface_id == champion.evaluation_surface_id
    )
    if not eligible_challengers:
        return {
            "registry_root": str(resolved_registry_root),
            "allowed": False,
            "champion_policy_id": champion.policy_id,
            "challenger_policy_id": None,
            "evaluation_surface_id": champion.evaluation_surface_id,
            "eligible_challenger_policy_ids": [],
            "blocking_reasons": ["no_same_surface_scored_challenger"],
        }
    return {
        "registry_root": str(resolved_registry_root),
        "allowed": True,
        "champion_policy_id": champion.policy_id,
        "challenger_policy_id": eligible_challengers[0],
        "evaluation_surface_id": champion.evaluation_surface_id,
        "eligible_challenger_policy_ids": eligible_challengers,
        "blocking_reasons": [],
    }


def preflight_distinct_surface(
    *,
    dataset_spec: DatasetSpec,
    reward_spec: RewardEventSpec,
    registry_roots: list[Path],
) -> dict[str, Any]:
    candidate = candidate_surface_identity(dataset_spec=dataset_spec, reward_spec=reward_spec)
    collisions: list[dict[str, str]] = []

    for registry_root in registry_roots:
        resolved_root = _require_registry_root(registry_root)
        store = LocalRegistryStore(resolved_root)
        for record in store.list_records():
            if record.evaluation_surface_id == candidate["evaluation_surface_id"]:
                collisions.append(
                    {
                        "field": "evaluation_surface_id",
                        "value": candidate["evaluation_surface_id"],
                        "registry_root": str(resolved_root),
                        "policy_id": record.policy_id,
                    }
                )
            if record.slice_id == candidate["slice_id"]:
                collisions.append(
                    {
                        "field": "slice_id",
                        "value": candidate["slice_id"],
                        "registry_root": str(resolved_root),
                        "policy_id": record.policy_id,
                    }
                )
            if _format_range(record.train_window) == candidate["train_window"]:
                collisions.append(
                    {
                        "field": "train_window",
                        "value": candidate["train_window"],
                        "registry_root": str(resolved_root),
                        "policy_id": record.policy_id,
                    }
                )

    return {
        "allowed": not collisions,
        "candidate": candidate,
        "registry_roots_checked": [str(root.expanduser().resolve()) for root in registry_roots],
        "collisions": collisions,
    }


def candidate_surface_identity(
    *,
    dataset_spec: DatasetSpec,
    reward_spec: RewardEventSpec,
) -> dict[str, str]:
    return {
        "evaluation_surface_id": build_evaluation_surface_id(
            slice_id=dataset_spec.slice_id,
            split_version=_CANONICAL_SPLIT_VERSION,
            reward_version=reward_spec.reward_version,
        ),
        "slice_id": dataset_spec.slice_id,
        "train_window": _format_range(dataset_spec.train_range),
    }


def _resolve_repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return repo_root.expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def _require_registry_root(registry_root: Path) -> Path:
    resolved_root = registry_root.expanduser().resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise FileNotFoundError(f"registry root does not exist: {resolved_root}")
    if not _looks_like_registry_root(resolved_root):
        raise ValueError(f"registry root does not contain usable retained registry state: {resolved_root}")
    return resolved_root


def _resolve_source_classification(
    registry_root: Path,
    *,
    inspected_evidence_kind: str,
    authority_status: str | None,
    repo_root: Path,
) -> dict[str, Any]:
    normalized_kind = inspected_evidence_kind.strip().lower().replace("-", "_")
    normalized_authority = authority_status.strip().lower().replace("-", "_") if authority_status is not None else None
    sidecar = _load_source_classification_sidecar(registry_root)
    if normalized_kind == "authoritative_evidence" and _is_repo_outputs_retained_bundle(registry_root, repo_root):
        raise ValueError("repo outputs retained bundles must not be relabeled as authoritative_evidence")
    if sidecar is not None:
        expected_kind = sidecar["inspected_evidence_kind"]
        expected_authority = sidecar["authority_status"]
        if normalized_kind != expected_kind:
            raise ValueError(
                f"classification sidecar requires inspected_evidence_kind={expected_kind}; got {normalized_kind}"
            )
        if normalized_authority is None:
            normalized_authority = expected_authority
        elif normalized_authority != expected_authority:
            raise ValueError(
                f"classification sidecar requires authority_status={expected_authority}; got {normalized_authority}"
            )
        return {
            "source_root": sidecar["source_root"],
            "source_kind": "diagnostic_import",
            "inspected_evidence_kind": expected_kind,
            "authority_status": expected_authority,
            "analysis_only": sidecar["analysis_only"],
            "promotion_eligible": sidecar["promotion_eligible"],
            "classification_reasons": sidecar["classification_reasons"],
        }
    return {
        "source_root": str(registry_root.parent),
        "source_kind": "registry_root",
        "inspected_evidence_kind": normalized_kind,
        "authority_status": normalized_authority,
        "analysis_only": False,
        "promotion_eligible": True,
        "classification_reasons": [],
    }


def _load_source_classification_sidecar(registry_root: Path) -> dict[str, Any] | None:
    classification_path = registry_root.parent / _DIAGNOSTIC_CLASSIFICATION_FILENAME
    if not classification_path.exists():
        return None
    return json.loads(classification_path.read_text(encoding="utf-8"))


def _validate_diagnostic_source_root(source_root: Path) -> None:
    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"diagnostic import source root does not exist: {source_root}")
    missing = [name for name in _REQUIRED_BUNDLE_FILES if not (source_root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"diagnostic import source root is missing required retained files: {', '.join(sorted(missing))}"
        )
    if _registry_state_for_run(source_root) == "usable_registry_state":
        raise ValueError(
            "diagnostic import source root already contains usable retained registry state; "
            "import is reserved for bundle-complete roots without usable registry state"
        )


def _validate_diagnostic_output_root(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"diagnostic import output root must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)


def _discover_candidate_run_roots(search_root: Path) -> list[Path]:
    if not search_root.exists() or not search_root.is_dir():
        return []
    candidates: dict[Path, None] = {}
    if _looks_like_run_root(search_root):
        candidates[search_root] = None
    if _looks_like_registry_root(search_root):
        candidates[search_root.parent] = None
    for child in sorted(search_root.iterdir()):
        if not child.is_dir():
            continue
        if _looks_like_run_root(child):
            candidates[child] = None
            continue
        if _looks_like_registry_root(child):
            candidates[child.parent] = None
    return sorted(path.resolve() for path in candidates)


def _summarize_retained_root(run_root: Path, *, repo_root: Path) -> dict[str, Any]:
    registry_root = _resolve_registry_root_for_run(run_root)
    has_bundle_artifacts = all((run_root / filename).exists() for filename in _REQUIRED_BUNDLE_FILES)
    registry_state = _registry_state_for_run(run_root)
    has_incomplete_registry_scaffold = registry_state == "incomplete_registry_scaffold"
    if registry_root is not None:
        classification = _load_source_classification_sidecar(registry_root)
        analysis_only = bool(classification["analysis_only"]) if classification is not None else False
        comparison_preflight = preflight_same_root_comparison(
            registry_root=registry_root,
            analysis_only=analysis_only,
        )
        promotion_eligible = False if analysis_only else True
        comparison_eligible = comparison_preflight["allowed"]
        candidate_classification = "diagnostic_import_only_bundle" if analysis_only else "usable_registry_root"
        classification_reasons = (
            ["analysis_only_sidecar_present"]
            if analysis_only
            else ["usable_retained_registry_state_present"]
        )
        surface_identities = _surface_identities_from_registry(registry_root)
    else:
        classification = None
        comparison_preflight = {
            "registry_root": None,
            "allowed": False,
            "champion_policy_id": None,
            "challenger_policy_id": None,
            "evaluation_surface_id": None,
            "eligible_challenger_policy_ids": [],
            "blocking_reasons": ["missing_registry_state"],
        }
        promotion_eligible = False
        comparison_eligible = False
        surface_identities = _surface_identities_from_bundle(run_root)
        if has_bundle_artifacts:
            candidate_classification = "diagnostic_import_only_bundle"
            classification_reasons = ["bundle_artifacts_present"]
            if has_incomplete_registry_scaffold:
                classification_reasons.append("registry_scaffold_incomplete_or_empty")
            else:
                classification_reasons.append("usable_registry_state_missing")
        else:
            candidate_classification = "not_usable"
            classification_reasons = ["bundle_artifacts_missing", "usable_registry_state_missing"]

    return {
        "run_root": str(run_root),
        "registry_root": str(registry_root) if registry_root is not None else None,
        "has_registry_state": registry_root is not None,
        "has_bundle_artifacts": has_bundle_artifacts,
        "has_incomplete_registry_scaffold": has_incomplete_registry_scaffold,
        "is_repo_outputs_retained_bundle": (
            _is_repo_outputs_retained_bundle(registry_root, repo_root) if registry_root is not None else False
        ),
        "analysis_only": bool(classification["analysis_only"]) if classification is not None else False,
        "candidate_classification": candidate_classification,
        "classification_reasons": classification_reasons,
        "surface_identities": surface_identities,
        "promotion_eligible": promotion_eligible,
        "comparison_eligible": comparison_eligible,
        "comparison_preflight": comparison_preflight,
    }


def _resolve_registry_root_for_run(run_root: Path) -> Path | None:
    direct_registry_root = run_root / "registry"
    if _looks_like_registry_root(direct_registry_root):
        return direct_registry_root.resolve()
    if _looks_like_registry_root(run_root):
        return run_root.resolve()
    return None


def _looks_like_registry_root(path: Path) -> bool:
    return path.is_dir() and (
        (path / "index.json").exists()
        or any(
            next((path / directory).glob("*.json"), None) is not None
            for directory in _RETAINED_REGISTRY_JSON_DIRS
        )
    )


def _looks_like_run_root(path: Path) -> bool:
    if _looks_like_registry_root(path):
        return True
    if (path / "registry").is_dir():
        return True
    return all((path / filename).exists() for filename in _REQUIRED_BUNDLE_FILES)


def _registry_state_for_run(run_root: Path) -> str:
    registry_root = run_root / "registry"
    if not registry_root.exists():
        return "missing_registry"
    if _looks_like_registry_root(registry_root):
        return "usable_registry_state"
    return "incomplete_registry_scaffold"


def _surface_identities_from_registry(registry_root: Path) -> list[dict[str, str]]:
    identities: dict[tuple[str, str, str], dict[str, str]] = {}
    store = LocalRegistryStore(registry_root)
    for record in store.list_records():
        key = (
            record.evaluation_surface_id,
            record.slice_id,
            _format_range(record.train_window),
        )
        identities.setdefault(
            key,
            {
                "evaluation_surface_id": record.evaluation_surface_id,
                "slice_id": record.slice_id,
                "train_window": _format_range(record.train_window),
            },
        )
    return [identities[key] for key in sorted(identities)]


def _surface_identities_from_bundle(run_root: Path) -> list[dict[str, str]]:
    if not all((run_root / filename).exists() for filename in _REQUIRED_BUNDLE_FILES):
        return []
    manifest = load_model(run_root / "manifest.json", TrajectoryManifest)
    return [
        {
            "evaluation_surface_id": build_evaluation_surface_id(
                slice_id=manifest.dataset_spec.slice_id,
                split_version=manifest.split_artifact.split_version,
                reward_version=manifest.reward_spec.reward_version,
            ),
            "slice_id": manifest.dataset_spec.slice_id,
            "train_window": _format_range(manifest.dataset_spec.train_range),
        }
    ]


def _resolve_diagnostic_trajectory_directory(*, source_root: Path, output_root: Path) -> Path | None:
    direct_tensor_cache_manifest = source_root / "tensor_cache_v1" / "tensor_cache_manifest.json"
    if direct_tensor_cache_manifest.exists():
        return source_root

    trajectories_dir = source_root / "trajectories"
    if (trajectories_dir / "tensor_cache_v1" / "tensor_cache_manifest.json").exists():
        return trajectories_dir

    legacy_root_tensor_cache_manifest = source_root / "tensor_cache_manifest.json"
    if not legacy_root_tensor_cache_manifest.exists():
        return None

    recovery_directory = output_root / "_coverage_recovery"
    cache_directory = recovery_directory / "tensor_cache_v1"
    cache_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        legacy_root_tensor_cache_manifest,
        cache_directory / "tensor_cache_manifest.json",
    )
    return recovery_directory


def _append_group_entry(bucket: dict[str, list[str]], policy_id: str, registry_root: Path) -> None:
    if policy_id not in bucket["policy_ids"]:
        bucket["policy_ids"].append(policy_id)
    root_text = str(registry_root)
    if root_text not in bucket["registry_roots"]:
        bucket["registry_roots"].append(root_text)


def _sorted_group_map(grouped: dict[str, dict[str, list[str]]]) -> dict[str, dict[str, list[str]]]:
    return {
        key: {
            "policy_ids": sorted(value["policy_ids"]),
            "registry_roots": sorted(value["registry_roots"]),
        }
        for key, value in sorted(grouped.items())
    }


def _format_range(time_range: Any) -> str:
    return f"{time_range.start.isoformat()} -> {time_range.end.isoformat()}"


def _source_limitations(
    *,
    audit: dict[str, Any],
    comparison_report_count: int,
    paper_sim_evidence_count: int,
    analysis_only: bool,
    comparison_preflight: dict[str, Any],
    missing_comparison_policy_ids: list[str],
    missing_paper_sim_policy_ids: list[str],
) -> list[str]:
    limitations: list[str] = []
    if audit["inspected_evidence_kind"] != "authoritative_evidence":
        limitations.append(
            "This source remains non-authoritative evidence and must not be relabeled as authoritative evidence."
        )
    if audit["authority_status"] != "confirmed":
        limitations.append("Authority remains unconfirmed for this source.")
    if analysis_only:
        limitations.append("This source is analysis-only and must not be used for promotion or comparison workflows.")
    if comparison_report_count == 0:
        limitations.append("No registry-backed comparison reports were found for this source.")
    if paper_sim_evidence_count == 0:
        limitations.append("No paper/sim evidence was linked for this source.")
    if comparison_preflight["blocking_reasons"]:
        limitations.append(
            f"Same-root comparison preflight is blocked: {', '.join(comparison_preflight['blocking_reasons'])}"
        )
    if missing_comparison_policy_ids or missing_paper_sim_policy_ids:
        limitations.append("Scored challengers still require explicit comparison and paper/sim linkage review.")
    return sorted(set(limitations))


def _missing_comparison_policy_ids(active_records: list[Any]) -> list[str]:
    return sorted(
        record.policy_id
        for record in active_records
        if record.status == "challenger" and record.score_history and record.comparison_report_id is None
    )


def _missing_paper_sim_policy_ids(active_records: list[Any]) -> list[str]:
    return sorted(
        record.policy_id
        for record in active_records
        if record.status == "challenger" and record.score_history and record.paper_sim_evidence_id is None
    )


def _render_group_section(grouped: dict[str, dict[str, list[str]]]) -> list[str]:
    if not grouped:
        return ["- No grouped entries found."]
    lines = [
        "| Key | Policy IDs | Registry Roots |",
        "| --- | --- | --- |",
    ]
    for key, value in grouped.items():
        policy_ids = ", ".join(f"`{policy_id}`" for policy_id in value["policy_ids"])
        registry_roots = ", ".join(f"`{registry_root}`" for registry_root in value["registry_roots"])
        lines.append(f"| `{key}` | {policy_ids} | {registry_roots} |")
    return lines


def _is_repo_outputs_retained_bundle(registry_root: Path, repo_root: Path) -> bool:
    try:
        relative = registry_root.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    return relative.parts[:1] == ("outputs",)
