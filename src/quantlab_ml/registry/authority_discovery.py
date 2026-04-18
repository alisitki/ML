from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quantlab_ml.registry.audit import audit_registry_continuity
from quantlab_ml.registry.store import LocalRegistryStore

DEFAULT_SEARCH_ROOTS = (Path("/workspace/runs"), Path("/root/runs"))
_STAGE_NAMES = ("build", "train", "evaluate", "score", "export")


def discover_continuity_authority(
    *,
    search_roots: Sequence[Path | str] | None = None,
    repo_root: Path | None = None,
    include_default_search_roots: bool = True,
) -> dict[str, Any]:
    resolved_repo_root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    root_specs = _build_search_root_specs(
        search_roots=search_roots,
        include_default_search_roots=include_default_search_roots,
    )

    searched_roots: list[dict[str, Any]] = []
    deduped_candidates: dict[Path, tuple[Path, str]] = {}
    for root, source_kind in root_specs:
        resolved_root = root.expanduser().resolve()
        candidate_roots = _discover_registry_roots(resolved_root)
        searched_roots.append(
            {
                "path": str(resolved_root),
                "source_kind": source_kind,
                "exists": resolved_root.exists(),
                "candidate_registry_roots": [str(candidate) for candidate in candidate_roots],
            }
        )
        for candidate_root in candidate_roots:
            deduped_candidates.setdefault(candidate_root, (resolved_root, source_kind))

    candidates = [
        summarize_registry_candidate(
            candidate_root,
            repo_root=resolved_repo_root,
            search_root=search_root,
            search_root_kind=search_root_kind,
        )
        for candidate_root, (search_root, search_root_kind) in sorted(
            deduped_candidates.items(),
            key=lambda item: str(item[0]),
        )
    ]

    eligible_external_candidates = [
        candidate
        for candidate in candidates
        if candidate["candidate_classification"] == "eligible_for_authority_confirmation"
    ]
    decision, decision_reason = _discovery_decision(eligible_external_candidates)
    authority_confirmation_candidate_path = (
        eligible_external_candidates[0]["registry_root"] if len(eligible_external_candidates) == 1 else None
    )
    closeout_sentence = (
        "authority confirmation step allowed"
        if authority_confirmation_candidate_path is not None
        else "authoritative scope still blocked; closeout remains pending with sharper blockers"
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "repo_root": str(resolved_repo_root),
        "default_search_roots": [str(path) for path in DEFAULT_SEARCH_ROOTS],
        "searched_roots": searched_roots,
        "candidates": candidates,
        "eligible_external_candidate_count": len(eligible_external_candidates),
        "eligible_external_candidate_paths": [
            str(candidate["registry_root"]) for candidate in eligible_external_candidates
        ],
        "decision": decision,
        "decision_reason": decision_reason,
        "authority_confirmation_step_allowed": authority_confirmation_candidate_path is not None,
        "authority_confirmation_candidate_path": authority_confirmation_candidate_path,
        "recommended_sprint_closeout_sentence": closeout_sentence,
    }


def summarize_registry_candidate(
    registry_root: Path,
    *,
    repo_root: Path,
    search_root: Path | None = None,
    search_root_kind: str | None = None,
) -> dict[str, Any]:
    resolved_registry_root = registry_root.resolve()
    run_root = resolved_registry_root.parent
    if not resolved_registry_root.exists() or not resolved_registry_root.is_dir():
        return {
            "registry_root": str(resolved_registry_root),
            "run_root": str(run_root),
            "search_root": str(search_root) if search_root is not None else None,
            "search_root_kind": search_root_kind,
            "is_repo_outputs_retained_bundle": _is_repo_outputs_retained_bundle(
                resolved_registry_root, repo_root
            ),
            "candidate_classification": "not_eligible",
            "non_eligibility_reasons": ["broken_registry_root"],
            "stage_exit_codes": {},
            "stage_exit_failures": [],
        }

    registry = LocalRegistryStore(resolved_registry_root)
    records = registry.list_records()
    latest_record_updated_at = (
        max(record.updated_at for record in records).isoformat() if records else None
    )
    stage_exit_codes, stage_exit_failures = _read_stage_exit_codes(run_root)
    audit_summary = audit_registry_continuity(
        registry,
        inspected_evidence_kind="external_retained_evidence",
        authority_status="unconfirmed",
    )
    non_eligibility_reasons = _classify_candidate(
        resolved_registry_root,
        repo_root=repo_root,
        audit_summary=audit_summary,
        stage_exit_failures=stage_exit_failures,
    )

    return {
        "registry_root": str(resolved_registry_root),
        "run_root": str(run_root),
        "search_root": str(search_root) if search_root is not None else None,
        "search_root_kind": search_root_kind,
        "is_repo_outputs_retained_bundle": _is_repo_outputs_retained_bundle(
            resolved_registry_root, repo_root
        ),
        "candidate_classification": _candidate_classification(
            resolved_registry_root,
            repo_root=repo_root,
            non_eligibility_reasons=non_eligibility_reasons,
        ),
        "non_eligibility_reasons": non_eligibility_reasons,
        "audit_scope_verdict": audit_summary["audit_scope_verdict"],
        "active_record_count": audit_summary["active_record_count"],
        "readable_active_artifact_count": audit_summary["readable_active_artifact_count"],
        "artifact_load_failures": audit_summary["artifact_load_failures"],
        "active_training_backend_counts": audit_summary["active_training_backend_counts"],
        "active_legacy_compat_artifact_count": audit_summary["active_legacy_compat_artifact_count"],
        "registry_local_fallback_policy_ids": audit_summary["registry_local_fallback_policy_ids"],
        "latest_record_updated_at": latest_record_updated_at,
        "stage_exit_codes": stage_exit_codes,
        "stage_exit_failures": stage_exit_failures,
        "closeout_blockers": audit_summary["closeout_blockers"],
    }


def _build_search_root_specs(
    *,
    search_roots: Sequence[Path | str] | None,
    include_default_search_roots: bool,
) -> list[tuple[Path, str]]:
    root_specs: list[tuple[Path, str]] = []
    for root in search_roots or ():
        root_specs.append((Path(root), "operator_supplied"))
    if include_default_search_roots:
        root_specs.extend((path, "default") for path in DEFAULT_SEARCH_ROOTS)
    return root_specs


def _discover_registry_roots(search_root: Path) -> list[Path]:
    if not search_root.exists() or not search_root.is_dir():
        return []

    candidates: dict[Path, None] = {}
    if _looks_like_registry_root(search_root):
        candidates[search_root.resolve()] = None

    direct_registry = search_root / "registry"
    if direct_registry.is_dir():
        candidates[direct_registry.resolve()] = None

    for child in sorted(search_root.iterdir()):
        if not child.is_dir():
            continue
        candidate = child / "registry"
        if candidate.is_dir():
            candidates[candidate.resolve()] = None
    return list(candidates)


def _looks_like_registry_root(path: Path) -> bool:
    return path.name == "registry" and (path / "records").exists()


def _read_stage_exit_codes(run_root: Path) -> tuple[dict[str, int], list[str]]:
    stage_exit_codes: dict[str, int] = {}
    failures: list[str] = []
    for stage in _STAGE_NAMES:
        exit_path = run_root / f"{stage}.exit"
        if not exit_path.exists():
            continue
        try:
            raw_value = exit_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            failures.append(f"unreadable_stage_exit_file:{stage}:{exc.__class__.__name__}")
            continue
        try:
            exit_code = int(raw_value)
        except ValueError:
            failures.append(f"invalid_stage_exit_code:{stage}:{raw_value}")
            continue
        stage_exit_codes[stage] = exit_code
        if exit_code != 0:
            failures.append(f"nonzero_stage_exit_code:{stage}={exit_code}")
    return stage_exit_codes, failures


def _classify_candidate(
    registry_root: Path,
    *,
    repo_root: Path,
    audit_summary: dict[str, Any],
    stage_exit_failures: list[str],
) -> list[str]:
    reasons: list[str] = []
    if _is_repo_outputs_retained_bundle(registry_root, repo_root):
        reasons.append("candidate_is_repo_local_retained_bundle")
    if audit_summary["active_record_count"] == 0:
        reasons.append("no_active_records_in_registry_scope")
    if audit_summary["artifact_load_failures"]:
        reasons.append("unreadable_active_artifact_paths")
    if audit_summary["readable_active_artifact_count"] != audit_summary["active_record_count"]:
        reasons.append("active_record_artifact_readability_mismatch")
    reasons.extend(stage_exit_failures)
    return reasons


def _candidate_classification(
    registry_root: Path,
    *,
    repo_root: Path,
    non_eligibility_reasons: list[str],
) -> str:
    if _is_repo_outputs_retained_bundle(registry_root, repo_root):
        return "retained_bundle_only"
    if non_eligibility_reasons:
        return "not_eligible"
    return "eligible_for_authority_confirmation"


def _is_repo_outputs_retained_bundle(registry_root: Path, repo_root: Path) -> bool:
    try:
        relative = registry_root.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    return relative.parts[:1] == ("outputs",)


def _discovery_decision(eligible_external_candidates: list[dict[str, Any]]) -> tuple[str, str]:
    if not eligible_external_candidates:
        return (
            "blocked_no_eligible_external_candidates",
            "no eligible external candidate found across searched roots",
        )
    if len(eligible_external_candidates) > 1:
        return (
            "blocked_ambiguous_external_candidates",
            "multiple eligible external candidates found across searched roots",
        )
    return (
        "authority_confirmation_step_allowed",
        "exactly one eligible external candidate found; authority confirmation step allowed",
    )
