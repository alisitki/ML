from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from quantlab_ml.common import load_model
from quantlab_ml.contracts import LEGACY_POLICY_ARTIFACT_SCHEMA_VERSION, PolicyArtifact
from quantlab_ml.contracts.registry import (
    ContinuityAuditSummary,
    RegistryRecord,
)
from quantlab_ml.registry.store import LocalRegistryStore

_ACTIVE_STATUSES = {"candidate", "challenger", "champion"}
_VALID_INSPECTED_EVIDENCE_KINDS = {
    "repo_tracked_artifact",
    "external_retained_evidence",
    "authoritative_evidence",
}
_VALID_AUTHORITY_STATUSES = {"confirmed", "unconfirmed", "unknown"}


def audit_registry_continuity(
    registry: LocalRegistryStore,
    *,
    inspected_evidence_kind: str = "external_retained_evidence",
    authority_status: str | None = None,
) -> dict[str, object]:
    inspected_evidence_kind = _normalize_inspected_evidence_kind(inspected_evidence_kind)
    authority_status = _resolve_authority_status(inspected_evidence_kind, authority_status)
    records = registry.list_records()
    active_records = [record for record in records if record.status in _ACTIVE_STATUSES]
    active_status_counts = Counter(record.status for record in active_records)
    active_training_backend_counts: Counter[str] = Counter()
    active_training_backend_policy_ids: dict[str, list[str]] = defaultdict(list)
    legacy_compat_policy_ids: list[str] = []
    deprecated_momentum_policy_ids: list[str] = []
    artifact_load_failures: list[dict[str, str]] = []
    registry_local_fallback_policy_ids: list[str] = []
    readable_active_artifact_count = 0

    for record in active_records:
        artifact_path, resolution_mode, resolution_attempts = _resolve_artifact_path(record, registry)
        if artifact_path is None:
            artifact_load_failures.append(
                {
                    "policy_id": record.policy_id,
                    "recorded_artifact_path": record.artifact_path,
                    "resolution_attempts": ", ".join(resolution_attempts),
                    "reason": (
                        "artifact path is unreadable from the inspected registry root; "
                        "use the authoritative registry root or provide a relocation-safe retained bundle"
                    ),
                }
            )
            continue
        if resolution_mode == "registry_local_fallback":
            registry_local_fallback_policy_ids.append(record.policy_id)
        try:
            artifact = load_model(artifact_path, PolicyArtifact)
        except Exception as exc:
            artifact_load_failures.append(
                {
                    "policy_id": record.policy_id,
                    "recorded_artifact_path": record.artifact_path,
                    "resolution_attempts": ", ".join(resolution_attempts),
                    "reason": f"{exc.__class__.__name__}: {exc}",
                }
            )
            continue
        readable_active_artifact_count += 1
        training_backend = str(artifact.training_summary.get("training_backend") or "unknown")
        active_training_backend_counts[training_backend] += 1
        active_training_backend_policy_ids[training_backend].append(record.policy_id)
        if (
            artifact.schema_version == LEGACY_POLICY_ARTIFACT_SCHEMA_VERSION
            and artifact.policy_payload.runtime_adapter == "linear-policy-v1"
        ):
            legacy_compat_policy_ids.append(record.policy_id)
        if artifact.policy_payload.runtime_adapter == "momentum-baseline-v1":
            deprecated_momentum_policy_ids.append(record.policy_id)

    backend_counts = dict(sorted(active_training_backend_counts.items()))
    backend_policy_ids = {
        backend: sorted(policy_ids)
        for backend, policy_ids in sorted(active_training_backend_policy_ids.items())
    }
    legacy_compat_policy_ids = sorted(legacy_compat_policy_ids)
    deprecated_momentum_policy_ids = sorted(deprecated_momentum_policy_ids)
    blocking_reasons = []
    if not active_records:
        blocking_reasons.append("no_active_records_in_registry_scope")
    if artifact_load_failures:
        blocking_reasons.append("unreadable_active_artifact_paths")
    closeout_blockers = list(blocking_reasons)
    if authority_status != "confirmed":
        closeout_blockers.append("authoritative_scope_not_confirmed")
    ready_to_close_numpy_continuity_window = (
        not blocking_reasons
        and bool(active_records)
        and active_training_backend_counts.get("numpy", 0) == 0
        and set(active_training_backend_counts).issubset({"pytorch"})
    )
    ready_to_retire_legacy_compat_window = (
        not blocking_reasons and bool(active_records) and len(legacy_compat_policy_ids) == 0
    )
    audit_scope_verdict = _audit_scope_verdict(
        blocking_reasons=blocking_reasons,
        numpy_dependency_count=active_training_backend_counts.get("numpy", 0),
        legacy_dependency_count=len(legacy_compat_policy_ids),
        deprecated_momentum_dependency_count=len(deprecated_momentum_policy_ids),
    )

    summary = ContinuityAuditSummary(
        registry_root=str(registry.root),
        inspected_evidence_kind=inspected_evidence_kind,
        authority_status=authority_status,
        closeout_decision_allowed=authority_status == "confirmed",
        closeout_blockers=closeout_blockers,
        record_count=len(records),
        active_record_count=len(active_records),
        readable_active_artifact_count=readable_active_artifact_count,
        active_status_counts=dict(sorted(active_status_counts.items())),
        active_training_backend_counts=backend_counts,
        active_training_backend_policy_ids=backend_policy_ids,
        active_numpy_training_backend_count=active_training_backend_counts.get("numpy", 0),
        active_numpy_training_backend_policy_ids=backend_policy_ids.get("numpy", []),
        active_legacy_compat_artifact_count=len(legacy_compat_policy_ids),
        active_legacy_compat_policy_ids=legacy_compat_policy_ids,
        active_deprecated_momentum_artifact_count=len(deprecated_momentum_policy_ids),
        active_deprecated_momentum_policy_ids=deprecated_momentum_policy_ids,
        registry_local_fallback_policy_ids=sorted(registry_local_fallback_policy_ids),
        artifact_load_failures=artifact_load_failures,
        blocking_reasons=blocking_reasons,
        audit_scope_verdict=audit_scope_verdict,
        ready_to_close_numpy_continuity_window=ready_to_close_numpy_continuity_window,
        ready_to_retire_legacy_compat_window=ready_to_retire_legacy_compat_window,
    )
    return summary.model_dump(mode="json")


def _resolve_artifact_path(
    record: RegistryRecord,
    registry: LocalRegistryStore,
) -> tuple[Path | None, str, list[str]]:
    recorded_path = Path(record.artifact_path)
    registry_local_path = registry.artifacts_dir / f"{record.policy_id}.json"
    resolution_attempts = [str(recorded_path)]

    if recorded_path.exists():
        return recorded_path, "recorded_path", resolution_attempts
    if registry_local_path != recorded_path:
        resolution_attempts.append(str(registry_local_path))
    if registry_local_path.exists():
        return registry_local_path, "registry_local_fallback", resolution_attempts
    return None, "unreadable", resolution_attempts


def _audit_scope_verdict(
    *,
    blocking_reasons: list[str],
    numpy_dependency_count: int,
    legacy_dependency_count: int,
    deprecated_momentum_dependency_count: int,
) -> str:
    if blocking_reasons:
        return "blocked"
    if numpy_dependency_count or legacy_dependency_count or deprecated_momentum_dependency_count:
        return "active_dependency_present"
    return "clear_in_inspected_scope"


def _normalize_inspected_evidence_kind(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in _VALID_INSPECTED_EVIDENCE_KINDS:
        raise ValueError(
            "unsupported inspected_evidence_kind: "
            f"{value!r}; expected one of {sorted(_VALID_INSPECTED_EVIDENCE_KINDS)}"
        )
    return normalized


def _resolve_authority_status(inspected_evidence_kind: str, authority_status: str | None) -> str:
    if inspected_evidence_kind == "authoritative_evidence":
        if authority_status is None:
            return "confirmed"
        normalized = _normalize_authority_status(authority_status)
        if normalized != "confirmed":
            raise ValueError("authoritative_evidence requires authority_status=confirmed")
        return normalized
    if authority_status is None:
        if inspected_evidence_kind == "repo_tracked_artifact":
            return "unknown"
        return "unconfirmed"
    return _normalize_authority_status(authority_status)


def _normalize_authority_status(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in _VALID_AUTHORITY_STATUSES:
        raise ValueError(
            "unsupported authority_status: "
            f"{value!r}; expected one of {sorted(_VALID_AUTHORITY_STATUSES)}"
        )
    return normalized
