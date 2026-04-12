from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from quantlab_ml.common import load_model
from quantlab_ml.contracts import LEGACY_POLICY_ARTIFACT_SCHEMA_VERSION, PolicyArtifact
from quantlab_ml.registry.store import LocalRegistryStore

_ACTIVE_STATUSES = {"candidate", "challenger", "champion"}


def audit_registry_continuity(registry: LocalRegistryStore) -> dict[str, object]:
    records = registry.list_records()
    active_records = [record for record in records if record.status in _ACTIVE_STATUSES]
    active_status_counts = Counter(record.status for record in active_records)
    active_training_backend_counts: Counter[str] = Counter()
    active_training_backend_policy_ids: dict[str, list[str]] = defaultdict(list)
    legacy_compat_policy_ids: list[str] = []
    deprecated_momentum_policy_ids: list[str] = []

    for record in active_records:
        artifact = load_model(Path(record.artifact_path), PolicyArtifact)
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
    ready_to_close_numpy_continuity_window = (
        bool(active_records)
        and active_training_backend_counts.get("numpy", 0) == 0
        and set(active_training_backend_counts).issubset({"pytorch"})
    )

    return {
        "record_count": len(records),
        "active_record_count": len(active_records),
        "active_status_counts": dict(sorted(active_status_counts.items())),
        "active_training_backend_counts": backend_counts,
        "active_training_backend_policy_ids": backend_policy_ids,
        "active_numpy_training_backend_count": active_training_backend_counts.get("numpy", 0),
        "active_numpy_training_backend_policy_ids": backend_policy_ids.get("numpy", []),
        "active_legacy_compat_artifact_count": len(legacy_compat_policy_ids),
        "active_legacy_compat_policy_ids": legacy_compat_policy_ids,
        "active_deprecated_momentum_artifact_count": len(deprecated_momentum_policy_ids),
        "active_deprecated_momentum_policy_ids": deprecated_momentum_policy_ids,
        "ready_to_close_numpy_continuity_window": ready_to_close_numpy_continuity_window,
        "ready_to_retire_legacy_compat_window": len(legacy_compat_policy_ids) == 0,
    }
