from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from quantlab_ml.contracts import ContinuityCloseoutRecord


def test_pending_closeout_record_must_not_carry_a_decision() -> None:
    with pytest.raises(ValueError, match="pending_authoritative_evidence"):
        ContinuityCloseoutRecord(
            window_id="numpy_training_backend",
            scope_kind="external_retained_evidence",
            authority_status="unconfirmed",
            latest_audit_scope_verdict="blocked",
            blocking_reasons=["authoritative_scope_not_confirmed"],
            next_required_evidence=["confirm authoritative scope"],
            last_reviewed=date(2026, 4, 18),
            decision_status="pending_authoritative_evidence",
            decision="RETIRE",
        )


def test_decided_closeout_record_requires_confirmed_authority() -> None:
    with pytest.raises(ValueError, match="confirmed authority"):
        ContinuityCloseoutRecord(
            window_id="legacy_linear_policy_v1_compat",
            scope_kind="external_retained_evidence",
            authority_status="unknown",
            latest_audit_scope_verdict="clear_in_inspected_scope",
            blocking_reasons=[],
            next_required_evidence=[],
            last_reviewed=date(2026, 4, 18),
            decision_status="decided",
            decision="RETIRE",
        )


def test_repo_tracked_closeout_records_remain_pending_until_authoritative_evidence(
    repo_root: Path,
) -> None:
    records_dir = repo_root / "docs" / "continuity_closeout"
    record_paths = sorted(records_dir.glob("*.yaml"))

    assert [path.name for path in record_paths] == [
        "legacy_linear_policy_v1_compat.yaml",
        "numpy_training_backend.yaml",
    ]

    for path in record_paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        record = ContinuityCloseoutRecord.model_validate(payload)

        assert record.decision_status == "pending_authoritative_evidence"
        assert record.decision is None
        assert record.blocking_reasons == ["authoritative_scope_not_confirmed"]
        assert record.next_required_evidence
