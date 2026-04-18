from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import Field, model_validator

from quantlab_ml.contracts.common import QuantBaseModel, TimeRange

ContinuityInspectedEvidenceKind = Literal[
    "repo_tracked_artifact",
    "external_retained_evidence",
    "authoritative_evidence",
]
ContinuityAuthorityStatus = Literal["confirmed", "unconfirmed", "unknown"]
ContinuityAuditScopeVerdict = Literal["blocked", "active_dependency_present", "clear_in_inspected_scope"]
ContinuityCloseoutDecision = Literal["RETIRE", "FREEZE", "KEEP-TEMPORARY-WITH-EXPLICIT-SCOPE"]
ContinuityCloseoutDecisionStatus = Literal["pending_authoritative_evidence", "decided"]


class SearchBudgetSummary(QuantBaseModel):
    tried_models: int
    tried_seeds: int
    tried_architectures: int
    tried_reward_variants: int
    tried_hyperparameter_variants: int
    total_candidate_count: int

    @model_validator(mode="after")
    def validate_positive_counts(self) -> "SearchBudgetSummary":
        for field_name in (
            "tried_models",
            "tried_seeds",
            "tried_architectures",
            "tried_reward_variants",
            "tried_hyperparameter_variants",
            "total_candidate_count",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        return self


class CoverageStats(QuantBaseModel):
    train_sample_count: int
    eval_sample_count: int
    covered_symbols: list[str]
    covered_venues: list[str]
    covered_streams: list[str]
    active_date_range: TimeRange
    reward_event_count: int
    realized_trade_count: int
    field_origins: dict[str, Literal["policy-evidence-derived", "dataset-surface-derived"]] = Field(
        default_factory=dict
    )


class SplitEvidence(QuantBaseModel):
    split_version: str
    train_window: TimeRange
    validation_window: TimeRange
    final_untouched_test_window: TimeRange
    purge_width_steps: int
    embargo_width_steps: int
    walkforward_fold_count: int
    evaluation_is_time_ordered: bool = True
    random_split_used: bool = False


class ReproducibilityMetadata(QuantBaseModel):
    data_snapshot_id: str
    code_commit_hash: str
    config_hash: str
    seed: int
    runtime_stack: dict[str, str] = Field(default_factory=dict)
    reproducible_within_tolerance: bool


class ContinuityAuditSummary(QuantBaseModel):
    registry_root: str
    inspected_evidence_kind: ContinuityInspectedEvidenceKind
    authority_status: ContinuityAuthorityStatus
    closeout_decision_allowed: bool
    closeout_blockers: list[str] = Field(default_factory=list)
    record_count: int
    active_record_count: int
    readable_active_artifact_count: int
    active_status_counts: dict[str, int] = Field(default_factory=dict)
    active_training_backend_counts: dict[str, int] = Field(default_factory=dict)
    active_training_backend_policy_ids: dict[str, list[str]] = Field(default_factory=dict)
    active_numpy_training_backend_count: int
    active_numpy_training_backend_policy_ids: list[str] = Field(default_factory=list)
    active_legacy_compat_artifact_count: int
    active_legacy_compat_policy_ids: list[str] = Field(default_factory=list)
    active_deprecated_momentum_artifact_count: int
    active_deprecated_momentum_policy_ids: list[str] = Field(default_factory=list)
    registry_local_fallback_policy_ids: list[str] = Field(default_factory=list)
    artifact_load_failures: list[dict[str, str]] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    audit_scope_verdict: ContinuityAuditScopeVerdict
    ready_to_close_numpy_continuity_window: bool
    ready_to_retire_legacy_compat_window: bool

    @model_validator(mode="after")
    def validate_summary(self) -> "ContinuityAuditSummary":
        if self.inspected_evidence_kind == "authoritative_evidence" and self.authority_status != "confirmed":
            raise ValueError("authoritative_evidence requires authority_status=confirmed")
        if self.closeout_decision_allowed and self.authority_status != "confirmed":
            raise ValueError("closeout_decision_allowed requires authority_status=confirmed")
        return self


class ContinuityCloseoutRecord(QuantBaseModel):
    window_id: str
    scope_kind: ContinuityInspectedEvidenceKind
    authority_status: ContinuityAuthorityStatus
    latest_audit_scope_verdict: ContinuityAuditScopeVerdict
    blocking_reasons: list[str] = Field(default_factory=list)
    next_required_evidence: list[str] = Field(default_factory=list)
    last_reviewed: date
    decision_status: ContinuityCloseoutDecisionStatus
    decision: ContinuityCloseoutDecision | None = None

    @model_validator(mode="after")
    def validate_decision_state(self) -> "ContinuityCloseoutRecord":
        if self.decision_status == "pending_authoritative_evidence":
            if self.decision is not None:
                raise ValueError("pending_authoritative_evidence records must not carry a decision")
            return self
        if self.decision is None:
            raise ValueError("decided records must carry a decision")
        if self.authority_status != "confirmed":
            raise ValueError("decided records require confirmed authority status")
        return self


class PaperSimEvidenceRecord(QuantBaseModel):
    evidence_id: str
    policy_id: str
    artifact_id: str
    created_at: datetime
    evaluation_report_id: str
    evaluation_report_path: str
    comparison_report_id: str | None = None
    report_path: str
    report_format: Literal["markdown", "json", "text", "unknown"] = "unknown"


class PromotionEvidence(QuantBaseModel):
    preprocessing_fit_on_train_only: bool
    no_future_features: bool
    no_future_masks: bool
    no_future_reward_construction: bool
    no_cross_split_contamination: bool
    final_untouched_test_unused_for_selection: bool
    realistic_execution_assumptions: bool
    superiority_not_one_lucky_slice_only: bool
    comparison_report_id: str | None = None
    paper_sim_evidence_id: str
    deployment_artifact_path: str
    runtime_uses_inference_artifact_only: bool
    no_live_learning: bool
    executor_boundary_respected: bool
    selector_boundary_respected: bool
    reproducibility: ReproducibilityMetadata


class ScoreSnapshot(QuantBaseModel):
    recorded_at: datetime
    evaluation_id: str
    evaluation_surface_id: str
    governing_objective: float
    expected_return_score: float
    risk_score: float
    turnover_score: float
    stability_score: float
    applicability_score: float
    composite_rank: float


class RegistryRecord(QuantBaseModel):
    policy_id: str
    artifact_id: str
    artifact_path: str
    status: Literal["candidate", "challenger", "champion", "rejected", "retired", "archived"] = "candidate"
    policy_family: str
    target_asset: str
    training_snapshot_id: str
    training_config_hash: str
    reward_version: str
    evaluation_surface_id: str
    search_budget_summary: SearchBudgetSummary
    artifact_compatibility_tags: list[str] = Field(default_factory=list)
    runtime_compatibility_tags: list[str] = Field(default_factory=list)
    dataset_hash: str
    slice_id: str
    reward_config_hash: str
    parent_policy_id: str | None = None
    lineage_chain: list[str] = Field(default_factory=list)
    train_window: TimeRange
    eval_window: TimeRange | None = None
    score_history: list[ScoreSnapshot] = Field(default_factory=list)
    coverage: CoverageStats
    split_evidence: SplitEvidence
    evaluation_report_path: str | None = None
    evaluation_report_id: str | None = None
    comparison_report_id: str | None = None
    paper_sim_evidence_id: str | None = None
    paper_sim_report_path: str | None = None
    deployment_artifact_path: str | None = None
    runtime_stack: dict[str, str] = Field(default_factory=dict)
    promotion_decision_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class PromotionDecisionRecord(QuantBaseModel):
    decision_id: str
    policy_id: str
    artifact_id: str
    champion_policy_id_before: str | None = None
    champion_policy_id_after: str | None = None
    decision: Literal["promote", "reject"]
    checked_at: datetime
    gate_checks: dict[str, bool] = Field(default_factory=dict)
    failure_reasons: list[str] = Field(default_factory=list)
    comparison_report_id: str | None = None
    paper_sim_evidence_id: str | None = None
    deployment_artifact_path: str | None = None
    evaluation_surface_id: str
    search_budget_summary: SearchBudgetSummary
    reproducibility: ReproducibilityMetadata


class RegistryIndex(QuantBaseModel):
    champion_policy_id: str | None = None
    challenger_policy_ids: list[str] = Field(default_factory=list)
