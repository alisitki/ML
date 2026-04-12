from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from quantlab_ml.contracts.common import QuantBaseModel, TimeRange


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
