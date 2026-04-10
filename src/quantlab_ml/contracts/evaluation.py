from __future__ import annotations

from datetime import datetime

from pydantic import Field

from quantlab_ml.contracts.common import QuantBaseModel, TimeRange


class EvaluationBoundary(QuantBaseModel):
    fee_handling: str
    funding_handling: str
    slippage_handling: str
    fill_assumption_mode: str
    timeout_semantics: str
    terminal_semantics: str
    infeasible_action_treatment: str


class EvaluationReport(QuantBaseModel):
    policy_id: str
    evaluation_id: str
    created_at: datetime
    boundary: EvaluationBoundary
    total_steps: int
    realized_trade_count: int
    infeasible_action_count: int
    infeasible_penalty_total: float
    total_net_return: float
    average_net_return: float
    risk_penalty_total: float
    turnover_penalty_total: float
    fee_total: float
    funding_total: float
    slippage_total: float
    action_counts: dict[str, int]
    step_reward_std: float
    coverage_symbols: list[str]
    coverage_venues: list[str]
    coverage_streams: list[str]
    active_date_range: TimeRange
    notes: list[str] = Field(default_factory=list)


class PolicyScore(QuantBaseModel):
    policy_id: str
    evaluation_id: str
    created_at: datetime
    expected_return_score: float
    risk_score: float
    turnover_score: float
    stability_score: float
    applicability_score: float
    composite_rank: float
