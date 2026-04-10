from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from quantlab_ml.contracts.common import QuantBaseModel, TimeRange
from quantlab_ml.contracts.evaluation import PolicyScore


class CoverageStats(QuantBaseModel):
    train_sample_count: int
    eval_sample_count: int
    covered_symbols: list[str]
    covered_venues: list[str]
    covered_streams: list[str]
    active_date_range: TimeRange
    reward_event_count: int
    realized_trade_count: int


class ScoreSnapshot(QuantBaseModel):
    recorded_at: datetime
    score: PolicyScore


class RegistryRecord(QuantBaseModel):
    policy_id: str
    artifact_path: str
    dataset_hash: str
    slice_id: str
    reward_config_hash: str
    training_config_hash: str
    parent_policy_id: str | None = None
    lineage_chain: list[str] = Field(default_factory=list)
    status: Literal["candidate", "challenger", "champion", "retired"] = "candidate"
    train_window: TimeRange
    eval_window: TimeRange | None = None
    score_history: list[ScoreSnapshot] = Field(default_factory=list)
    coverage: CoverageStats
    created_at: datetime
    updated_at: datetime


class RegistryIndex(QuantBaseModel):
    champion_policy_id: str | None = None
    challenger_policy_ids: list[str] = Field(default_factory=list)
