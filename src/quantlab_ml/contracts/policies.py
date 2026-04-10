from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from quantlab_ml.contracts.common import LineagePointer, NumericBand, QuantBaseModel
from quantlab_ml.contracts.learning_surface import ActionSpaceSpec, ObservationSchema


class OpaquePolicyPayload(QuantBaseModel):
    runtime_adapter: str
    payload_format: str = "json"
    blob: str
    digest: str


class ExecutorMetadata(QuantBaseModel):
    asset_universe: list[str]
    venue_compatibility: list[str]
    instrument_compatibility: list[str]
    min_capital_requirement: float
    size_bounds: NumericBand
    leverage_bounds: NumericBand
    liquidity_flags: dict[str, bool]
    applicability_flags: dict[str, bool]
    expected_return: float
    risk_score: float
    turnover_score: float
    confidence_score: float
    artifact_version: str
    lineage_pointer: LineagePointer


class PolicyArtifact(QuantBaseModel):
    policy_id: str
    created_at: datetime
    observation_schema: ObservationSchema
    action_space: ActionSpaceSpec
    policy_payload: OpaquePolicyPayload
    executor_metadata: ExecutorMetadata
    training_summary: dict[str, Any] = Field(default_factory=dict)


class ExecutorPolicyExport(QuantBaseModel):
    policy_id: str
    created_at: datetime
    runtime_adapter: str
    policy_payload: OpaquePolicyPayload
    executor_metadata: ExecutorMetadata
    score_summary: dict[str, float] = Field(default_factory=dict)
