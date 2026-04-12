from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, model_validator

from quantlab_ml.contracts.common import LineagePointer, NumericBand, QuantBaseModel
from quantlab_ml.contracts.learning_surface import ActionSpaceSpec, ObservationSchema, ScaleSpec

LEGACY_POLICY_ARTIFACT_SCHEMA_VERSION = "policy_artifact_v1"
POLICY_ARTIFACT_SCHEMA_VERSION = "policy_artifact_v2"
OBSERVATION_SCHEMA_VERSION = "observation_schema_v1"
ACTION_SPACE_VERSION = "action_space_v1"
EXECUTION_INTENT_SCHEMA_VERSION = "execution_intent_v1"
STRICT_RUNTIME_CONTRACT_VERSION = "runtime_contract_v1"
DERIVED_CHANNEL_TARGET_PLACEHOLDER = "__target_symbol__"
DYNAMIC_TARGET_ASSET = "__dynamic_target_symbol__"


class OpaquePolicyPayload(QuantBaseModel):
    runtime_adapter: str
    payload_format: str = "json"
    payload_format_version: str = "json-v1"
    blob: str
    digest: str


class DerivedChannelTemplate(QuantBaseModel):
    key_template: str
    shape: list[int]
    skip_if_target_symbol_equals: str | None = None

    def resolve_key(self, target_symbol: str) -> str | None:
        if self.skip_if_target_symbol_equals == target_symbol:
            return None
        return self.key_template.replace(DERIVED_CHANNEL_TARGET_PLACEHOLDER, target_symbol)


class StrictRuntimeContract(QuantBaseModel):
    runtime_contract_version: str = STRICT_RUNTIME_CONTRACT_VERSION
    policy_kind: str
    required_scale_specs: list[ScaleSpec]
    required_raw_surface_shapes: dict[str, list[int]]
    derived_contract_version: str
    derived_channel_templates: list[DerivedChannelTemplate] = Field(default_factory=list)
    derived_channel_template_signature: str
    expected_feature_dim: int

    @model_validator(mode="after")
    def validate_contract(self) -> "StrictRuntimeContract":
        if not self.required_scale_specs:
            raise ValueError("strict runtime contract must define required_scale_specs")
        if self.expected_feature_dim <= 0:
            raise ValueError("strict runtime contract expected_feature_dim must be positive")
        return self


class RuntimeMetadata(QuantBaseModel):
    target_asset: str
    allowed_venues: list[str]
    action_space_version: str = ACTION_SPACE_VERSION
    required_streams: list[str] = Field(default_factory=list)
    required_field_families: dict[str, list[str]] = Field(default_factory=dict)
    required_scale_preset: list[str] = Field(default_factory=list)
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION
    reward_version: str
    policy_state_requirements: list[str] = Field(default_factory=list)
    expected_return_score: float
    risk_score: float
    turnover_score: float
    confidence_or_quality_score: float
    min_capital_requirement: float
    size_bounds: NumericBand
    leverage_bounds: NumericBand
    artifact_compatibility_tags: list[str] = Field(default_factory=list)
    runtime_adapter: str
    strict_runtime_contract: StrictRuntimeContract | None = None
    required_context: dict[str, Any] = Field(default_factory=dict)
    lineage_pointer: LineagePointer

    @model_validator(mode="after")
    def validate_runtime_metadata(self) -> "RuntimeMetadata":
        if not self.allowed_venues:
            raise ValueError("runtime_metadata.allowed_venues must not be empty")
        return self


class PolicyArtifact(QuantBaseModel):
    schema_version: str = POLICY_ARTIFACT_SCHEMA_VERSION
    policy_id: str
    artifact_id: str
    artifact_version: str
    policy_family: str
    training_snapshot_id: str
    training_config_hash: str
    code_commit_hash: str
    reward_version: str
    evaluation_surface_id: str
    target_asset: str
    allowed_venues: list[str]
    allowed_action_family: list[str]
    required_context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    observation_schema: ObservationSchema
    action_space: ActionSpaceSpec
    policy_payload: OpaquePolicyPayload
    runtime_metadata: RuntimeMetadata
    training_run_id: str
    parent_artifact_id: str | None = None
    evaluation_report_id: str | None = None
    paper_sim_report_id: str | None = None
    training_summary: dict[str, Any] = Field(default_factory=dict)

    @property
    def executor_metadata(self) -> RuntimeMetadata:
        return self.runtime_metadata

    @model_validator(mode="after")
    def validate_contract_alignment(self) -> "PolicyArtifact":
        if not self.allowed_venues:
            raise ValueError("allowed_venues must not be empty")
        if not self.allowed_action_family:
            raise ValueError("allowed_action_family must not be empty")
        if self.reward_version != self.runtime_metadata.reward_version:
            raise ValueError("policy artifact reward_version must match runtime_metadata.reward_version")
        if self.target_asset != self.runtime_metadata.target_asset:
            raise ValueError("policy artifact target_asset must match runtime_metadata.target_asset")
        if self.allowed_venues != self.runtime_metadata.allowed_venues:
            raise ValueError("policy artifact allowed_venues must match runtime_metadata.allowed_venues")
        if self.required_context != self.runtime_metadata.required_context:
            raise ValueError("policy artifact required_context must match runtime_metadata.required_context")
        if (
            self.schema_version == POLICY_ARTIFACT_SCHEMA_VERSION
            and self.runtime_metadata.strict_runtime_contract is None
        ):
            raise ValueError("policy artifact v2 requires runtime_metadata.strict_runtime_contract")
        if self.runtime_metadata.strict_runtime_contract is not None:
            strict_contract = self.runtime_metadata.strict_runtime_contract
            if strict_contract.policy_kind != self.policy_payload.runtime_adapter:
                raise ValueError("strict runtime contract policy_kind must match policy payload runtime adapter")
            if strict_contract.policy_kind != self.runtime_metadata.runtime_adapter:
                raise ValueError("strict runtime contract policy_kind must match runtime metadata adapter")
        return self


class InferenceArtifactExport(QuantBaseModel):
    schema_version: str = POLICY_ARTIFACT_SCHEMA_VERSION
    policy_id: str
    artifact_id: str
    created_at: datetime
    runtime_adapter: str
    policy_payload: OpaquePolicyPayload
    runtime_metadata: RuntimeMetadata
    score_summary: dict[str, float] = Field(default_factory=dict)

    @property
    def executor_metadata(self) -> RuntimeMetadata:
        return self.runtime_metadata


class ExecutionIntent(QuantBaseModel):
    schema_version: str = EXECUTION_INTENT_SCHEMA_VERSION
    intent_id: str
    policy_id: str
    artifact_id: str
    decision_timestamp: datetime
    target_asset: str
    venue: str
    action: str
    notional_or_size: float
    leverage: float
    ttl_seconds: int
    confidence_or_score: float
    selector_trace_id: str
    size_band_key: str | None = None
    leverage_band_key: str | None = None

    @model_validator(mode="after")
    def validate_intent(self) -> "ExecutionIntent":
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if not self.target_asset:
            raise ValueError("target_asset must not be empty")
        if not self.venue:
            raise ValueError("venue must not be empty")
        if self.action != "abstain" and self.notional_or_size <= 0.0:
            raise ValueError("directional execution intent requires positive notional_or_size")
        if self.action != "abstain" and self.leverage <= 0.0:
            raise ValueError("directional execution intent requires positive leverage")
        if self.action == "abstain" and self.notional_or_size < 0.0:
            raise ValueError("abstain notional_or_size must be non-negative")
        return self


ExecutorMetadata = RuntimeMetadata
ExecutorPolicyExport = InferenceArtifactExport
