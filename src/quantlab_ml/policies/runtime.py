from __future__ import annotations

import logging

from quantlab_ml.common import hash_payload
from quantlab_ml.contracts import (
    DYNAMIC_TARGET_ASSET,
    ExecutionIntent,
    InferenceArtifactExport,
    LEGACY_POLICY_ARTIFACT_SCHEMA_VERSION,
    NumericBand,
    ObservationContext,
    OBSERVATION_SCHEMA_VERSION,
    POLICY_ARTIFACT_SCHEMA_VERSION,
    PolicyArtifact,
    PolicyScore,
    STRICT_RUNTIME_CONTRACT_VERSION,
    StrictRuntimeContract,
)
from quantlab_ml.models import (
    LinearPolicyModel,
    LinearPolicyParameters,
    RuntimeDecision,
)
from quantlab_ml.models.features import observation_feature_vector
from quantlab_ml.runtime_contract import (
    build_strict_runtime_contract,
    canonical_raw_surface_shapes,
    resolve_derived_channel_templates,
    scale_specs_match,
)

logger = logging.getLogger(__name__)


class PolicyRuntimeBridge:
    def __init__(self) -> None:
        self._legacy_warning_artifact_ids: set[str] = set()

    def decide(self, artifact: PolicyArtifact, observation: ObservationContext) -> RuntimeDecision:
        self._validate_artifact_compatibility(artifact, observation)
        runtime_adapter = artifact.policy_payload.runtime_adapter
        if runtime_adapter == "momentum-baseline-v1":
            raise ValueError(
                "artifact runtime adapter 'momentum-baseline-v1' is deprecated and unsupported by the active "
                "runtime contract; retrain or re-export under 'linear-policy-v1'"
            )
        if runtime_adapter == "linear-policy-v1":
            linear_params = LinearPolicyParameters.model_validate_json(artifact.policy_payload.blob)
            return LinearPolicyModel(linear_params).decide(observation, artifact.action_space)
        raise ValueError(f"unsupported runtime adapter: {runtime_adapter}")

    def export(self, artifact: PolicyArtifact, score: PolicyScore | None) -> InferenceArtifactExport:
        summary: dict[str, float] = {}
        if score is not None:
            summary = {
                "expected_return_score": score.expected_return_score,
                "risk_score": score.risk_score,
                "turnover_score": score.turnover_score,
                "stability_score": score.stability_score,
                "applicability_score": score.applicability_score,
                "composite_rank": score.composite_rank,
            }
        return InferenceArtifactExport(
            policy_id=artifact.policy_id,
            artifact_id=artifact.artifact_id,
            created_at=artifact.created_at,
            runtime_adapter=artifact.policy_payload.runtime_adapter,
            policy_payload=artifact.policy_payload,
            runtime_metadata=artifact.runtime_metadata,
            score_summary=summary,
        )

    def build_execution_intent(
        self,
        artifact: PolicyArtifact,
        observation: ObservationContext,
        ttl_seconds: int = 60,
        selector_trace_id: str | None = None,
    ) -> ExecutionIntent:
        decision = self.decide(artifact, observation)
        action = decision.action_key
        venue = self._resolve_venue(artifact, decision)
        notional_or_size = 0.0
        leverage = 0.0
        if action != "abstain":
            size_band = self._resolve_band(artifact.action_space.size_bands, decision.size_band_key, "size_band_key")
            leverage_band = self._resolve_band(
                artifact.action_space.leverage_bands,
                decision.leverage_band_key,
                "leverage_band_key",
            )
            notional_or_size = size_band.lower
            leverage = leverage_band.lower

        resolved_selector_trace = selector_trace_id or self._selector_trace_id(artifact, observation, decision)
        return ExecutionIntent(
            intent_id=self._intent_id(artifact, observation, decision, resolved_selector_trace),
            policy_id=artifact.policy_id,
            artifact_id=artifact.artifact_id,
            decision_timestamp=observation.as_of,
            target_asset=observation.target_symbol,
            venue=venue,
            action=action,
            notional_or_size=notional_or_size,
            leverage=leverage,
            ttl_seconds=ttl_seconds,
            confidence_or_score=decision.confidence,
            selector_trace_id=resolved_selector_trace,
            size_band_key=decision.size_band_key,
            leverage_band_key=decision.leverage_band_key,
        )

    def _validate_artifact_compatibility(
        self,
        artifact: PolicyArtifact,
        observation: ObservationContext,
    ) -> StrictRuntimeContract:
        metadata = artifact.runtime_metadata
        if metadata.runtime_adapter != artifact.policy_payload.runtime_adapter:
            raise ValueError("runtime metadata must match policy payload runtime adapter")
        if metadata.runtime_adapter == "momentum-baseline-v1":
            raise ValueError(
                "artifact runtime adapter 'momentum-baseline-v1' is deprecated and unsupported by the active "
                "runtime contract; retrain or re-export under 'linear-policy-v1'"
            )
        if artifact.target_asset != DYNAMIC_TARGET_ASSET and observation.target_symbol != artifact.target_asset:
            raise ValueError("artifact target_asset is incompatible with observation target_symbol")

        strict_contract = self._resolved_runtime_contract(artifact)
        self._validate_schema_versions(artifact, observation)
        self._validate_shape_and_scale_contract(strict_contract, observation)
        self._validate_stream_and_field_requirements(artifact, observation)
        self._validate_venue_requirements(artifact, observation)
        self._validate_derived_contract(strict_contract, observation)
        self._validate_feature_dimension(strict_contract, artifact, observation)
        return strict_contract

    def _resolved_runtime_contract(self, artifact: PolicyArtifact) -> StrictRuntimeContract:
        if artifact.schema_version == POLICY_ARTIFACT_SCHEMA_VERSION:
            strict_contract = artifact.runtime_metadata.strict_runtime_contract
            if strict_contract is None:
                raise ValueError(
                    "artifact schema_version 'policy_artifact_v2' requires runtime_metadata.strict_runtime_contract"
                )
            self._validate_strict_contract_header(artifact, strict_contract)
            return strict_contract
        if artifact.schema_version == LEGACY_POLICY_ARTIFACT_SCHEMA_VERSION:
            return self._resolve_legacy_runtime_contract(artifact)
        raise ValueError(f"unsupported policy artifact schema_version: {artifact.schema_version}")

    def _validate_strict_contract_header(
        self,
        artifact: PolicyArtifact,
        strict_contract: StrictRuntimeContract,
    ) -> None:
        if strict_contract.runtime_contract_version != STRICT_RUNTIME_CONTRACT_VERSION:
            raise ValueError(
                "unsupported runtime contract version: "
                f"{strict_contract.runtime_contract_version}"
            )
        if strict_contract.policy_kind != artifact.policy_payload.runtime_adapter:
            raise ValueError("strict runtime contract policy_kind must match policy payload runtime adapter")
        if strict_contract.policy_kind != artifact.runtime_metadata.runtime_adapter:
            raise ValueError("strict runtime contract policy_kind must match runtime metadata adapter")

    def _resolve_legacy_runtime_contract(self, artifact: PolicyArtifact) -> StrictRuntimeContract:
        runtime_adapter = artifact.policy_payload.runtime_adapter
        if runtime_adapter != "linear-policy-v1":
            raise ValueError(
                "legacy compat window only supports 'linear-policy-v1' artifacts; "
                f"got {runtime_adapter!r}"
            )
        reconstructed = build_strict_runtime_contract(
            artifact.observation_schema,
            policy_kind=runtime_adapter,
        )
        payload_feature_dim = self._payload_feature_dim(artifact)
        if payload_feature_dim != reconstructed.expected_feature_dim:
            raise ValueError(
                "legacy artifact feature dimension is incompatible with the reconstructed runtime contract: "
                f"payload_feature_dim={payload_feature_dim}, expected_feature_dim={reconstructed.expected_feature_dim}"
            )
        self._validate_legacy_metadata_consistency(artifact, reconstructed)
        self._warn_legacy_acceptance(artifact, reconstructed)
        return reconstructed

    def _validate_legacy_metadata_consistency(
        self,
        artifact: PolicyArtifact,
        strict_contract: StrictRuntimeContract,
    ) -> None:
        metadata = artifact.runtime_metadata
        expected_scale_labels = [scale.label for scale in strict_contract.required_scale_specs]
        if metadata.observation_schema_version and metadata.observation_schema_version != artifact.observation_schema.schema_version:
            raise ValueError(
                "legacy artifact observation_schema_version does not match embedded observation schema: "
                f"{metadata.observation_schema_version!r} vs {artifact.observation_schema.schema_version!r}"
            )
        if metadata.required_scale_preset and metadata.required_scale_preset != expected_scale_labels:
            raise ValueError(
                "legacy artifact required_scale_preset is inconsistent with the embedded observation schema: "
                f"{metadata.required_scale_preset!r} vs {expected_scale_labels!r}"
            )
        expected_streams = list(artifact.observation_schema.stream_axis)
        if metadata.required_streams and metadata.required_streams != expected_streams:
            raise ValueError(
                "legacy artifact required_streams are inconsistent with the embedded observation schema: "
                f"{metadata.required_streams!r} vs {expected_streams!r}"
            )
        expected_field_families = {
            stream: artifact.observation_schema.field_axis.get(stream, [])
            for stream in artifact.observation_schema.stream_axis
        }
        if metadata.required_field_families and metadata.required_field_families != expected_field_families:
            raise ValueError(
                "legacy artifact required_field_families are inconsistent with the embedded observation schema"
            )

    def _warn_legacy_acceptance(
        self,
        artifact: PolicyArtifact,
        strict_contract: StrictRuntimeContract,
    ) -> None:
        if artifact.artifact_id in self._legacy_warning_artifact_ids:
            return
        self._legacy_warning_artifact_ids.add(artifact.artifact_id)
        missing_fields = ["runtime_metadata.strict_runtime_contract"]
        logger.warning(
            "legacy artifact accepted via compat window artifact_id=%s missing_metadata=%s "
            "inferred_runtime_contract_version=%s inferred_feature_dim=%d deprecation=temporary_compat_window",
            artifact.artifact_id,
            ",".join(missing_fields),
            strict_contract.runtime_contract_version,
            strict_contract.expected_feature_dim,
        )

    def _validate_schema_versions(
        self,
        artifact: PolicyArtifact,
        observation: ObservationContext,
    ) -> None:
        metadata = artifact.runtime_metadata
        if artifact.observation_schema.schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported artifact observation schema version: {artifact.observation_schema.schema_version}"
            )
        if metadata.observation_schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported runtime metadata observation schema version: {metadata.observation_schema_version}"
            )
        if observation.observation_schema.schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported observation schema version: {observation.observation_schema.schema_version}"
            )
        if metadata.observation_schema_version != observation.observation_schema.schema_version:
            raise ValueError(
                "observation schema version mismatch between artifact runtime metadata and observation: "
                f"{metadata.observation_schema_version!r} vs {observation.observation_schema.schema_version!r}"
            )

    def _validate_shape_and_scale_contract(
        self,
        strict_contract: StrictRuntimeContract,
        observation: ObservationContext,
    ) -> None:
        observation_schema = observation.observation_schema
        if not scale_specs_match(strict_contract.required_scale_specs, observation_schema.scale_axis):
            raise ValueError("observation scale spec does not match artifact runtime contract")
        expected_raw_shapes = canonical_raw_surface_shapes(observation_schema)
        if strict_contract.required_raw_surface_shapes != expected_raw_shapes:
            raise ValueError("observation raw surface shape contract does not match artifact runtime contract")
        actual_raw_shapes = {
            scale_label: list(tensor.shape)
            for scale_label, tensor in observation.raw_surface.items()
        }
        if strict_contract.required_raw_surface_shapes != actual_raw_shapes:
            raise ValueError(
                "observation raw surface shapes do not match artifact runtime contract: "
                f"expected={strict_contract.required_raw_surface_shapes}, got={actual_raw_shapes}"
            )

    def _validate_stream_and_field_requirements(
        self,
        artifact: PolicyArtifact,
        observation: ObservationContext,
    ) -> None:
        metadata = artifact.runtime_metadata
        observation_schema = observation.observation_schema
        available_streams = set(observation_schema.stream_axis)
        missing_streams = [stream for stream in metadata.required_streams if stream not in available_streams]
        if missing_streams:
            raise ValueError(f"observation schema missing required streams: {missing_streams}")

        missing_fields: dict[str, list[str]] = {}
        for stream, required_fields in metadata.required_field_families.items():
            available_fields = set(observation_schema.field_axis.get(stream, []))
            absent = [field for field in required_fields if field not in available_fields]
            if absent:
                missing_fields[stream] = absent
        if missing_fields:
            raise ValueError(f"observation schema missing required field families: {missing_fields}")

    def _validate_venue_requirements(
        self,
        artifact: PolicyArtifact,
        observation: ObservationContext,
    ) -> None:
        metadata = artifact.runtime_metadata
        observation_schema = observation.observation_schema
        unavailable_venues = [venue for venue in metadata.allowed_venues if venue not in observation_schema.exchange_axis]
        if unavailable_venues:
            raise ValueError(f"observation schema missing allowed venues: {unavailable_venues}")

    def _validate_derived_contract(
        self,
        strict_contract: StrictRuntimeContract,
        observation: ObservationContext,
    ) -> None:
        expected_contract = build_strict_runtime_contract(
            observation.observation_schema,
            policy_kind=strict_contract.policy_kind,
        )
        if strict_contract.derived_contract_version != expected_contract.derived_contract_version:
            raise ValueError("derived contract version mismatch between artifact and observation schema")
        if strict_contract.derived_channel_template_signature != expected_contract.derived_channel_template_signature:
            raise ValueError("derived channel template signature mismatch between artifact and observation schema")
        if strict_contract.derived_channel_templates != expected_contract.derived_channel_templates:
            raise ValueError("derived channel templates mismatch between artifact and observation schema")

        expected_channels = resolve_derived_channel_templates(
            strict_contract.derived_channel_templates,
            target_symbol=observation.target_symbol,
        )
        actual_surface = observation.derived_surface
        if expected_channels and actual_surface is None:
            raise ValueError("observation is missing derived_surface required by the artifact runtime contract")
        if actual_surface is None:
            return
        if actual_surface.contract_version != strict_contract.derived_contract_version:
            raise ValueError(
                "derived contract version mismatch between artifact and observation: "
                f"{strict_contract.derived_contract_version!r} vs {actual_surface.contract_version!r}"
            )
        actual_keys = [channel.key for channel in actual_surface.channels]
        expected_keys = [template.key_template for template in expected_channels]
        if actual_keys != expected_keys:
            raise ValueError(
                "derived channel identity/order mismatch: "
                f"expected={expected_keys}, got={actual_keys}"
            )
        for actual_channel, expected_channel in zip(actual_surface.channels, expected_channels, strict=True):
            if actual_channel.shape != expected_channel.shape:
                raise ValueError(
                    f"derived channel shape mismatch for {actual_channel.key!r}: "
                    f"expected={expected_channel.shape}, got={actual_channel.shape}"
                )

    def _validate_feature_dimension(
        self,
        strict_contract: StrictRuntimeContract,
        artifact: PolicyArtifact,
        observation: ObservationContext,
    ) -> None:
        payload_feature_dim = self._payload_feature_dim(artifact)
        if payload_feature_dim != strict_contract.expected_feature_dim:
            raise ValueError(
                "artifact payload feature dimension does not match runtime contract: "
                f"payload_feature_dim={payload_feature_dim}, expected_feature_dim={strict_contract.expected_feature_dim}"
            )
        actual_feature_dim = len(observation_feature_vector(observation))
        if actual_feature_dim != strict_contract.expected_feature_dim:
            raise ValueError(
                "observation feature dimension does not match runtime contract: "
                f"observation_feature_dim={actual_feature_dim}, expected_feature_dim={strict_contract.expected_feature_dim}"
            )

    def _payload_feature_dim(self, artifact: PolicyArtifact) -> int:
        runtime_adapter = artifact.policy_payload.runtime_adapter
        if runtime_adapter != "linear-policy-v1":
            raise ValueError(
                f"runtime adapter {runtime_adapter!r} is not supported by the active strict runtime contract"
            )
        linear_params = LinearPolicyParameters.model_validate_json(artifact.policy_payload.blob)
        return len(linear_params.feature_mean)

    def _resolve_venue(self, artifact: PolicyArtifact, decision: RuntimeDecision) -> str:
        if decision.venue is not None:
            if decision.venue not in artifact.allowed_venues:
                raise ValueError(f"decision venue '{decision.venue}' is not allowed by the artifact")
            return decision.venue
        if artifact.allowed_venues:
            return artifact.allowed_venues[0]
        raise ValueError("artifact allowed_venues must not be empty")

    def _resolve_band(
        self,
        bands: list[NumericBand],
        band_key: str | None,
        field_name: str,
    ) -> NumericBand:
        if band_key is None:
            raise ValueError(f"directional execution intent requires explicit {field_name}")
        for band in bands:
            if band.key == band_key:
                return band
        raise ValueError(f"unknown {field_name}: {band_key}")

    def _selector_trace_id(
        self,
        artifact: PolicyArtifact,
        observation: ObservationContext,
        decision: RuntimeDecision,
    ) -> str:
        trace_hash = hash_payload(
            {
                "artifact_id": artifact.artifact_id,
                "decision_timestamp": observation.as_of.isoformat(),
                "target_symbol": observation.target_symbol,
                "action_key": decision.action_key,
                "venue": decision.venue,
            }
        )
        return f"selector-{trace_hash[:12]}"

    def _intent_id(
        self,
        artifact: PolicyArtifact,
        observation: ObservationContext,
        decision: RuntimeDecision,
        selector_trace_id: str,
    ) -> str:
        intent_hash = hash_payload(
            {
                "policy_id": artifact.policy_id,
                "artifact_id": artifact.artifact_id,
                "decision_timestamp": observation.as_of.isoformat(),
                "target_symbol": observation.target_symbol,
                "action_key": decision.action_key,
                "selector_trace_id": selector_trace_id,
            }
        )
        return f"intent-{intent_hash[:12]}"
