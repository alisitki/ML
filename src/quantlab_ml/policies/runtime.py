from __future__ import annotations

from quantlab_ml.common import hash_payload
from quantlab_ml.contracts import (
    DYNAMIC_TARGET_ASSET,
    ExecutionIntent,
    InferenceArtifactExport,
    NumericBand,
    ObservationContext,
    PolicyArtifact,
    PolicyScore,
)
from quantlab_ml.models import (
    LinearPolicyModel,
    LinearPolicyParameters,
    MomentumBaselineModel,
    MomentumBaselineParameters,
    RuntimeDecision,
)


class PolicyRuntimeBridge:
    def decide(self, artifact: PolicyArtifact, observation: ObservationContext) -> RuntimeDecision:
        self._validate_artifact_compatibility(artifact, observation)
        runtime_adapter = artifact.policy_payload.runtime_adapter
        if runtime_adapter == "momentum-baseline-v1":
            momentum_params = MomentumBaselineParameters.model_validate_json(artifact.policy_payload.blob)
            return MomentumBaselineModel(momentum_params).decide(observation, artifact.action_space)
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
    ) -> None:
        metadata = artifact.runtime_metadata
        if metadata.runtime_adapter != artifact.policy_payload.runtime_adapter:
            raise ValueError("runtime metadata must match policy payload runtime adapter")
        if artifact.target_asset != DYNAMIC_TARGET_ASSET and observation.target_symbol != artifact.target_asset:
            raise ValueError("artifact target_asset is incompatible with observation target_symbol")

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

        available_scales = {scale.label for scale in observation_schema.scale_axis}
        missing_scales = [label for label in metadata.required_scale_preset if label not in available_scales]
        if missing_scales:
            raise ValueError(f"observation schema missing required scale preset labels: {missing_scales}")

        unavailable_venues = [venue for venue in metadata.allowed_venues if venue not in observation_schema.exchange_axis]
        if unavailable_venues:
            raise ValueError(f"observation schema missing allowed venues: {unavailable_venues}")

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
