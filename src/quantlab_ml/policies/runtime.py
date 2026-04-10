from __future__ import annotations

from quantlab_ml.contracts import ExecutorPolicyExport, ObservationContext, PolicyArtifact, PolicyScore
from quantlab_ml.models import MomentumBaselineModel, MomentumBaselineParameters, RuntimeDecision


class PolicyRuntimeBridge:
    def decide(self, artifact: PolicyArtifact, observation: ObservationContext) -> RuntimeDecision:
        if artifact.policy_payload.runtime_adapter != "momentum-baseline-v1":
            raise ValueError(f"unsupported runtime adapter: {artifact.policy_payload.runtime_adapter}")
        parameters = MomentumBaselineParameters.model_validate_json(artifact.policy_payload.blob)
        model = MomentumBaselineModel(parameters)
        return model.decide(observation, artifact.action_space)

    def export(self, artifact: PolicyArtifact, score: PolicyScore | None) -> ExecutorPolicyExport:
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
        return ExecutorPolicyExport(
            policy_id=artifact.policy_id,
            created_at=artifact.created_at,
            runtime_adapter=artifact.policy_payload.runtime_adapter,
            policy_payload=artifact.policy_payload,
            executor_metadata=artifact.executor_metadata,
            score_summary=summary,
        )
