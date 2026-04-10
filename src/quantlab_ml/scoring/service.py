from __future__ import annotations

from quantlab_ml.common import utcnow
from quantlab_ml.contracts import EvaluationReport, PolicyScore


class PolicyScorer:
    def score(self, report: EvaluationReport) -> PolicyScore:
        total_steps = max(report.total_steps, 1)
        expected_return_score = report.average_net_return
        risk_score = max(0.0, 1.0 - abs(report.risk_penalty_total) / total_steps)
        turnover_score = max(0.0, 1.0 - abs(report.turnover_penalty_total) / total_steps)
        stability_score = 1.0 / (1.0 + report.step_reward_std)
        applicability_score = 1.0 - (report.infeasible_action_count / total_steps)
        composite_rank = (
            expected_return_score
            + (0.2 * risk_score)
            + (0.2 * turnover_score)
            + (0.2 * stability_score)
            + (0.2 * applicability_score)
        )
        return PolicyScore(
            policy_id=report.policy_id,
            evaluation_id=report.evaluation_id,
            created_at=utcnow(),
            expected_return_score=expected_return_score,
            risk_score=risk_score,
            turnover_score=turnover_score,
            stability_score=stability_score,
            applicability_score=applicability_score,
            composite_rank=composite_rank,
        )
