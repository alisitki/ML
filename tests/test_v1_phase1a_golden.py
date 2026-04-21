from __future__ import annotations

import json

from quantlab_ml.evaluation import EvaluationEngine
from quantlab_ml.policies import PolicyRuntimeBridge


def test_v1_phase1a_golden_no_regression(repo_root, trajectory_bundle, policy_artifact, evaluation_boundary) -> None:
    golden = json.loads((repo_root / "tests" / "fixtures" / "v1_phase1a_golden.json").read_text(encoding="utf-8"))
    observation = trajectory_bundle.splits["validation"][0].steps[0].observation
    decision = PolicyRuntimeBridge().decide(policy_artifact, observation, policy_state=None)
    report = EvaluationEngine(evaluation_boundary).evaluate(trajectory_bundle, policy_artifact)

    assert decision.model_dump(mode="json") == golden["decision"]
    assert {
        "action_counts": report.action_counts,
        "average_net_return": report.average_net_return,
        "fee_total": report.fee_total,
        "funding_total": report.funding_total,
        "infeasible_action_count": report.infeasible_action_count,
        "realized_trade_count": report.realized_trade_count,
        "risk_penalty_total": report.risk_penalty_total,
        "slippage_total": report.slippage_total,
        "total_net_return": report.total_net_return,
        "total_steps": report.total_steps,
        "turnover_penalty_total": report.turnover_penalty_total,
    } == golden["evaluation_summary"]
