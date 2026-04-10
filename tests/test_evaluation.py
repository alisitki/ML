from __future__ import annotations

from quantlab_ml.contracts import (
    ActionFeasibilitySurface,
    EvaluationBoundary,
    FeasibilityCell,
    RewardContext,
    RewardTimeline,
    VenueExecutionRef,
)
from quantlab_ml.evaluation import EvaluationEngine
from quantlab_ml.models import RuntimeDecision
from quantlab_ml.rewards import RewardEngine


def test_evaluation_respects_boundary(trajectory_bundle, policy_artifact, evaluation_boundary: EvaluationBoundary):
    report = EvaluationEngine(evaluation_boundary).evaluate(trajectory_bundle, policy_artifact)

    assert report.boundary.fill_assumption_mode == "next_mark_price"
    assert report.boundary.infeasible_action_treatment == "force_abstain"
    assert report.boundary.fill_assumption_mode == trajectory_bundle.reward_spec.timestamping
    assert report.total_steps == sum(len(item.steps) for item in trajectory_bundle.splits["eval"])
    assert set(report.action_counts) >= {"abstain", "enter_long", "enter_short"}
    assert report.realized_trade_count >= 0


def _make_infeasible_feasibility(action_space) -> ActionFeasibilitySurface:
    """abstain=True, directional=False olan feasibility surface."""
    surface = {}
    for action in action_space.actions:
        surface[action.key] = {}
        feasible = action.key == "abstain"
        for venue in ["binance", "bybit", "okx"]:
            surface[action.key][venue] = {
                "micro": {
                    "low": FeasibilityCell(feasible=feasible, reason="" if feasible else "forced_infeasible"),
                    "medium": FeasibilityCell(feasible=feasible, reason="" if feasible else "forced_infeasible"),
                },
                "small": {
                    "low": FeasibilityCell(feasible=feasible, reason="" if feasible else "forced_infeasible"),
                    "medium": FeasibilityCell(feasible=feasible, reason="" if feasible else "forced_infeasible"),
                },
            }
    return ActionFeasibilitySurface(surface=surface)


def test_evaluation_force_abstain_keeps_infeasible_penalty(
    trajectory_bundle,
    policy_artifact,
    evaluation_boundary: EvaluationBoundary,
):
    reward_engine = RewardEngine(trajectory_bundle.reward_spec, trajectory_bundle.action_space)
    eval_trajectory = trajectory_bundle.splits["eval"][0].model_copy(deep=True)

    infeasible_feasibility = _make_infeasible_feasibility(trajectory_bundle.action_space)

    for step in eval_trajectory.steps:
        # Replace feasibility with all-infeasible surface (except abstain)
        object.__setattr__(step, "action_feasibility", infeasible_feasibility)
        # Rebuild reward_snapshot using V2 API
        new_snapshot = reward_engine.build_snapshot(
            event_time=step.event_time,
            reward_context=step.reward_context,
            reward_timeline=step.reward_timeline,
            action_feasibility=infeasible_feasibility,
        )
        object.__setattr__(step, "reward_snapshot", new_snapshot)

    trajectory_bundle.splits["eval"] = [eval_trajectory]

    engine = EvaluationEngine(evaluation_boundary)
    engine.runtime_bridge = _AlwaysLongBridge()
    report = engine.evaluate(trajectory_bundle, policy_artifact)

    assert report.infeasible_action_count == len(eval_trajectory.steps)
    assert report.infeasible_penalty_total == (
        trajectory_bundle.reward_spec.infeasible_action_penalty * len(eval_trajectory.steps)
    )
    assert report.total_net_return == report.infeasible_penalty_total
    assert report.action_counts["abstain"] == len(eval_trajectory.steps)
    assert report.realized_trade_count == 0


class _AlwaysLongBridge:
    def decide(self, artifact, observation) -> RuntimeDecision:
        return RuntimeDecision(action_key="enter_long")
