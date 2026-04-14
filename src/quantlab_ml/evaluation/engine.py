from __future__ import annotations

from collections.abc import Iterable
from statistics import pstdev
from datetime import timedelta

from quantlab_ml.common import utcnow
from quantlab_ml.contracts import (
    DatasetSpec,
    EvaluationBoundary,
    EvaluationReport,
    PolicyArtifact,
    PolicyState,
    RewardEventSpec,
    TimeRange,
    TrajectoryBundle,
    TrajectoryRecord,
)
from quantlab_ml.policies import PolicyRuntimeBridge
from quantlab_ml.rewards import RewardEngine


class EvaluationEngine:
    def __init__(self, boundary: EvaluationBoundary):
        self.boundary = boundary
        self.runtime_bridge = PolicyRuntimeBridge()

    def evaluate(self, bundle: TrajectoryBundle, artifact: PolicyArtifact, split: str = "validation") -> EvaluationReport:
        return self.evaluate_records(
            bundle.dataset_spec,
            bundle.reward_spec,
            bundle.splits[split],
            artifact,
        )

    def evaluate_records(
        self,
        dataset_spec: DatasetSpec,
        reward_spec: RewardEventSpec,
        trajectories: Iterable[TrajectoryRecord],
        artifact: PolicyArtifact,
    ) -> EvaluationReport:
        self._validate_boundary(reward_spec)
        reward_engine = RewardEngine(reward_spec, artifact.action_space)
        rewards: list[float] = []
        total_steps = 0
        realized_trade_count = 0
        infeasible_action_count = 0
        infeasible_penalty_total = 0.0
        risk_penalty_total = 0.0
        turnover_penalty_total = 0.0
        fee_total = 0.0
        funding_total = 0.0
        slippage_total = 0.0
        action_counts = {action.key: 0 for action in artifact.action_space.actions}
        notes: list[str] = []
        first_step_time = None
        last_step_time = None
        saw_trajectory = False

        for trajectory in trajectories:
            saw_trajectory = True
            current_policy_state = PolicyState()
            for step in trajectory.steps:
                total_steps += 1
                if first_step_time is None:
                    first_step_time = step.event_time
                last_step_time = step.event_time
                decision = self.runtime_bridge.decide(artifact, step.observation)
                applied = reward_engine.apply_decision(
                    snapshot=step.reward_snapshot,
                    requested_action_key=decision.action_key,
                    action_feasibility=step.action_feasibility,
                    infeasible_action_treatment=self.boundary.infeasible_action_treatment,
                    venue=decision.venue,
                    size_band_key=decision.size_band_key,
                    leverage_band_key=decision.leverage_band_key,
                    policy_state=current_policy_state,
                )
                if applied.reward_context is not None:
                    step.reward_context = applied.reward_context
                    step.reward_snapshot.context = applied.reward_context
                if applied.infeasible:
                    infeasible_action_count += 1
                    infeasible_penalty_total += applied.infeasible_penalty
                    notes.append(f"infeasible:{decision.action_key}:{step.event_time.isoformat()}")
                rewards.append(applied.net_reward)
                action_counts[applied.applied_action_key] = action_counts.get(applied.applied_action_key, 0) + 1
                risk_penalty_total += applied.risk_penalty
                turnover_penalty_total += applied.turnover_penalty
                fee_total += applied.fee
                funding_total += applied.funding
                slippage_total += applied.slippage
                if applied.applied_action_key != "abstain":
                    realized_trade_count += 1
                current_policy_state = reward_engine.advance_policy_state(current_policy_state, applied)

        if not saw_trajectory or first_step_time is None or last_step_time is None:
            raise ValueError("evaluation trajectories are empty")

        active_range = TimeRange(
            start=first_step_time,
            end=last_step_time + timedelta(seconds=dataset_spec.sampling_interval_seconds),
        )
        total_net_return = sum(rewards)
        return EvaluationReport(
            policy_id=artifact.policy_id,
            evaluation_id=f"eval-{artifact.policy_id}",
            created_at=utcnow(),
            boundary=self.boundary,
            total_steps=total_steps,
            realized_trade_count=realized_trade_count,
            infeasible_action_count=infeasible_action_count,
            infeasible_penalty_total=infeasible_penalty_total,
            total_net_return=total_net_return,
            average_net_return=total_net_return / max(total_steps, 1),
            risk_penalty_total=risk_penalty_total,
            turnover_penalty_total=turnover_penalty_total,
            fee_total=fee_total,
            funding_total=funding_total,
            slippage_total=slippage_total,
            action_counts=action_counts,
            step_reward_std=pstdev(rewards) if len(rewards) > 1 else 0.0,
            coverage_symbols=dataset_spec.symbols,
            coverage_venues=dataset_spec.exchanges,
            coverage_streams=dataset_spec.stream_universe,
            active_date_range=active_range,
            notes=notes,
        )

    def _validate_boundary(self, reward_spec: RewardEventSpec) -> None:
        if self.boundary.fill_assumption_mode != reward_spec.timestamping:
            raise ValueError(
                "evaluation fill_assumption_mode must match reward timestamping for replay parity"
            )
        if self.boundary.fee_handling != "shared_reward_contract":
            raise ValueError("unsupported fee_handling for v1 replay surface")
        if self.boundary.funding_handling != "carry_from_funding_stream":
            raise ValueError("unsupported funding_handling for v1 replay surface")
        if self.boundary.slippage_handling != "fixed_bps":
            raise ValueError("unsupported slippage_handling for v1 replay surface")
        if self.boundary.terminal_semantics != "trajectory_boundary_is_terminal":
            raise ValueError("unsupported terminal_semantics for v1 replay surface")
        if self.boundary.timeout_semantics != "force_terminal_at_data_end":
            raise ValueError("unsupported timeout_semantics for v1 replay surface")
