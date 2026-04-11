from __future__ import annotations

from statistics import pstdev
from datetime import timedelta

from quantlab_ml.common import utcnow
from quantlab_ml.contracts import EvaluationBoundary, EvaluationReport, PolicyArtifact, PolicyState, TimeRange, TrajectoryBundle
from quantlab_ml.policies import PolicyRuntimeBridge
from quantlab_ml.rewards import RewardEngine


class EvaluationEngine:
    def __init__(self, boundary: EvaluationBoundary):
        self.boundary = boundary
        self.runtime_bridge = PolicyRuntimeBridge()

    def evaluate(self, bundle: TrajectoryBundle, artifact: PolicyArtifact, split: str = "validation") -> EvaluationReport:
        self._validate_boundary(bundle)
        trajectories = bundle.splits[split]
        if not trajectories:
            raise ValueError(f"{split} split is empty")

        reward_engine = RewardEngine(bundle.reward_spec, artifact.action_space)
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

        for trajectory in trajectories:
            current_policy_state = PolicyState()
            for step in trajectory.steps:
                total_steps += 1
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

        first_step_time = trajectories[0].steps[0].event_time
        last_step_time = trajectories[-1].steps[-1].event_time
        active_range = TimeRange(
            start=first_step_time,
            end=last_step_time + timedelta(seconds=bundle.dataset_spec.sampling_interval_seconds),
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
            coverage_symbols=bundle.dataset_spec.symbols,
            coverage_venues=bundle.dataset_spec.exchanges,
            coverage_streams=bundle.dataset_spec.stream_universe,
            active_date_range=active_range,
            notes=notes,
        )

    def _validate_boundary(self, bundle: TrajectoryBundle) -> None:
        if self.boundary.fill_assumption_mode != bundle.reward_spec.timestamping:
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
