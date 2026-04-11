from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quantlab_ml.contracts import (
    ActionFeasibilitySurface,
    ActionChoice,
    ActionReward,
    ActionSpaceSpec,
    PolicyState,
    RewardContext,
    RewardEventSpec,
    RewardSnapshot,
    RewardTimeline,
)


@dataclass(slots=True)
class AppliedReward:
    requested_action_key: str
    applied_action_key: str
    net_reward: float
    fee: float
    funding: float
    slippage: float
    risk_penalty: float
    turnover_penalty: float
    infeasible: bool
    infeasible_penalty: float
    venue: str | None = None
    size_band_key: str | None = None
    leverage_band_key: str | None = None
    previous_position_side: str = "flat"
    resulting_position_side: str = "flat"
    reward_context: RewardContext | None = None


class RewardEngine:
    def __init__(self, reward_spec: RewardEventSpec, action_space: ActionSpaceSpec):
        self.reward_spec = reward_spec
        self.action_space = action_space

    def build_snapshot(
        self,
        event_time: datetime,
        reward_context: RewardContext,
        reward_timeline: RewardTimeline,
        action_feasibility: ActionFeasibilitySurface,
    ) -> RewardSnapshot:
        """V2: Venue-aware snapshot; her available venue için ayrı ActionReward üretir."""
        action_rewards: list[ActionReward] = []

        for action in self.action_space.actions:
            if action.key == "abstain":
                action_rewards.append(
                    ActionReward(
                        action_key="abstain",
                        gross_return=0.0,
                        fee=0.0,
                        funding=0.0,
                        slippage=0.0,
                        risk_penalty=0.0,
                        turnover_penalty=0.0,
                        net_reward=0.0,
                        applicable=True,
                        venue=None,
                    )
                )
                continue

            venue_rewards = self._build_venue_rewards(
                action=action,
                reward_context=reward_context,
                reward_timeline=reward_timeline,
                action_feasibility=action_feasibility,
            )
            action_rewards.extend(venue_rewards)

        # Legacy-compat scalar alanları: context'teki ilk available venue'dan al
        ref_price = 0.0
        next_price = 0.0
        if reward_context.venues:
            first_venue = next(iter(reward_context.venues.values()))
            ref_price = first_venue.reference_price
            first_series = next(iter(reward_timeline.venue_reference_series.values()), [])
            next_price = first_series[0] if first_series else 0.0

        return RewardSnapshot(
            event_time=event_time,
            reference_price=ref_price,
            next_price=next_price,
            action_rewards=action_rewards,
            context=reward_context,
            timeline=reward_timeline,
        )

    def _build_venue_rewards(
        self,
        action: ActionChoice,
        reward_context: RewardContext,
        reward_timeline: RewardTimeline,
        action_feasibility: ActionFeasibilitySurface,
    ) -> list[ActionReward]:
        rewards: list[ActionReward] = []
        for exchange, venue_ref in reward_context.venues.items():
            ref_price = venue_ref.reference_price
            series = reward_timeline.venue_reference_series.get(exchange, [])
            horizon_end_price = series[-1] if series else 0.0
            venue_applicable = self._any_feasible_for_venue(action_feasibility, action.key, exchange)

            if ref_price <= 0.0 or horizon_end_price <= 0.0:
                rewards.append(
                    ActionReward(
                        action_key=action.key,
                        gross_return=0.0,
                        fee=0.0,
                        funding=0.0,
                        slippage=0.0,
                        risk_penalty=0.0,
                        turnover_penalty=0.0,
                        net_reward=0.0,
                        applicable=False,
                        venue=exchange,
                    )
                )
                continue

            price_return = (horizon_end_price - ref_price) / ref_price
            gross_return = price_return if action.direction == "long" else -price_return
            fee = -(venue_ref.fee_regime_bps / 10_000.0)
            slippage = -(venue_ref.slippage_proxy_bps / 10_000.0)
            funding_rate_effective = (
                venue_ref.funding_rate
                if venue_ref.funding_freshness_seconds <= self.reward_spec.funding_freshness_threshold_seconds
                else 0.0
            )
            funding = -funding_rate_effective
            risk_penalty = -(abs(gross_return) * self.reward_spec.risk_aversion)
            turnover_penalty = 0.0
            net_reward = gross_return + fee + slippage + funding + risk_penalty

            rewards.append(
                ActionReward(
                    action_key=action.key,
                    gross_return=gross_return,
                    fee=fee,
                    funding=funding,
                    slippage=slippage,
                    risk_penalty=risk_penalty,
                    turnover_penalty=turnover_penalty,
                    net_reward=net_reward,
                    applicable=venue_applicable,
                    venue=exchange,
                )
            )
        return rewards

    def apply_decision(
        self,
        snapshot: RewardSnapshot,
        requested_action_key: str,
        action_feasibility: ActionFeasibilitySurface,
        infeasible_action_treatment: str,
        venue: str | None = None,
        size_band_key: str | None = None,
        leverage_band_key: str | None = None,
        policy_state: PolicyState | None = None,
    ) -> AppliedReward:
        action = self._action_by_key(requested_action_key)
        self._validate_decision_dimensions(
            action=action,
            venue=venue,
            size_band_key=size_band_key,
            leverage_band_key=leverage_band_key,
        )
        effective_reward_context = self._effective_reward_context(snapshot.context, venue)
        if effective_reward_context is not None:
            snapshot.context = effective_reward_context
        previous_position_side = self._previous_position_side(policy_state, snapshot.context)
        resulting_position_side = self._resulting_position_side(action, previous_position_side)

        if action.key == "abstain":
            abstain_reward = snapshot.for_action("abstain")
            return AppliedReward(
                requested_action_key=requested_action_key,
                applied_action_key="abstain",
                net_reward=abstain_reward.net_reward,
                fee=abstain_reward.fee,
                funding=abstain_reward.funding,
                slippage=abstain_reward.slippage,
                risk_penalty=abstain_reward.risk_penalty,
                turnover_penalty=0.0,
                infeasible=False,
                infeasible_penalty=0.0,
                venue=None,
                previous_position_side=previous_position_side,
                resulting_position_side=previous_position_side,
                reward_context=effective_reward_context,
            )

        requested_reward = self._select_reward(snapshot, requested_action_key, venue)
        requested_available = (
            requested_reward is not None
            and requested_reward.applicable
            and self._decision_is_feasible(
                action_feasibility=action_feasibility,
                action=action,
                venue=venue,
                size_band_key=size_band_key,
                leverage_band_key=leverage_band_key,
            )
        )

        if requested_available:
            turnover_penalty = self._turnover_penalty(previous_position_side, resulting_position_side)
            return AppliedReward(
                requested_action_key=requested_action_key,
                applied_action_key=requested_action_key,
                net_reward=requested_reward.net_reward + turnover_penalty,
                fee=requested_reward.fee,
                funding=requested_reward.funding,
                slippage=requested_reward.slippage,
                risk_penalty=requested_reward.risk_penalty,
                turnover_penalty=turnover_penalty,
                infeasible=False,
                infeasible_penalty=0.0,
                venue=venue,
                size_band_key=size_band_key,
                leverage_band_key=leverage_band_key,
                previous_position_side=previous_position_side,
                resulting_position_side=resulting_position_side,
                reward_context=effective_reward_context,
            )

        if infeasible_action_treatment == "force_abstain":
            abstain_reward = snapshot.for_action("abstain")
            infeasible_penalty = self.reward_spec.infeasible_action_penalty
            return AppliedReward(
                requested_action_key=requested_action_key,
                applied_action_key="abstain",
                net_reward=abstain_reward.net_reward + infeasible_penalty,
                fee=abstain_reward.fee,
                funding=abstain_reward.funding,
                slippage=abstain_reward.slippage,
                risk_penalty=abstain_reward.risk_penalty,
                turnover_penalty=abstain_reward.turnover_penalty,
                infeasible=True,
                infeasible_penalty=infeasible_penalty,
                venue=None,
                size_band_key=size_band_key,
                leverage_band_key=leverage_band_key,
                previous_position_side=previous_position_side,
                resulting_position_side=previous_position_side,
                reward_context=effective_reward_context,
            )

        raise ValueError(f"unsupported infeasible_action_treatment: {infeasible_action_treatment}")

    def advance_policy_state(
        self,
        previous_state: PolicyState | None,
        applied_reward: AppliedReward,
    ) -> PolicyState:
        prior = previous_state or PolicyState()
        same_position = applied_reward.resulting_position_side == prior.previous_position_side
        hold_age_steps = 0
        if applied_reward.resulting_position_side != "flat":
            hold_age_steps = prior.hold_age_steps + 1 if same_position else 0
        turnover_accumulator = prior.turnover_accumulator + (
            1.0 if applied_reward.previous_position_side != applied_reward.resulting_position_side else 0.0
        )
        previous_venue = applied_reward.venue if applied_reward.resulting_position_side != "flat" else None
        return PolicyState(
            previous_position_side=applied_reward.resulting_position_side,
            previous_venue=previous_venue,
            hold_age_steps=hold_age_steps,
            turnover_accumulator=turnover_accumulator,
        )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _select_reward(
        self,
        snapshot: RewardSnapshot,
        action_key: str,
        venue: str | None,
    ) -> ActionReward | None:
        if action_key == "abstain":
            return snapshot.for_action("abstain")
        if venue is None:
            raise ValueError(f"directional reward evaluation requires explicit venue for action '{action_key}'")

        for reward in snapshot.action_rewards:
            if reward.action_key == action_key and reward.venue == venue:
                return reward
        return None

    def _action_by_key(self, action_key: str) -> ActionChoice:
        for action in self.action_space.actions:
            if action.key == action_key:
                return action
        raise KeyError(f"unknown action key: {action_key}")

    def _effective_reward_context(
        self,
        reward_context: RewardContext | None,
        venue: str | None,
    ) -> RewardContext | None:
        if reward_context is None:
            return None
        return RewardContext(
            **reward_context.model_dump(exclude={"selected_venue"}),
            selected_venue=venue,
        )

    def _validate_decision_dimensions(
        self,
        action: ActionChoice,
        venue: str | None,
        size_band_key: str | None,
        leverage_band_key: str | None,
    ) -> None:
        if action.requires_venue and venue is None:
            raise ValueError(f"action '{action.key}' requires explicit venue")
        if action.requires_size_band and size_band_key is None:
            raise ValueError(f"action '{action.key}' requires explicit size_band_key")
        if action.requires_leverage_band and leverage_band_key is None:
            raise ValueError(f"action '{action.key}' requires explicit leverage_band_key")

    def _decision_is_feasible(
        self,
        action_feasibility: ActionFeasibilitySurface,
        action: ActionChoice,
        venue: str | None,
        size_band_key: str | None,
        leverage_band_key: str | None,
    ) -> bool:
        if action.key == "abstain":
            return action_feasibility.abstain_feasible()
        assert venue is not None
        assert size_band_key is not None
        assert leverage_band_key is not None
        return action_feasibility.is_feasible(action.key, venue, size_band_key, leverage_band_key)

    def _any_feasible_for_venue(
        self,
        action_feasibility: ActionFeasibilitySurface,
        action_key: str,
        venue: str,
    ) -> bool:
        venue_surface = action_feasibility.surface.get(action_key, {}).get(venue, {})
        return any(
            cell.feasible
            for leverage_map in venue_surface.values()
            for cell in leverage_map.values()
        )

    def _previous_position_side(
        self,
        policy_state: PolicyState | None,
        reward_context: RewardContext | None,
    ) -> str:
        if policy_state is not None:
            return policy_state.previous_position_side
        if reward_context is not None:
            return reward_context.previous_position_state
        return "flat"

    def _resulting_position_side(
        self,
        action: ActionChoice,
        previous_position_side: str,
    ) -> str:
        if action.key in {"abstain", "hold"} or action.category == "abstain":
            return previous_position_side
        if action.key == "exit":
            return "flat"
        if action.direction == "long":
            return "long"
        if action.direction == "short":
            return "short"
        return previous_position_side

    def _turnover_penalty(
        self,
        previous_position_side: str,
        resulting_position_side: str,
    ) -> float:
        turnover_event = previous_position_side != resulting_position_side
        return -self.reward_spec.turnover_penalty if turnover_event else 0.0
