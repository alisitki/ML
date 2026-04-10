from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quantlab_ml.contracts import (
    ActionFeasibilitySurface,
    ActionReward,
    ActionSpaceSpec,
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
        """V2: Venue-aware snapshot; her available venue için ayrı ActionReward üretir.

        Averaging yapılmaz. Reward, seçilen venue'nun fiyat serisinden hesaplanır.
        selected_venue context'te None ise tüm available venue'lar için hesaplar.
        """
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

            feasible_mask = action_feasibility.to_flat_mask()
            applicable = feasible_mask.get(action.key, False)

            if not applicable:
                action_rewards.append(
                    ActionReward(
                        action_key=action.key,
                        gross_return=0.0,
                        fee=0.0,
                        funding=0.0,
                        slippage=0.0,
                        risk_penalty=0.0,
                        turnover_penalty=0.0,
                        net_reward=self.reward_spec.infeasible_action_penalty,
                        applicable=False,
                        venue=None,
                    )
                )
                continue

            # Her available venue için hesapla; sonuçları doğrudan listeye ekle
            venue_rewards = self._build_venue_rewards(
                action_key=action.key,
                direction=action.direction,
                reward_context=reward_context,
                reward_timeline=reward_timeline,
            )
            action_rewards.extend(venue_rewards)
            # venue=None fallback kaydı yok — herkes for_action_venue(key, venue) kullanmalı.

        # Legacy-compat scalar alanları: context'teki ilk available venue'dan al
        ref_price = 0.0
        next_price = 0.0
        if reward_context.venues:
            first_venue = next(iter(reward_context.venues.values()))
            ref_price = first_venue.reference_price
            # Timeline'dan ilk adım
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
        action_key: str,
        direction: str,
        reward_context: RewardContext,
        reward_timeline: RewardTimeline,
    ) -> list[ActionReward]:
        rewards: list[ActionReward] = []
        for exchange, venue_ref in reward_context.venues.items():
            ref_price = venue_ref.reference_price
            # Sonraki fiyatı timeline'dan al
            series = reward_timeline.venue_reference_series.get(exchange, [])
            next_price_val = series[0] if series else 0.0

            if ref_price <= 0.0 or next_price_val <= 0.0:
                rewards.append(
                    ActionReward(
                        action_key=action_key,
                        gross_return=0.0,
                        fee=0.0,
                        funding=0.0,
                        slippage=0.0,
                        risk_penalty=0.0,
                        turnover_penalty=0.0,
                        net_reward=self.reward_spec.infeasible_action_penalty,
                        applicable=False,
                        venue=exchange,
                    )
                )
                continue

            price_return = (next_price_val - ref_price) / ref_price
            gross_return = price_return if direction == "long" else -price_return
            fee = -(venue_ref.fee_regime_bps / 10_000.0)
            slippage = -(venue_ref.slippage_proxy_bps / 10_000.0)
            funding_rate = venue_ref.funding_rate
            funding = (
                -(funding_rate * self.reward_spec.funding_weight)
                if direction == "long"
                else (funding_rate * self.reward_spec.funding_weight)
            )
            risk_penalty = -(abs(price_return) * self.reward_spec.risk_aversion)
            turnover_penalty = -self.reward_spec.turnover_penalty
            net_reward = gross_return + fee + slippage + funding + risk_penalty + turnover_penalty

            rewards.append(
                ActionReward(
                    action_key=action_key,
                    gross_return=gross_return,
                    fee=fee,
                    funding=funding,
                    slippage=slippage,
                    risk_penalty=risk_penalty,
                    turnover_penalty=turnover_penalty,
                    net_reward=net_reward,
                    applicable=True,
                    venue=exchange,
                )
            )
        return rewards

    def apply_decision(
        self,
        snapshot: RewardSnapshot,
        requested_action_key: str,
        action_mask: dict[str, bool],
        infeasible_action_treatment: str,
        venue: str | None = None,
    ) -> AppliedReward:
        """Karar uygula.

        venue=None ise tüm venue'lardan en iyi (max net_reward) applicable kaydı seçer.
        venue verilmisse sadece o venue'nun kaydını kullanır.
        """
        requested_reward = self._select_best_reward(snapshot, requested_action_key, venue)
        requested_available = action_mask.get(requested_action_key, False) and (requested_reward is not None) and requested_reward.applicable

        if requested_available:
            return AppliedReward(
                requested_action_key=requested_action_key,
                applied_action_key=requested_action_key,
                net_reward=requested_reward.net_reward,
                fee=requested_reward.fee,
                funding=requested_reward.funding,
                slippage=requested_reward.slippage,
                risk_penalty=requested_reward.risk_penalty,
                turnover_penalty=requested_reward.turnover_penalty,
                infeasible=False,
                infeasible_penalty=0.0,
                venue=requested_reward.venue,
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
            )

        raise ValueError(f"unsupported infeasible_action_treatment: {infeasible_action_treatment}")

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _select_best_reward(
        self,
        snapshot: RewardSnapshot,
        action_key: str,
        venue: str | None,
    ) -> ActionReward | None:
        """Belirli venue veya tüm venue'lar için en iyi applicable ActionReward'u seçer.

        abstain için venue=None, for_action('abstain') yiüne gider.
        """
        if action_key == "abstain":
            return snapshot.for_action("abstain")

        candidates = [
            r for r in snapshot.action_rewards
            if r.action_key == action_key
            and (venue is None or r.venue == venue)
            and r.applicable
        ]
        if not candidates:
            # Herhangi bir applicable kayıt yok — infeasible fallback için None dön
            non_applicable = [
                r for r in snapshot.action_rewards
                if r.action_key == action_key
                and (venue is None or r.venue == venue)
            ]
            return non_applicable[0] if non_applicable else None

        return max(candidates, key=lambda r: r.net_reward)
