"""test_v2_reward_replay.py

Venue-specific reference farklarının reward sonucunu değiştirdiğini
ve exchange-average kullanılmadığını doğrular.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quantlab_ml.contracts import (
    ActionFeasibilitySurface,
    ActionSpaceSpec,
    FeasibilityCell,
    RewardContext,
    RewardEventSpec,
    RewardTimeline,
    VenueExecutionRef,
)
from quantlab_ml.rewards import RewardEngine


def _make_feasibility(action_space: ActionSpaceSpec) -> ActionFeasibilitySurface:
    surface = {}
    for action in action_space.actions:
        surface[action.key] = {}
        for venue in ["binance", "bybit"]:
            surface[action.key][venue] = {
                "micro": {
                    "low": FeasibilityCell(feasible=True),
                    "medium": FeasibilityCell(feasible=True),
                },
                "small": {
                    "low": FeasibilityCell(feasible=True),
                    "medium": FeasibilityCell(feasible=True),
                },
            }
    return ActionFeasibilitySurface(surface=surface)


def test_venue_specific_reference_affects_reward(training_bundle, reward_spec) -> None:
    """Farklı venue reference price'ları farklı reward üretir; averaging yapılmaz."""
    _, action_space, _ = training_bundle
    engine = RewardEngine(reward_spec, action_space)

    # binance: 100→110 (long profitable)
    # bybit: 100→105 (long less profitable)
    context = RewardContext(
        venues={
            "binance": VenueExecutionRef(
                exchange="binance",
                reference_price=100.0,
                fee_regime_bps=reward_spec.fee_bps,
                slippage_proxy_bps=reward_spec.slippage_bps,
                funding_rate=0.0,
                funding_freshness_seconds=60.0,
            ),
            "bybit": VenueExecutionRef(
                exchange="bybit",
                reference_price=100.0,
                fee_regime_bps=reward_spec.fee_bps,
                slippage_proxy_bps=reward_spec.slippage_bps,
                funding_rate=0.0,
                funding_freshness_seconds=60.0,
            ),
        }
    )
    timeline_binance_up = RewardTimeline(
        horizon_steps=1,
        venue_reference_series={"binance": [110.0], "bybit": [105.0]},
    )
    timeline_bybit_up = RewardTimeline(
        horizon_steps=1,
        venue_reference_series={"binance": [95.0], "bybit": [110.0]},
    )
    feasibility = _make_feasibility(action_space)

    snap_a = engine.build_snapshot(
        event_time=datetime(2024, 1, 1, tzinfo=UTC),
        reward_context=context,
        reward_timeline=timeline_binance_up,
        action_feasibility=feasibility,
    )
    snap_b = engine.build_snapshot(
        event_time=datetime(2024, 1, 1, tzinfo=UTC),
        reward_context=context,
        reward_timeline=timeline_bybit_up,
        action_feasibility=feasibility,
    )

    # binance venue için ödüller farklı olmalı
    binance_long_a = next(
        r for r in snap_a.action_rewards
        if r.action_key == "enter_long" and r.venue == "binance"
    )
    binance_long_b = next(
        r for r in snap_b.action_rewards
        if r.action_key == "enter_long" and r.venue == "binance"
    )
    assert binance_long_a.net_reward != pytest.approx(binance_long_b.net_reward), (
        "Different next prices on binance should produce different rewards"
    )


def test_no_exchange_average_in_rewards(training_bundle, reward_spec) -> None:
    """Her venue'nun kendi reward'ı var; tek bir 'average' reward yok."""
    _, action_space, _ = training_bundle
    engine = RewardEngine(reward_spec, action_space)

    context = RewardContext(
        venues={
            "binance": VenueExecutionRef(
                exchange="binance",
                reference_price=100.0,
                fee_regime_bps=reward_spec.fee_bps,
                slippage_proxy_bps=reward_spec.slippage_bps,
                funding_rate=0.001,
                funding_freshness_seconds=60.0,
            ),
            "bybit": VenueExecutionRef(
                exchange="bybit",
                reference_price=100.0,
                fee_regime_bps=reward_spec.fee_bps,
                slippage_proxy_bps=reward_spec.slippage_bps,
                funding_rate=0.0,
                funding_freshness_seconds=60.0,
            ),
        }
    )
    timeline = RewardTimeline(
        horizon_steps=1,
        venue_reference_series={"binance": [110.0], "bybit": [110.0]},
    )
    feasibility = _make_feasibility(action_space)

    snap = engine.build_snapshot(
        event_time=datetime(2024, 1, 1, tzinfo=UTC),
        reward_context=context,
        reward_timeline=timeline,
        action_feasibility=feasibility,
    )

    # enter_long için hem binance hem bybit venue-specific reward mevcut olmalı
    venues_with_long = {
        r.venue
        for r in snap.action_rewards
        if r.action_key == "enter_long" and r.venue is not None and r.applicable
    }
    assert "binance" in venues_with_long
    assert "bybit" in venues_with_long

    # Funding farklı olduğu için binance ve bybit long reward'ları farklı
    binance_r = next(r for r in snap.action_rewards if r.action_key == "enter_long" and r.venue == "binance")
    bybit_r = next(r for r in snap.action_rewards if r.action_key == "enter_long" and r.venue == "bybit")
    assert binance_r.funding != pytest.approx(bybit_r.funding), (
        "Binance (funding=0.001) and Bybit (funding=0.0) should have different funding components"
    )


def test_venue_rewards_present_in_full_trajectory(trajectory_bundle) -> None:
    """Full trajectory'de her step'in reward_snapshot'ı venue bilgisi taşımalı."""
    first_step = trajectory_bundle.splits["train"][0].steps[0]
    assert first_step.reward_snapshot.context is not None
    assert first_step.reward_snapshot.timeline is not None
    # En az bir available venue için ActionReward var mı
    venue_rewards = [
        r for r in first_step.reward_snapshot.action_rewards
        if r.venue is not None and r.applicable
    ]
    assert len(venue_rewards) > 0, "Expected at least one venue-specific action reward"
