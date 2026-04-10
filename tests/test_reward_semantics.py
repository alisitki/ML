from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quantlab_ml.contracts import RewardContext, RewardTimeline, VenueExecutionRef
from quantlab_ml.rewards import RewardEngine


def _make_context(ref_price_binance: float, ref_price_bybit: float) -> RewardContext:
    return RewardContext(
        venues={
            "binance": VenueExecutionRef(
                exchange="binance",
                reference_price=ref_price_binance,
                fee_regime_bps=5.0,
                slippage_proxy_bps=2.0,
                funding_rate=0.001,
                funding_freshness_seconds=60.0,
            ),
            "bybit": VenueExecutionRef(
                exchange="bybit",
                reference_price=ref_price_bybit,
                fee_regime_bps=4.0,
                slippage_proxy_bps=2.0,
                funding_rate=0.0005,
                funding_freshness_seconds=120.0,
            ),
        },
        hold_horizon_steps=1,
        turnover_state=0.0,
        previous_position_state="flat",
        selected_venue=None,
    )


def _make_timeline(
    next_binance: float, next_bybit: float, horizon: int = 1
) -> RewardTimeline:
    return RewardTimeline(
        horizon_steps=horizon,
        venue_reference_series={
            "binance": [next_binance],
            "bybit": [next_bybit],
        },
    )


def test_reward_engine_applies_feasible_actions(training_bundle, reward_spec) -> None:
    _, action_space, _ = training_bundle
    engine = RewardEngine(reward_spec, action_space)
    context = _make_context(100.0, 100.0)
    timeline = _make_timeline(110.0, 110.0)
    action_mask = {"abstain": True, "enter_long": True, "enter_short": True}

    from quantlab_ml.contracts import ActionFeasibilitySurface, FeasibilityCell

    surface: dict = {}
    for action in action_space.actions:
        surface[action.key] = {}
        for venue in ["binance", "bybit"]:
            surface[action.key][venue] = {
                "micro": {"low": FeasibilityCell(feasible=True), "medium": FeasibilityCell(feasible=True)},
                "small": {"low": FeasibilityCell(feasible=True), "medium": FeasibilityCell(feasible=True)},
            }
    feasibility = ActionFeasibilitySurface(surface=surface)

    snapshot = engine.build_snapshot(
        event_time=datetime(2024, 1, 1, tzinfo=UTC),
        reward_context=context,
        reward_timeline=timeline,
        action_feasibility=feasibility,
    )

    long_outcome = engine.apply_decision(snapshot, "enter_long", action_mask, "force_abstain")
    short_outcome = engine.apply_decision(snapshot, "enter_short", action_mask, "force_abstain")
    abstain_outcome = engine.apply_decision(snapshot, "abstain", action_mask, "force_abstain")

    assert long_outcome.applied_action_key == "enter_long"
    # enter_long için en iyi venue'yu bul (venue=None fallback kaldırıldı)
    best_long = engine._select_best_reward(snapshot, "enter_long", venue=None)
    assert best_long is not None
    assert long_outcome.net_reward == pytest.approx(best_long.net_reward)
    assert short_outcome.applied_action_key == "enter_short"
    assert abstain_outcome.applied_action_key == "abstain"
    assert abstain_outcome.net_reward == pytest.approx(0.0)


def test_reward_engine_force_abstain_keeps_infeasible_penalty(training_bundle, reward_spec) -> None:
    _, action_space, _ = training_bundle
    engine = RewardEngine(reward_spec, action_space)
    context = _make_context(100.0, 100.0)
    timeline = _make_timeline(110.0, 110.0)

    from quantlab_ml.contracts import ActionFeasibilitySurface, FeasibilityCell

    surface: dict = {}
    for action in action_space.actions:
        surface[action.key] = {}
        feasible = action.key == "abstain"
        for venue in ["binance", "bybit"]:
            surface[action.key][venue] = {
                "micro": {
                    "low": FeasibilityCell(feasible=feasible, reason="" if feasible else "no_price"),
                    "medium": FeasibilityCell(feasible=feasible, reason="" if feasible else "no_price"),
                },
                "small": {
                    "low": FeasibilityCell(feasible=feasible, reason="" if feasible else "no_price"),
                    "medium": FeasibilityCell(feasible=feasible, reason="" if feasible else "no_price"),
                },
            }
    feasibility = ActionFeasibilitySurface(surface=surface)

    snapshot = engine.build_snapshot(
        event_time=datetime(2024, 1, 1, tzinfo=UTC),
        reward_context=context,
        reward_timeline=timeline,
        action_feasibility=feasibility,
    )
    action_mask = {"abstain": True, "enter_long": False, "enter_short": False}
    outcome = engine.apply_decision(snapshot, "enter_long", action_mask, "force_abstain")

    assert outcome.applied_action_key == "abstain"
    assert outcome.infeasible is True
    assert outcome.infeasible_penalty == pytest.approx(reward_spec.infeasible_action_penalty)
    assert outcome.net_reward == pytest.approx(reward_spec.infeasible_action_penalty)
