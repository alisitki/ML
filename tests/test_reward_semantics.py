from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quantlab_ml.contracts import (
    ActionFeasibilitySurface,
    FeasibilityCell,
    PolicyState,
    RewardContext,
    RewardTimeline,
    VenueExecutionRef,
)
from quantlab_ml.rewards import RewardEngine


def _make_context(
    *,
    ref_price_binance: float = 100.0,
    ref_price_bybit: float = 100.0,
    funding_binance: float = 0.001,
    funding_bybit: float = 0.0005,
    freshness_binance: float = 60.0,
    freshness_bybit: float = 120.0,
) -> RewardContext:
    return RewardContext(
        venues={
            "binance": VenueExecutionRef(
                exchange="binance",
                reference_price=ref_price_binance,
                fee_regime_bps=5.0,
                slippage_proxy_bps=2.0,
                funding_rate=funding_binance,
                funding_freshness_seconds=freshness_binance,
            ),
            "bybit": VenueExecutionRef(
                exchange="bybit",
                reference_price=ref_price_bybit,
                fee_regime_bps=4.0,
                slippage_proxy_bps=2.0,
                funding_rate=funding_bybit,
                funding_freshness_seconds=freshness_bybit,
            ),
        },
        hold_horizon_steps=1,
        turnover_state=0.0,
        previous_position_state="flat",
        selected_venue=None,
    )


def _make_timeline(
    *series_by_venue: tuple[str, list[float]],
) -> RewardTimeline:
    timeline = dict(series_by_venue)
    first_series = next(iter(timeline.values()))
    return RewardTimeline(horizon_steps=len(first_series), venue_reference_series=timeline)


def _make_feasibility(
    action_space,
    *,
    directional_feasible: bool = True,
    per_cell_overrides: dict[tuple[str, str, str, str], bool] | None = None,
) -> ActionFeasibilitySurface:
    overrides = per_cell_overrides or {}
    surface: dict = {}
    for action in action_space.actions:
        surface[action.key] = {}
        feasible = directional_feasible or action.key == "abstain"
        for venue in ["binance", "bybit", "okx"]:
            surface[action.key][venue] = {}
            for size_band in ["micro", "small"]:
                surface[action.key][venue][size_band] = {}
                for leverage_band in ["low", "medium"]:
                    cell_feasible = overrides.get(
                        (action.key, venue, size_band, leverage_band),
                        feasible,
                    )
                    surface[action.key][venue][size_band][leverage_band] = FeasibilityCell(
                        feasible=cell_feasible,
                        reason="" if cell_feasible else "forced_infeasible",
                    )
    return ActionFeasibilitySurface(surface=surface)


def test_reward_engine_requires_explicit_directional_dimensions(training_bundle, reward_spec) -> None:
    _, action_space, _ = training_bundle
    engine = RewardEngine(reward_spec, action_space)
    feasibility = _make_feasibility(action_space)
    snapshot = engine.build_snapshot(
        event_time=datetime(2024, 1, 1, tzinfo=UTC),
        reward_context=_make_context(),
        reward_timeline=_make_timeline(("binance", [110.0]), ("bybit", [105.0])),
        action_feasibility=feasibility,
    )

    with pytest.raises(ValueError, match="requires explicit venue"):
        engine.apply_decision(
            snapshot=snapshot,
            requested_action_key="enter_long",
            action_feasibility=feasibility,
            infeasible_action_treatment="force_abstain",
        )


def test_reward_engine_uses_requested_venue_not_best_venue(training_bundle, reward_spec) -> None:
    _, action_space, _ = training_bundle
    engine = RewardEngine(reward_spec, action_space)
    feasibility = _make_feasibility(action_space)
    snapshot = engine.build_snapshot(
        event_time=datetime(2024, 1, 1, tzinfo=UTC),
        reward_context=_make_context(),
        reward_timeline=_make_timeline(("binance", [110.0]), ("bybit", [102.0])),
        action_feasibility=feasibility,
    )

    outcome = engine.apply_decision(
        snapshot=snapshot,
        requested_action_key="enter_long",
        action_feasibility=feasibility,
        infeasible_action_treatment="force_abstain",
        venue="bybit",
        size_band_key="micro",
        leverage_band_key="low",
    )

    bybit_reward = next(
        reward
        for reward in snapshot.action_rewards
        if reward.action_key == "enter_long" and reward.venue == "bybit"
    )
    binance_reward = next(
        reward
        for reward in snapshot.action_rewards
        if reward.action_key == "enter_long" and reward.venue == "binance"
    )
    assert bybit_reward.net_reward < binance_reward.net_reward
    assert outcome.net_reward == pytest.approx(bybit_reward.net_reward - reward_spec.turnover_penalty)
    assert outcome.venue == "bybit"
    assert outcome.reward_context is not None
    assert outcome.reward_context.selected_venue == "bybit"
    assert snapshot.context is not None
    assert snapshot.context.selected_venue == "bybit"


def test_reward_engine_missing_requested_venue_forces_abstain_without_crash(training_bundle, reward_spec) -> None:
    _, action_space, _ = training_bundle
    engine = RewardEngine(reward_spec, action_space)
    feasibility = _make_feasibility(action_space)
    snapshot = engine.build_snapshot(
        event_time=datetime(2024, 1, 1, tzinfo=UTC),
        reward_context=_make_context(),
        reward_timeline=_make_timeline(("binance", [110.0]), ("bybit", [105.0])),
        action_feasibility=feasibility,
    )

    outcome = engine.apply_decision(
        snapshot=snapshot,
        requested_action_key="enter_long",
        action_feasibility=feasibility,
        infeasible_action_treatment="force_abstain",
        venue="okx",
        size_band_key="micro",
        leverage_band_key="low",
    )

    assert outcome.applied_action_key == "abstain"
    assert outcome.infeasible is True
    assert outcome.reward_context is not None
    assert outcome.reward_context.selected_venue is None


def test_reward_engine_enforces_requested_size_and_leverage_cell(training_bundle, reward_spec) -> None:
    _, action_space, _ = training_bundle
    engine = RewardEngine(reward_spec, action_space)
    feasibility = _make_feasibility(
        action_space,
        per_cell_overrides={
            ("enter_long", "binance", "micro", "low"): False,
            ("enter_short", "binance", "micro", "low"): False,
        },
    )
    snapshot = engine.build_snapshot(
        event_time=datetime(2024, 1, 1, tzinfo=UTC),
        reward_context=_make_context(),
        reward_timeline=_make_timeline(("binance", [110.0]), ("bybit", [105.0])),
        action_feasibility=feasibility,
    )

    outcome = engine.apply_decision(
        snapshot=snapshot,
        requested_action_key="enter_long",
        action_feasibility=feasibility,
        infeasible_action_treatment="force_abstain",
        venue="binance",
        size_band_key="micro",
        leverage_band_key="low",
    )

    assert outcome.applied_action_key == "abstain"
    assert outcome.infeasible is True
    assert outcome.infeasible_penalty == pytest.approx(reward_spec.infeasible_action_penalty)


def test_reward_engine_uses_horizon_end_price(training_bundle, reward_spec) -> None:
    _, action_space, _ = training_bundle
    engine = RewardEngine(reward_spec.model_copy(update={"horizon_steps": 3}), action_space)
    feasibility = _make_feasibility(action_space)
    snapshot = engine.build_snapshot(
        event_time=datetime(2024, 1, 1, tzinfo=UTC),
        reward_context=_make_context(),
        reward_timeline=_make_timeline(
            ("binance", [101.0, 104.0, 108.0]),
            ("bybit", [99.0, 99.5, 100.0]),
        ),
        action_feasibility=feasibility,
    )

    outcome = engine.apply_decision(
        snapshot=snapshot,
        requested_action_key="enter_long",
        action_feasibility=feasibility,
        infeasible_action_treatment="force_abstain",
        venue="binance",
        size_band_key="micro",
        leverage_band_key="low",
    )

    gross_return = (108.0 - 100.0) / 100.0
    expected = (
        gross_return
        - (5.0 / 10_000.0)
        - (2.0 / 10_000.0)
        - 0.001
        - abs(gross_return) * reward_spec.risk_aversion
        - reward_spec.turnover_penalty
    )
    assert outcome.net_reward == pytest.approx(expected)


def test_funding_component_uses_freshness_threshold_and_v1_formula(training_bundle, reward_spec) -> None:
    _, action_space, _ = training_bundle
    engine = RewardEngine(
        reward_spec.model_copy(update={"funding_freshness_threshold_seconds": 90.0}),
        action_space,
    )
    feasibility = _make_feasibility(action_space)
    context = _make_context(
        funding_binance=0.002,
        funding_bybit=0.003,
        freshness_binance=60.0,
        freshness_bybit=120.0,
    )
    snapshot = engine.build_snapshot(
        event_time=datetime(2024, 1, 1, tzinfo=UTC),
        reward_context=context,
        reward_timeline=_make_timeline(("binance", [110.0]), ("bybit", [110.0])),
        action_feasibility=feasibility,
    )

    long_outcome = engine.apply_decision(
        snapshot=snapshot,
        requested_action_key="enter_long",
        action_feasibility=feasibility,
        infeasible_action_treatment="force_abstain",
        venue="binance",
        size_band_key="micro",
        leverage_band_key="low",
    )
    short_outcome = engine.apply_decision(
        snapshot=snapshot,
        requested_action_key="enter_short",
        action_feasibility=feasibility,
        infeasible_action_treatment="force_abstain",
        venue="binance",
        size_band_key="micro",
        leverage_band_key="low",
    )
    stale_outcome = engine.apply_decision(
        snapshot=snapshot,
        requested_action_key="enter_long",
        action_feasibility=feasibility,
        infeasible_action_treatment="force_abstain",
        venue="bybit",
        size_band_key="micro",
        leverage_band_key="low",
    )

    assert long_outcome.funding == pytest.approx(-0.002)
    assert short_outcome.funding == pytest.approx(-0.002)
    assert stale_outcome.funding == pytest.approx(0.0)


def test_turnover_penalty_only_on_exposure_change(training_bundle, reward_spec) -> None:
    _, action_space, _ = training_bundle
    engine = RewardEngine(reward_spec, action_space)
    feasibility = _make_feasibility(action_space)
    snapshot = engine.build_snapshot(
        event_time=datetime(2024, 1, 1, tzinfo=UTC),
        reward_context=_make_context(),
        reward_timeline=_make_timeline(("binance", [110.0]), ("bybit", [105.0])),
        action_feasibility=feasibility,
    )

    flat_state = PolicyState(previous_position_side="flat")
    long_state = PolicyState(previous_position_side="long", previous_venue="binance")

    enter_from_flat = engine.apply_decision(
        snapshot=snapshot,
        requested_action_key="enter_long",
        action_feasibility=feasibility,
        infeasible_action_treatment="force_abstain",
        venue="binance",
        size_band_key="micro",
        leverage_band_key="low",
        policy_state=flat_state,
    )
    stay_long = engine.apply_decision(
        snapshot=snapshot,
        requested_action_key="enter_long",
        action_feasibility=feasibility,
        infeasible_action_treatment="force_abstain",
        venue="binance",
        size_band_key="micro",
        leverage_band_key="low",
        policy_state=long_state,
    )
    flip_short = engine.apply_decision(
        snapshot=snapshot,
        requested_action_key="enter_short",
        action_feasibility=feasibility,
        infeasible_action_treatment="force_abstain",
        venue="binance",
        size_band_key="micro",
        leverage_band_key="low",
        policy_state=long_state,
    )

    assert enter_from_flat.turnover_penalty == pytest.approx(-reward_spec.turnover_penalty)
    assert stay_long.turnover_penalty == pytest.approx(0.0)
    assert flip_short.turnover_penalty == pytest.approx(-reward_spec.turnover_penalty)
    assert stay_long.resulting_position_side == "long"
    assert flip_short.resulting_position_side == "short"


def test_reward_engine_force_abstain_keeps_infeasible_penalty(training_bundle, reward_spec) -> None:
    _, action_space, _ = training_bundle
    engine = RewardEngine(reward_spec, action_space)
    feasibility = _make_feasibility(action_space, directional_feasible=False)
    snapshot = engine.build_snapshot(
        event_time=datetime(2024, 1, 1, tzinfo=UTC),
        reward_context=_make_context(),
        reward_timeline=_make_timeline(("binance", [110.0]), ("bybit", [110.0])),
        action_feasibility=feasibility,
    )

    outcome = engine.apply_decision(
        snapshot=snapshot,
        requested_action_key="enter_long",
        action_feasibility=feasibility,
        infeasible_action_treatment="force_abstain",
        venue="binance",
        size_band_key="micro",
        leverage_band_key="low",
    )

    assert outcome.applied_action_key == "abstain"
    assert outcome.infeasible is True
    assert outcome.infeasible_penalty == pytest.approx(reward_spec.infeasible_action_penalty)
    assert outcome.net_reward == pytest.approx(reward_spec.infeasible_action_penalty)
    assert outcome.reward_context is not None
    assert outcome.reward_context.selected_venue == "binance"
