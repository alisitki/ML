from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from quantlab_ml.contracts import (
    ActionFeasibilitySurface,
    FeasibilityCell,
    PolicyState,
    RewardContext,
    RewardTimeline,
    VenueExecutionRef,
)
from quantlab_ml.rewards import RewardEngine
from quantlab_ml.training.phase1a_oracle import (
    phase1a_joint_action_mask,
    phase1a_label_available,
    solve_phase1a_oracle,
)


def _make_context() -> RewardContext:
    return RewardContext(
        venues={
            "binance": VenueExecutionRef(
                exchange="binance",
                reference_price=100.0,
                fee_regime_bps=1.0,
                slippage_proxy_bps=0.5,
                funding_rate=0.0,
                funding_freshness_seconds=30.0,
            ),
            "bybit": VenueExecutionRef(
                exchange="bybit",
                reference_price=100.0,
                fee_regime_bps=1.0,
                slippage_proxy_bps=0.5,
                funding_rate=0.0,
                funding_freshness_seconds=30.0,
            ),
            "okx": VenueExecutionRef(
                exchange="okx",
                reference_price=100.0,
                fee_regime_bps=1.0,
                slippage_proxy_bps=0.5,
                funding_rate=0.0,
                funding_freshness_seconds=30.0,
            ),
        },
        hold_horizon_steps=4,
        turnover_state=0.0,
        previous_position_state="flat",
        selected_venue=None,
    )


def _make_timeline(binance: list[float], bybit: list[float], okx: list[float]) -> RewardTimeline:
    return RewardTimeline(
        horizon_steps=len(binance),
        venue_reference_series={
            "binance": binance,
            "bybit": bybit,
            "okx": okx,
        },
    )


def _make_feasibility(action_space) -> ActionFeasibilitySurface:
    surface: dict = {}
    for action in action_space.actions:
        surface[action.key] = {}
        feasible = action.key == "abstain" or action.key.startswith("enter_")
        for venue in ["binance", "bybit", "okx"]:
            surface[action.key][venue] = {
                "micro": {
                    "low": FeasibilityCell(feasible=feasible, reason="" if feasible else "forced_infeasible"),
                }
            }
    return ActionFeasibilitySurface(surface=surface)


def test_phase1a_joint_action_mask_respects_flat_and_in_position(phase1a_training_bundle) -> None:
    _, action_space, _ = phase1a_training_bundle
    feasibility = _make_feasibility(action_space)

    flat_mask = phase1a_joint_action_mask(
        venue_choices=action_space.venue_choices,
        action_feasibility=feasibility,
        policy_state=PolicyState(),
        preferred_size_band="micro",
        preferred_leverage_band="low",
    )
    long_mask = phase1a_joint_action_mask(
        venue_choices=action_space.venue_choices,
        action_feasibility=feasibility,
        policy_state=PolicyState(previous_position_side="long", previous_venue="binance"),
        preferred_size_band="micro",
        preferred_leverage_band="low",
    )

    assert flat_mask.tolist() == [True, False, False, True, True, True, True, True, True]
    assert long_mask.tolist() == [False, True, True, False, False, False, False, False, False]


def test_phase1a_label_availability_masks_tail_rows() -> None:
    assert phase1a_label_available(row_count=8, row_index=3, horizon_steps=4) is True
    assert phase1a_label_available(row_count=8, row_index=4, horizon_steps=4) is True
    assert phase1a_label_available(row_count=8, row_index=5, horizon_steps=4) is False


def test_phase1a_oracle_prefers_best_legal_enter_under_h4(phase1a_training_bundle, reward_spec) -> None:
    _, action_space, _ = phase1a_training_bundle
    reward_engine = RewardEngine(reward_spec.model_copy(update={"horizon_steps": 4}), action_space)
    feasibility = _make_feasibility(action_space)
    rows = []
    for _ in range(4):
        snapshot = reward_engine.build_snapshot(
            event_time=datetime(2024, 1, 1, tzinfo=UTC),
            reward_context=_make_context(),
            reward_timeline=_make_timeline(
                [101.0, 103.0, 106.0, 110.0],
                [99.5, 99.0, 98.5, 98.0],
                [100.0, 100.0, 100.0, 100.0],
            ),
            action_feasibility=feasibility,
        )
        rows.append(SimpleNamespace(reward_snapshot=snapshot, action_feasibility=feasibility))

    label = solve_phase1a_oracle(
        rows=rows,
        row_index=0,
        horizon_steps=4,
        venue_choices=action_space.venue_choices,
        reward_engine=reward_engine,
        policy_state=PolicyState(),
        preferred_size_band="micro",
        preferred_leverage_band="low",
    )

    assert label.joint_action_key == "enter_long@binance"
    assert label.oracle_return > 0.0
