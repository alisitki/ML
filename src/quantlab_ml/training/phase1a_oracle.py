from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from quantlab_ml.contracts import ActionFeasibilitySurface, PolicyState
from quantlab_ml.rewards import RewardEngine


@dataclass(frozen=True, slots=True)
class Phase1AOracleLabel:
    joint_action_key: str
    oracle_return: float


def phase1a_joint_action_keys(venue_choices: list[str]) -> list[str]:
    keys = ["abstain", "hold", "exit"]
    keys.extend(f"enter_long@{venue}" for venue in venue_choices)
    keys.extend(f"enter_short@{venue}" for venue in venue_choices)
    return keys


def decode_phase1a_joint_action_key(joint_action_key: str) -> tuple[str, str | None]:
    if "@" not in joint_action_key:
        return joint_action_key, None
    action_key, venue = joint_action_key.split("@", 1)
    return action_key, venue


def phase1a_joint_action_mask(
    *,
    venue_choices: list[str],
    action_feasibility: ActionFeasibilitySurface,
    policy_state: PolicyState,
    preferred_size_band: str,
    preferred_leverage_band: str,
) -> np.ndarray:
    flat = policy_state.previous_position_side == "flat"
    mask = [flat, not flat, not flat]
    for venue in venue_choices:
        mask.append(
            flat
            and action_feasibility.is_feasible(
                "enter_long",
                venue,
                preferred_size_band,
                preferred_leverage_band,
            )
        )
    for venue in venue_choices:
        mask.append(
            flat
            and action_feasibility.is_feasible(
                "enter_short",
                venue,
                preferred_size_band,
                preferred_leverage_band,
            )
        )
    return np.asarray(mask, dtype=np.bool_)


def phase1a_label_available(*, row_count: int, row_index: int, horizon_steps: int) -> bool:
    return row_index + horizon_steps <= row_count


def solve_phase1a_oracle(
    *,
    rows: list[Any],
    row_index: int,
    horizon_steps: int,
    venue_choices: list[str],
    reward_engine: RewardEngine,
    policy_state: PolicyState,
    preferred_size_band: str,
    preferred_leverage_band: str,
) -> Phase1AOracleLabel:
    if not phase1a_label_available(row_count=len(rows), row_index=row_index, horizon_steps=horizon_steps):
        raise ValueError("phase1a oracle requires a full local horizon inside the same trajectory chunk")

    joint_keys = phase1a_joint_action_keys(venue_choices)

    @lru_cache(maxsize=None)
    def _solve(
        offset: int,
        previous_position_side: str,
        previous_venue: str | None,
        hold_age_steps: int,
        turnover_accumulator: float,
    ) -> tuple[float, str]:
        if offset >= horizon_steps:
            return 0.0, "abstain"

        row = rows[row_index + offset]
        current_state = PolicyState(
            previous_position_side=previous_position_side,
            previous_venue=previous_venue,
            hold_age_steps=hold_age_steps,
            turnover_accumulator=turnover_accumulator,
        )
        valid_mask = phase1a_joint_action_mask(
            venue_choices=venue_choices,
            action_feasibility=row.action_feasibility,
            policy_state=current_state,
            preferred_size_band=preferred_size_band,
            preferred_leverage_band=preferred_leverage_band,
        )
        best_total = float("-inf")
        best_key = "abstain"

        for joint_action_key, valid in zip(joint_keys, valid_mask, strict=True):
            if not valid:
                continue
            applied = apply_phase1a_joint_action(
                reward_engine=reward_engine,
                row=row,
                joint_action_key=joint_action_key,
                policy_state=current_state,
                preferred_size_band=preferred_size_band,
                preferred_leverage_band=preferred_leverage_band,
            )
            next_state = reward_engine.advance_policy_state(current_state, applied)
            future_total, _ = _solve(
                offset + 1,
                next_state.previous_position_side,
                next_state.previous_venue,
                next_state.hold_age_steps,
                next_state.turnover_accumulator,
            )
            total = applied.net_reward + future_total
            if total > best_total:
                best_total = total
                best_key = joint_action_key

        return best_total, best_key

    best_total, best_key = _solve(
        0,
        policy_state.previous_position_side,
        policy_state.previous_venue,
        policy_state.hold_age_steps,
        policy_state.turnover_accumulator,
    )
    return Phase1AOracleLabel(joint_action_key=best_key, oracle_return=best_total)


def apply_phase1a_joint_action(
    *,
    reward_engine: RewardEngine,
    row: Any,
    joint_action_key: str,
    policy_state: PolicyState,
    preferred_size_band: str,
    preferred_leverage_band: str,
):
    action_key, venue = decode_phase1a_joint_action_key(joint_action_key)
    size_band_key = preferred_size_band if action_key.startswith("enter_") else None
    leverage_band_key = preferred_leverage_band if action_key.startswith("enter_") else None
    return reward_engine.apply_decision(
        snapshot=row.reward_snapshot,
        requested_action_key=action_key,
        action_feasibility=row.action_feasibility,
        infeasible_action_treatment="force_abstain",
        venue=venue,
        size_band_key=size_band_key,
        leverage_band_key=leverage_band_key,
        policy_state=policy_state,
    )
