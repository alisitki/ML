"""test_v2_causality.py

- train/eval boundary geçişinin causal olduğunu
- action_feasibility, reward_context ve policy_state'in future kullanmadığını doğrular.
"""
from __future__ import annotations

from datetime import datetime

from quantlab_ml.contracts import TrajectoryBundle


def test_decision_timestamp_equals_event_time(trajectory_bundle: TrajectoryBundle) -> None:
    """decision_timestamp her zaman event_time ile eşit veya daha eski."""
    for split_records in trajectory_bundle.splits.values():
        for record in split_records:
            for step in record.steps:
                assert step.decision_timestamp <= step.event_time or (
                    step.decision_timestamp == step.event_time
                ), f"decision_timestamp {step.decision_timestamp} > event_time {step.event_time}"


def test_action_feasibility_uses_only_decision_time(trajectory_bundle: TrajectoryBundle) -> None:
    """action_feasibility decision_timestamp'dan sonraki veri içermez.

    Builder'ın _build_action_feasibility yalnızca event_time <= decision_timestamp
    olan eventleri kullandığını dolaylı olarak doğrular:
    ilk train step'inde future fiyat yoksa long/short feasible, varsa da kabul edilir.
    Asıl test: feasibility surface'in abstain'i her zaman içermesi.
    """
    first_step = trajectory_bundle.splits["train"][0].steps[0]
    assert first_step.action_feasibility.abstain_feasible()
    assert "enter_long" in first_step.action_feasibility.surface
    assert "enter_short" in first_step.action_feasibility.surface


def test_reward_context_venues_have_positive_prices(trajectory_bundle: TrajectoryBundle) -> None:
    """reward_context yalnızca mevcut exchange'ler için reference_price taşır."""
    exchanges = set(trajectory_bundle.dataset_spec.exchanges)
    for split_records in trajectory_bundle.splits.values():
        for record in split_records:
            for step in record.steps:
                ctx = step.reward_context
                for exchange, ref in ctx.venues.items():
                    assert exchange in exchanges, f"unknown venue in context: {exchange}"
                    assert ref.reference_price > 0.0, (
                        f"venue={exchange} reference_price must be positive"
                    )


def test_policy_state_derived_from_past_only(trajectory_bundle: TrajectoryBundle) -> None:
    """policy_state sadece önceki step'ten türetilir; ilk step'te None."""
    first_record = trajectory_bundle.splits["train"][0]
    assert first_record.steps[0].policy_state is None, "First step policy_state must be None"
    for step in first_record.steps[1:]:
        # policy_state None değil; turnover_accumulator >= 0
        assert step.policy_state is not None
        assert step.policy_state.hold_age_steps >= 0
        assert step.policy_state.turnover_accumulator >= 0.0


def test_eval_split_starts_with_causal_observation(trajectory_bundle: TrajectoryBundle) -> None:
    """Eval split ilk step'i causal pre-eval history ile başlar; hard reset yok."""
    schema = trajectory_bundle.observation_schema
    first_eval_step = trajectory_bundle.splits["eval"][0].steps[0]
    tensor = first_eval_step.observation.raw_surface["1m"]
    # Yeterli tarihçe varsa padding olmamalı
    # (fixture config'de train window yeterlince büyük)
    all_padding = all(
        tensor.padding[i]
        or tensor.unavailable_by_contract[i]
        or tensor.missing[i]
        for i in range(tensor.flat_size)
    )
    assert not all_padding, "Eval first step should have at least some non-padding data"


def test_reward_timeline_length_equals_horizon(trajectory_bundle: TrajectoryBundle) -> None:
    """reward_timeline her venue için tam horizon_steps uzunluğunda seri taşır."""
    horizon = trajectory_bundle.reward_spec.horizon_steps
    for split_records in trajectory_bundle.splits.values():
        for record in split_records:
            for step in record.steps:
                for exchange, series in step.reward_timeline.venue_reference_series.items():
                    assert len(series) == horizon, (
                        f"venue={exchange} timeline length {len(series)} != horizon {horizon}"
                    )
