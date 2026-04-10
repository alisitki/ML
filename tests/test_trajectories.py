from __future__ import annotations

from pathlib import Path

from quantlab_ml.contracts import TrajectoryBundle
from quantlab_ml.contracts.compat import target_stream_series
from quantlab_ml.data import LocalFixtureSource
from quantlab_ml.trajectories import TrajectoryBuilder


def test_trajectory_builder_preserves_axes_and_masks(trajectory_bundle: TrajectoryBundle) -> None:
    schema = trajectory_bundle.observation_schema
    # V2: scale_axis yerine lookback_steps değil
    assert len(schema.scale_axis) == 1
    assert schema.scale_axis[0].label == "1m"
    assert schema.scale_axis[0].num_buckets == 4
    assert len(trajectory_bundle.splits["train"]) >= 1
    assert len(trajectory_bundle.splits["eval"]) >= 1

    first_step = trajectory_bundle.splits["train"][0].steps[0]
    # V2: action_feasibility surface
    assert first_step.action_feasibility.abstain_feasible()
    flat_mask = first_step.action_feasibility.to_flat_mask()
    assert flat_mask["abstain"] is True
    # raw_surface var ve ilk scale için tensor mevcut
    assert "1m" in first_step.observation.raw_surface
    tensor = first_step.observation.raw_surface["1m"]
    assert len(tensor.values) == tensor.flat_size
    tensor = first_step.observation.raw_surface["1m"]
    # En az bir koordinat non-valid olmalı (padding, unavailable veya missing)
    assert any(tensor.padding) or any(tensor.unavailable_by_contract) or any(tensor.missing)


def test_eval_observation_uses_causal_pre_eval_history(trajectory_bundle: TrajectoryBundle) -> None:
    first_eval_step = trajectory_bundle.splits["eval"][0].steps[0]
    tensor = first_eval_step.observation.raw_surface["1m"]
    # eval split'in ilk step'inde padding olmamalı (causal pre-eval history var)
    assert not any(tensor.padding)


def test_masks_distinguish_padding_structural_missing_and_stale(
    trajectory_bundle: TrajectoryBundle,
) -> None:
    """padding | unavailable_by_contract | missing | stale birbirinden bağımsız."""
    first_train_step = trajectory_bundle.splits["train"][0].steps[0]
    tensor = first_train_step.observation.raw_surface["1m"]

    # En az bir padding koordinat mevcut olmalı (tarih dışı)
    assert any(tensor.padding)
    # padding=True olan koordinatlarda unavailable ve missing=False olmalı
    for i, p in enumerate(tensor.padding):
        if p:
            assert tensor.unavailable_by_contract[i] is False
            assert tensor.missing[i] is False

    # binance open_interest unavailable_by_contract=True olmalı
    schema = first_train_step.observation.observation_schema
    binance_oi_unavailable = not schema.stream_available("binance", "open_interest")
    assert binance_oi_unavailable, "binance open_interest should be structurally unavailable"


def test_action_feasibility_does_not_depend_on_future_price(
    trajectory_bundle: TrajectoryBundle,
) -> None:
    """action_feasibility yalnızca decision-time bilgi kullanır."""
    first_step = trajectory_bundle.splits["train"][0].steps[0]
    # decision_timestamp == event_time olmalı
    assert first_step.decision_timestamp == first_step.event_time
    # Feasibility surface tüm action key'leri içermeli
    for action_key in ["abstain", "enter_long", "enter_short"]:
        assert action_key in first_step.action_feasibility.surface


def test_target_symbol_in_step(trajectory_bundle: TrajectoryBundle) -> None:
    """Her step target_symbol taşımalı."""
    first_record = trajectory_bundle.splits["train"][0]
    assert first_record.target_symbol in trajectory_bundle.dataset_spec.symbols
    for step in first_record.steps:
        assert step.target_symbol == first_record.target_symbol


def test_compat_target_stream_series_returns_series(trajectory_bundle: TrajectoryBundle) -> None:
    """Legacy compat: target_stream_series doğru uzunlukta seri döner."""
    first_step = trajectory_bundle.splits["train"][0].steps[0]
    series = target_stream_series(first_step.observation, "mark_price")
    assert len(series) == trajectory_bundle.trajectory_spec.scale_preset[0].num_buckets
    # Son eleman None değil (veri var)
    assert series[-1] is not None
