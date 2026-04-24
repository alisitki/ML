from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from quantlab_ml.contracts import DatasetSpec, NormalizedMarketEvent
from quantlab_ml.trajectories import TrajectoryBuilder
from quantlab_ml.trajectories.event_token_cache import (
    EventTokenCacheSplitWriter,
    _EventCandidate,
    _InformativeUnit,
    _bbo_payload,
    _canonical_payload_hash,
    _canonical_payload_identity,
    _trade_payload,
    event_token_cache_directory,
    event_token_cache_manifest_path,
    load_event_token_cache_shard,
    read_event_token_cache_diagnostics,
    read_event_token_cache_manifest,
)


def _overflow_dataset_spec() -> DatasetSpec:
    return DatasetSpec.model_validate(
        {
            "dataset_hash": "event-cache-overflow-fixture",
            "slice_id": "event-cache-overflow-fixture",
            "exchanges": ["binance", "bybit", "okx"],
            "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "stream_universe": ["trade", "bbo", "mark_price"],
            "available_streams_by_exchange": {
                "binance": ["trade", "bbo", "mark_price"],
                "bybit": ["trade", "bbo", "mark_price"],
                "okx": ["trade", "bbo", "mark_price"],
            },
            "train_range": {
                "start": "2024-01-01T00:01:00Z",
                "end": "2024-01-01T00:02:00Z",
            },
            "validation_range": {
                "start": "2024-01-01T00:03:00Z",
                "end": "2024-01-01T00:04:00Z",
            },
            "final_untouched_test_range": {
                "start": "2024-01-01T00:05:00Z",
                "end": "2024-01-01T00:06:00Z",
            },
            "walkforward": {
                "train_window_steps": 2,
                "validation_window_steps": 2,
                "step_size_steps": 1,
            },
            "sampling_interval_seconds": 60,
        }
    )


def _overflow_events(dataset_spec: DatasetSpec) -> list[NormalizedMarketEvent]:
    events: list[NormalizedMarketEvent] = []
    source_event_index = 0
    base_prices = {"BTCUSDT": 100.0, "ETHUSDT": 50.0, "SOLUSDT": 20.0}
    venue_shift = {"binance": 0.0, "bybit": 0.2, "okx": -0.2}
    start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)

    for minute in range(0, 7):
        event_time = start + timedelta(minutes=minute)
        for exchange, shift in venue_shift.items():
            for symbol, base_price in base_prices.items():
                events.append(
                    NormalizedMarketEvent.model_validate(
                        {
                            "event_time": event_time,
                            "exchange": exchange,
                            "symbol": symbol,
                            "stream_type": "mark_price",
                            "fields": {
                                "mark_price": base_price + shift + minute,
                                "event_delta": 1.0 if minute > 0 else 0.0,
                                "index_price_if_available": base_price + shift + minute,
                            },
                            "ingest_metadata": {
                                "source": "synthetic://mark_price",
                                "source_event_index": source_event_index,
                            },
                        }
                    )
                )
                source_event_index += 1

    for tick in range(0, 20):
        for exchange_index, (exchange, shift) in enumerate(venue_shift.items()):
            for symbol_index, (symbol, base_price) in enumerate(base_prices.items()):
                event_time = start + timedelta(
                    milliseconds=(tick * 100) + (exchange_index * 7) + (symbol_index * 3)
                )
                price = base_price + shift + (tick * 0.01)
                qty = 1.0 + (symbol_index * 0.1)
                signed_flow = qty if (tick + symbol_index) % 2 == 0 else -qty
                events.append(
                    NormalizedMarketEvent.model_validate(
                        {
                            "event_time": event_time,
                            "exchange": exchange,
                            "symbol": symbol,
                            "stream_type": "trade",
                            "fields": {
                                "price": price,
                                "qty": qty,
                                "side_or_signed_flow_proxy": signed_flow,
                                "event_delta": 0.01 if tick > 0 else 0.0,
                                "count_or_burst": tick + 1,
                            },
                            "ingest_metadata": {
                                "source": "synthetic://overflow",
                                "source_event_index": source_event_index,
                            },
                        }
                    )
                )
                source_event_index += 1
                events.append(
                    NormalizedMarketEvent.model_validate(
                        {
                            "event_time": event_time + timedelta(milliseconds=1),
                            "exchange": exchange,
                            "symbol": symbol,
                            "stream_type": "bbo",
                            "fields": {
                                "bid_price": price - 0.05,
                                "ask_price": price + 0.05,
                                "bid_size": 2.0 + symbol_index,
                                "ask_size": 1.5 + symbol_index,
                                "spread": 0.1,
                                "mid": price,
                                "imbalance_inputs": (0.5 / (3.5 + (2 * symbol_index))),
                            },
                            "ingest_metadata": {
                                "source": "synthetic://overflow",
                                "source_event_index": source_event_index,
                            },
                        }
                    )
                )
                source_event_index += 1
    return events


def test_event_token_cache_compresses_bbo_flood_and_preserves_symbol_structure(
    tmp_path: Path,
    training_bundle,
    reward_spec,
) -> None:
    dataset_spec = _overflow_dataset_spec()
    trajectory_spec, action_space, _ = training_bundle
    builder = TrajectoryBuilder(dataset_spec, trajectory_spec, action_space, reward_spec)
    builder.build_to_directory(_overflow_events(dataset_spec), tmp_path)

    manifest = read_event_token_cache_manifest(tmp_path)
    diagnostics = read_event_token_cache_diagnostics(tmp_path)
    train_split = manifest.splits["train"]
    first_shard = load_event_token_cache_shard(tmp_path, train_split.shards[0])
    first_row = first_shard.window_stats[0]
    train_diag = diagnostics.splits["train"]

    assert train_split.row_count == 3
    assert first_row.candidate_token_count > 256
    assert first_row.informative_candidate_count < first_row.candidate_token_count
    assert first_row.selected_token_count <= 256
    assert set(first_row.retained_by_symbol) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    assert first_row.target_symbol_retained_rate is not None
    assert 0.0 < first_row.target_symbol_retained_rate <= 1.0
    assert first_row.raw_target_symbol_retained_rate is not None
    assert 0.0 < first_row.raw_target_symbol_retained_rate <= 1.0
    assert first_row.symbol_with_zero_retained_tokens_count == 0
    assert first_row.burst_count > 0
    assert first_row.burst_retention_rate is not None
    assert 0.0 < first_row.burst_retention_rate <= 1.0
    assert first_row.drop_reason_counts_by_tier["COMPRESSION"]["lost_after_compression"] > 0
    assert first_row.significant_bbo_emitted_count_by_reason
    assert first_row.has_cross_venue_ordered_adjacency is True
    assert first_row.has_trade_to_bbo_ordered_adjacency is True

    assert train_diag.informative_candidate_total < train_diag.candidate_token_total
    assert train_diag.weighted_target_symbol_retained_rate is not None
    assert train_diag.weighted_raw_target_symbol_retained_rate is not None
    assert train_diag.weighted_target_trade_retained_rate is not None
    assert train_diag.weighted_target_bbo_sig_retained_rate is not None
    assert train_diag.weighted_burst_retention_rate is not None
    assert train_diag.cross_venue_ordered_adjacency_rate > 0.0
    assert train_diag.trade_to_bbo_ordered_adjacency_rate > 0.0
    assert train_diag.significant_bbo_preservation_rate is not None
    assert train_diag.informative_candidate_by_tier
    assert train_diag.t4_anchor_total >= 0
    assert train_diag.t4_candidate_total >= 0
    assert train_diag.t4_resolution_wall_sec >= 0.0
    assert train_diag.bbo_significance_wall_sec >= 0.0
    assert train_diag.quota_fill_wall_sec >= 0.0
    assert train_diag.diagnostics_serialization_wall_sec >= 0.0
    assert train_diag.total_selector_wall_sec >= train_diag.diagnostics_serialization_wall_sec
    partial_profile_path = event_token_cache_directory(tmp_path) / "train_partial_selector_profile.json"
    partial_profile = json.loads(partial_profile_path.read_text(encoding="utf-8"))
    assert partial_profile["partial_split_completion_status"] == "complete"
    assert partial_profile["rows_processed"] == train_split.row_count
    assert partial_profile["raw_candidate_count"] == train_diag.candidate_token_total
    assert partial_profile["post_compression_informative_unit_count"] == train_diag.informative_candidate_total
    assert partial_profile["t4_candidate_count"] == train_diag.t4_candidate_total
    assert partial_profile["t4_anchor_count"] == train_diag.t4_anchor_total
    assert set(partial_profile["tier_counts"]) == {"T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"}
    assert sum(partial_profile["tier_counts"].values()) == train_diag.informative_candidate_total
    assert partial_profile["window_base_cache_miss_count"] >= 1
    assert partial_profile["window_base_cache_hit_count"] >= 1
    assert partial_profile["partial_profile_write_wall_sec"] >= 0.0


def test_partial_selector_profile_is_written_before_event_cache_manifest(tmp_path: Path) -> None:
    dataset_spec = _overflow_dataset_spec()
    writer = EventTokenCacheSplitWriter(
        directory=tmp_path,
        split_name="development",
        dataset_spec=dataset_spec,
        indexed={},
        source_labels=["synthetic://test"],
    )
    partial_profile_path = event_token_cache_directory(tmp_path) / "development_partial_selector_profile.json"

    initialized_profile = json.loads(partial_profile_path.read_text(encoding="utf-8"))
    assert initialized_profile["partial_split_completion_status"] == "initialized"
    assert initialized_profile["rows_processed"] == 0

    writer._append_step(
        record=SimpleNamespace(target_symbol="BTCUSDT", trajectory_id="trajectory-0"),
        step=SimpleNamespace(event_time=datetime(2024, 1, 1, 0, 1, tzinfo=UTC)),
        trajectory_start=True,
    )

    partial_profile = json.loads(partial_profile_path.read_text(encoding="utf-8"))
    assert partial_profile["partial_split_completion_status"] == "in_progress"
    assert partial_profile["rows_processed"] == 1
    assert partial_profile["raw_candidate_count"] == 0
    assert partial_profile["post_compression_informative_unit_count"] == 0
    assert set(partial_profile["tier_counts"]) == {"T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"}
    assert not event_token_cache_manifest_path(tmp_path).exists()


def test_bbo_significance_uses_canonical_reason_precedence(
    tmp_path: Path,
) -> None:
    dataset_spec = _overflow_dataset_spec()
    writer = EventTokenCacheSplitWriter(
        directory=tmp_path,
        split_name="train",
        dataset_spec=dataset_spec,
        indexed={},
        source_labels=["synthetic://test"],
    )
    burst_start = _FakeCompactEvent(
        event_time_ts=1.0,
        fields={
            "bid_price": 100.0,
            "ask_price": 100.1,
            "bid_size": 12.0,
            "ask_size": 8.0,
        },
        source_label_id=0,
        source_event_index=0,
    )
    candidate_event = _FakeCompactEvent(
        event_time_ts=1.2,
        fields={
            "bid_price": 100.4,
            "ask_price": 100.8,
            "bid_size": 1.0,
            "ask_size": 9.0,
        },
        source_label_id=0,
        source_event_index=1,
    )
    candidate = _EventCandidate(
        event_time_ts=candidate_event.event_time_ts,
        exchange="binance",
        symbol="BTCUSDT",
        stream="bbo",
        event=candidate_event,
        lane_events=[burst_start, candidate_event],
        lane_position=1,
    )
    start_candidate = _EventCandidate(
        event_time_ts=burst_start.event_time_ts,
        exchange="binance",
        symbol="BTCUSDT",
        stream="bbo",
        event=burst_start,
        lane_events=[burst_start, candidate_event],
        lane_position=0,
    )

    matched_reasons, salience = writer._bbo_significance_assessment(
        burst_start=start_candidate,
        candidate=candidate,
    )

    assert {"liquidity_vacuum", "spread_regime_jump", "mid_excursion", "imbalance_regime_flip"} <= matched_reasons
    assert salience > 0.0
    units = {}
    writer._upsert_informative_unit(
        units_by_key=units,
        candidate=candidate,
        source_bucket="bbo_recent_sig",
        decision_time_ms=2_000,
        lane_key=("binance", "BTCUSDT", "bbo"),
        burst_id=(("binance", "BTCUSDT", "bbo"), 0, 1),
        salience=salience,
        emission_tag="significant",
        matched_reasons=matched_reasons,
    )
    unit = next(iter(units.values()))
    assert unit.canonical_significance_reason == "liquidity_vacuum"
    assert unit.matched_reasons == matched_reasons


def test_t4_anchor_resolution_prefers_nearest_then_higher_priority_target_anchor(
    tmp_path: Path,
) -> None:
    dataset_spec = _overflow_dataset_spec()
    writer = EventTokenCacheSplitWriter(
        directory=tmp_path,
        split_name="train",
        dataset_spec=dataset_spec,
        indexed={},
        source_labels=["synthetic://test"],
    )

    def _unit(
        *,
        event_time_ts: float,
        exchange: str,
        symbol: str,
        stream: str,
        source_bucket: str,
        salience: float,
        source_event_index: int,
    ) -> _InformativeUnit:
        event = _FakeCompactEvent(
            event_time_ts=event_time_ts,
            fields={},
            source_label_id=0,
            source_event_index=source_event_index,
        )
        candidate = _EventCandidate(
            event_time_ts=event_time_ts,
            exchange=exchange,
            symbol=symbol,
            stream=stream,
            event=event,
            lane_events=[event],
            lane_position=0,
        )
        return _InformativeUnit(
            candidate=candidate,
            lag_ms=0,
            source_bucket=source_bucket,
            lane_key=(exchange, symbol, stream),
            burst_id=((exchange, symbol, stream), source_event_index),
            salience=salience,
        )

    target_trade = _unit(
        event_time_ts=10.040,
        exchange="binance",
        symbol="BTCUSDT",
        stream="trade",
        source_bucket="trade_recent_raw",
        salience=3.0,
        source_event_index=1,
    )
    target_bbo = _unit(
        event_time_ts=10.200,
        exchange="binance",
        symbol="BTCUSDT",
        stream="bbo",
        source_bucket="bbo_recent_sig",
        salience=9.0,
        source_event_index=2,
    )
    non_target = _unit(
        event_time_ts=10.120,
        exchange="binance",
        symbol="ETHUSDT",
        stream="trade",
        source_bucket="trade_recent_raw",
        salience=1.0,
        source_event_index=3,
    )
    informative_units = [target_trade, target_bbo, non_target]

    writer._assign_priority_tiers(informative_units=informative_units, target_symbol="BTCUSDT")

    assert target_trade.priority_tier == "T0"
    assert target_bbo.priority_tier == "T1"
    assert non_target.priority_tier == "T4"
    assert non_target.best_anchor_key == (
        target_trade.event_time_ts,
        target_trade.source_label_id,
        target_trade.source_event_index,
        target_trade.symbol,
        target_trade.exchange,
        target_trade.stream,
    )
    assert non_target.best_anchor_tier == "T0"
    assert non_target.best_anchor_delta_ms == 80


def test_indexed_t4_anchor_resolution_matches_naive_semantics(tmp_path: Path) -> None:
    dataset_spec = _overflow_dataset_spec()
    writer = EventTokenCacheSplitWriter(
        directory=tmp_path,
        split_name="train",
        dataset_spec=dataset_spec,
        indexed={},
        source_labels=["synthetic://test"],
    )

    def _unit(
        *,
        event_time_ts: float,
        exchange: str,
        symbol: str,
        stream: str,
        source_bucket: str,
        salience: float,
        source_event_index: int,
    ) -> _InformativeUnit:
        event = _FakeCompactEvent(
            event_time_ts=event_time_ts,
            fields={},
            source_label_id=0,
            source_event_index=source_event_index,
        )
        candidate = _EventCandidate(
            event_time_ts=event_time_ts,
            exchange=exchange,
            symbol=symbol,
            stream=stream,
            event=event,
            lane_events=[event],
            lane_position=0,
        )
        return _InformativeUnit(
            candidate=candidate,
            lag_ms=0,
            source_bucket=source_bucket,
            lane_key=(exchange, symbol, stream),
            burst_id=((exchange, symbol, stream), source_event_index),
            salience=salience,
        )

    anchors = [
        _unit(
            event_time_ts=20.000,
            exchange="binance",
            symbol="BTCUSDT",
            stream="trade",
            source_bucket="trade_recent_raw",
            salience=5.0,
            source_event_index=10,
        ),
        _unit(
            event_time_ts=20.000,
            exchange="binance",
            symbol="BTCUSDT",
            stream="trade",
            source_bucket="trade_recent_raw",
            salience=5.0,
            source_event_index=9,
        ),
        _unit(
            event_time_ts=20.210,
            exchange="binance",
            symbol="BTCUSDT",
            stream="bbo",
            source_bucket="bbo_recent_sig",
            salience=99.0,
            source_event_index=11,
        ),
        _unit(
            event_time_ts=20.010,
            exchange="bybit",
            symbol="BTCUSDT",
            stream="trade",
            source_bucket="trade_recent_raw",
            salience=1.0,
            source_event_index=12,
        ),
    ]
    non_targets = [
        _unit(
            event_time_ts=20.010,
            exchange="binance",
            symbol="ETHUSDT",
            stream="trade",
            source_bucket="trade_recent_raw",
            salience=1.0,
            source_event_index=20,
        ),
        _unit(
            event_time_ts=21.300,
            exchange="binance",
            symbol="SOLUSDT",
            stream="trade",
            source_bucket="trade_recent_raw",
            salience=1.0,
            source_event_index=21,
        ),
        _unit(
            event_time_ts=20.010,
            exchange="okx",
            symbol="ETHUSDT",
            stream="trade",
            source_bucket="trade_recent_raw",
            salience=1.0,
            source_event_index=22,
        ),
    ]
    informative_units = anchors + non_targets

    writer._assign_priority_tiers(informative_units=informative_units, target_symbol="BTCUSDT")

    expected_by_source_index = {
        unit.source_event_index: _naive_t4_best_anchor(unit=unit, anchors=anchors)
        for unit in non_targets
    }
    for unit in non_targets:
        expected = expected_by_source_index[unit.source_event_index]
        if expected is None:
            assert unit.priority_tier in {"T5", "T6", "T7"}
            assert unit.best_anchor_key is None
        else:
            assert unit.priority_tier == "T4"
            assert unit.best_anchor_key == (
                expected.event_time_ts,
                expected.source_label_id,
                expected.source_event_index,
                expected.symbol,
                expected.exchange,
                expected.stream,
            )


def test_t4_anchor_resolution_matches_naive_semantics_across_dense_ties(tmp_path: Path) -> None:
    dataset_spec = _overflow_dataset_spec()
    writer = EventTokenCacheSplitWriter(
        directory=tmp_path,
        split_name="train",
        dataset_spec=dataset_spec,
        indexed={},
        source_labels=["synthetic://test"],
    )

    anchors = [
        _make_unit(
            event_time_ts=30.0000,
            exchange="binance",
            symbol="BTCUSDT",
            stream="trade",
            source_bucket="trade_recent_raw",
            salience=4.0,
            source_event_index=1,
        ),
        _make_unit(
            event_time_ts=30.0004,
            exchange="binance",
            symbol="BTCUSDT",
            stream="bbo",
            source_bucket="bbo_recent_sig",
            salience=50.0,
            source_event_index=2,
        ),
        _make_unit(
            event_time_ts=30.0008,
            exchange="binance",
            symbol="BTCUSDT",
            stream="trade",
            source_bucket="trade_recent_raw",
            salience=1.0,
            source_event_index=0,
        ),
        _make_unit(
            event_time_ts=30.0500,
            exchange="bybit",
            symbol="BTCUSDT",
            stream="trade",
            source_bucket="trade_recent_raw",
            salience=100.0,
            source_event_index=3,
        ),
    ]
    non_targets = [
        _make_unit(
            event_time_ts=30.0009,
            exchange="binance",
            symbol="ETHUSDT",
            stream="trade",
            source_bucket="trade_recent_raw",
            salience=1.0,
            source_event_index=10,
        ),
        _make_unit(
            event_time_ts=29.0010,
            exchange="binance",
            symbol="SOLUSDT",
            stream="bbo",
            source_bucket="bbo_recent_sig",
            salience=2.0,
            source_event_index=11,
        ),
    ]
    informative_units = anchors + non_targets

    writer._assign_priority_tiers(informative_units=informative_units, target_symbol="BTCUSDT")

    for unit in non_targets:
        expected = _naive_t4_best_anchor(unit=unit, anchors=anchors)
        if expected is None:
            assert unit.priority_tier in {"T5", "T6", "T7"}
            assert unit.best_anchor_key is None
        else:
            assert unit.priority_tier == "T4"
            assert unit.best_anchor_key == (
                expected.event_time_ts,
                expected.source_label_id,
                expected.source_event_index,
                expected.symbol,
                expected.exchange,
                expected.stream,
            )
            assert unit.best_anchor_tier == expected.priority_tier


def test_canonical_payload_identity_preserves_hash_dedupe_semantics() -> None:
    first = {
        "price": 100.0,
        "qty": 1,
        "is_buyer_maker": False,
        "nested_ignored": {"not": "canonical"},
        "none_value": None,
    }
    same = {
        "none_value": None,
        "nested_ignored": ["also", "ignored"],
        "is_buyer_maker": False,
        "qty": 1,
        "price": 100.0,
    }
    different_numeric_type = {
        "price": 100,
        "qty": 1,
        "is_buyer_maker": False,
        "none_value": None,
    }
    different_bool_type = {
        "price": 100.0,
        "qty": 1,
        "is_buyer_maker": 0,
        "none_value": None,
    }

    assert _canonical_payload_hash(first) == _canonical_payload_hash(same)
    assert _canonical_payload_identity(first) == _canonical_payload_identity(same)
    assert _canonical_payload_hash(first) != _canonical_payload_hash(different_numeric_type)
    assert _canonical_payload_identity(first) != _canonical_payload_identity(different_numeric_type)
    assert _canonical_payload_hash(first) != _canonical_payload_hash(different_bool_type)
    assert _canonical_payload_identity(first) != _canonical_payload_identity(different_bool_type)


def test_r5_window_base_optimized_path_matches_slow_reference_ordered_output(tmp_path: Path) -> None:
    writer, decision_time = _r5_window_base_writer(tmp_path)

    for offset_seconds in (0, 1, 60):
        row_time = decision_time + timedelta(seconds=offset_seconds)
        slow = writer._compute_window_base_slow_reference(decision_time=row_time)
        optimized = writer._compute_window_base_optimized(decision_time=row_time)

        assert [_candidate_snapshot(candidate) for candidate in optimized.raw_candidates] == [
            _candidate_snapshot(candidate) for candidate in slow.raw_candidates
        ]
        assert [_candidate_snapshot(candidate) for candidate in optimized.deduped_candidates] == [
            _candidate_snapshot(candidate) for candidate in slow.deduped_candidates
        ]
        assert _window_base_snapshot(optimized.window_base) == _window_base_snapshot(slow.window_base)


def test_r5_window_base_precompute_does_not_leak_future_events_or_bursts(tmp_path: Path) -> None:
    clean_writer, decision_time = _r5_window_base_writer(tmp_path / "clean", include_future_events=False)
    future_writer, _ = _r5_window_base_writer(tmp_path / "future", include_future_events=True)

    clean = clean_writer._compute_window_base_optimized(decision_time=decision_time)
    future = future_writer._compute_window_base_optimized(decision_time=decision_time)
    future_slow = future_writer._compute_window_base_slow_reference(decision_time=decision_time)

    assert [_candidate_snapshot(candidate) for candidate in future.raw_candidates] == [
        _candidate_snapshot(candidate) for candidate in clean.raw_candidates
    ]
    assert [_candidate_snapshot(candidate) for candidate in future.deduped_candidates] == [
        _candidate_snapshot(candidate) for candidate in clean.deduped_candidates
    ]
    assert _window_base_snapshot(future.window_base) == _window_base_snapshot(clean.window_base)
    assert _window_base_snapshot(future.window_base) == _window_base_snapshot(future_slow.window_base)


def test_r5_bbo_burst_clipping_and_reason_precedence_match_slow_reference(tmp_path: Path) -> None:
    writer, decision_time = _r5_window_base_writer(tmp_path)
    row_times = [
        decision_time,
        decision_time + timedelta(seconds=1),
        decision_time + timedelta(seconds=60),
    ]

    for row_time in row_times:
        slow = writer._compute_window_base_slow_reference(decision_time=row_time)
        optimized = writer._compute_window_base_optimized(decision_time=row_time)
        assert _window_base_snapshot(optimized.window_base) == _window_base_snapshot(slow.window_base)

    optimized = writer._compute_window_base_optimized(decision_time=decision_time)
    units_by_source = {unit.source_event_index: unit for unit in optimized.window_base.informative_units}
    collision_unit = units_by_source[1005]
    burst_end_unit = units_by_source[1006]

    assert 1001 not in units_by_source
    assert 1002 not in units_by_source
    assert collision_unit.canonical_significance_reason == "liquidity_vacuum"
    assert {"liquidity_vacuum", "spread_regime_jump", "mid_excursion", "imbalance_regime_flip"} <= (
        collision_unit.matched_reasons
    )
    assert "burst_boundary" in burst_end_unit.matched_reasons


def test_r5_bbo_identity_precompute_preserves_lane_and_source_boundaries(tmp_path: Path) -> None:
    writer, decision_time = _r5_window_base_writer(tmp_path)

    slow = writer._compute_window_base_slow_reference(decision_time=decision_time)
    optimized = writer._compute_window_base_optimized(decision_time=decision_time)
    deduped = [_candidate_snapshot(candidate) for candidate in optimized.deduped_candidates]
    same_timestamp_ms = int((decision_time.timestamp() - 30.000) * 1000)
    duplicate_like_ms = int((decision_time.timestamp() - 59.890) * 1000)

    assert _window_base_snapshot(optimized.window_base) == _window_base_snapshot(slow.window_base)
    assert optimized.window_base.duplicate_count == slow.window_base.duplicate_count == 1
    assert (
        same_timestamp_ms,
        "bybit",
        "BTCUSDT",
        "bbo",
        1,
        2000,
        _canonical_payload_identity(_bbo_fields(mid=100.0, bid_size=10.0, ask_size=10.0)),
    ) in deduped
    assert (
        same_timestamp_ms,
        "binance",
        "ETHUSDT",
        "bbo",
        1,
        3000,
        _canonical_payload_identity(_bbo_fields(mid=100.0, bid_size=10.0, ask_size=10.0)),
    ) in deduped
    assert (
        duplicate_like_ms,
        "binance",
        "BTCUSDT",
        "bbo",
        1,
        1005,
        _canonical_payload_identity(_bbo_fields(mid=101.0, bid_size=1, ask_size=9.0, spread=0.6)),
    ) in deduped
    assert (
        duplicate_like_ms,
        "binance",
        "BTCUSDT",
        "bbo",
        1,
        1006,
        _canonical_payload_identity(_bbo_fields(mid=101.0, bid_size=1.0, ask_size=9.0, spread=0.6)),
    ) in deduped


def _naive_t4_best_anchor(
    *,
    unit: _InformativeUnit,
    anchors: list[_InformativeUnit],
) -> _InformativeUnit | None:
    matches: list[tuple[int, int, float, float, int, int, _InformativeUnit]] = []
    for anchor in anchors:
        delta_ms = abs(int((unit.event_time_ts - anchor.event_time_ts) * 1000))
        if delta_ms > 1000:
            continue
        if not (
            (unit.symbol == anchor.symbol and unit.exchange != anchor.exchange)
            or (unit.exchange == anchor.exchange and unit.symbol != anchor.symbol)
        ):
            continue
        matches.append(
            (
                delta_ms,
                {"T0": 0, "T1": 1, "T2": 2, "T3": 3}[anchor.priority_tier or "T3"],
                -anchor.salience,
                -anchor.event_time_ts,
                anchor.source_label_id,
                anchor.source_event_index,
                anchor,
            )
        )
    return min(matches)[-1] if matches else None


def _make_unit(
    *,
    event_time_ts: float,
    exchange: str,
    symbol: str,
    stream: str,
    source_bucket: str,
    salience: float,
    source_event_index: int,
) -> _InformativeUnit:
    event = _FakeCompactEvent(
        event_time_ts=event_time_ts,
        fields={},
        source_label_id=0,
        source_event_index=source_event_index,
    )
    candidate = _EventCandidate(
        event_time_ts=event_time_ts,
        exchange=exchange,
        symbol=symbol,
        stream=stream,
        event=event,
        lane_events=[event],
        lane_position=0,
    )
    return _InformativeUnit(
        candidate=candidate,
        lag_ms=0,
        source_bucket=source_bucket,
        lane_key=(exchange, symbol, stream),
        burst_id=((exchange, symbol, stream), source_event_index),
        salience=salience,
    )


@dataclass(slots=True)
class _FakeCompactEvent:
    event_time_ts: float
    fields: dict[str, object]
    source_label_id: int = 0
    source_event_index: int = 0


def _r5_window_base_writer(
    tmp_path: Path,
    *,
    include_future_events: bool = False,
) -> tuple[EventTokenCacheSplitWriter, datetime]:
    dataset_spec = _overflow_dataset_spec()
    decision_time = datetime(2024, 1, 1, 0, 59, tzinfo=UTC)
    base_ts = decision_time.timestamp()
    lanes: dict[tuple[str, str, str], list[_FakeCompactEvent]] = {
        ("BTCUSDT", "binance", "bbo"): [
            _fake_event(base_ts - 60.200, _bbo_fields(mid=90.0, bid_size=50.0, ask_size=50.0), 1, 1000),
            _fake_event(base_ts - 60.000, _bbo_fields(mid=100.0, bid_size=12.0, ask_size=8.0), 1, 1001),
            _fake_event(base_ts - 59.950, _bbo_fields(mid=100.0, bid_size=12.0, ask_size=8.0), 1, 1002),
            _fake_event(base_ts - 59.900, _bbo_fields(mid=101.0, bid_size=1.0, ask_size=9.0, spread=0.6), 1, 1003),
            _fake_event(base_ts - 59.900, _bbo_fields(mid=101.0, bid_size=1.0, ask_size=9.0, spread=0.6), 1, 1004),
            _fake_event(base_ts - 59.890, _bbo_fields(mid=101.0, bid_size=1, ask_size=9.0, spread=0.6), 1, 1005),
            _fake_event(base_ts - 59.890, _bbo_fields(mid=101.0, bid_size=1.0, ask_size=9.0, spread=0.6), 1, 1006),
            _fake_event(base_ts - 30.000, _bbo_fields(mid=100.0, bid_size=10.0, ask_size=10.0), 1, 1008),
            _fake_event(base_ts - 0.150, _bbo_fields(mid=102.0, bid_size=11.0, ask_size=11.0), 1, 1009),
            _fake_event(base_ts - 0.050, _bbo_fields(mid=103.0, bid_size=2.0, ask_size=12.0), 1, 1010),
            _fake_event(base_ts + 60.000, _bbo_fields(mid=103.5, bid_size=2.0, ask_size=12.0), 1, 1011),
        ],
        ("BTCUSDT", "binance", "trade"): [
            _fake_event(base_ts - 30.000, {"price": 100.1, "qty": 1.0, "side_or_signed_flow_proxy": 1.0}, 1, 1012),
            _fake_event(base_ts - 30.000, {"price": 100.2, "qty": 2.0, "side_or_signed_flow_proxy": -2.0}, 1, 1014),
            _fake_event(base_ts - 30.000, {"price": 100.3, "qty": 3.0, "side_or_signed_flow_proxy": 3.0}, 1, 1013),
        ],
        ("BTCUSDT", "bybit", "bbo"): [
            _fake_event(base_ts - 30.000, _bbo_fields(mid=100.0, bid_size=10.0, ask_size=10.0), 1, 2000),
        ],
        ("ETHUSDT", "binance", "bbo"): [
            _fake_event(base_ts - 30.000, _bbo_fields(mid=100.0, bid_size=10.0, ask_size=10.0), 1, 3000),
            _fake_event(base_ts - 29.900, _bbo_fields(mid=100.5, bid_size=4.0, ask_size=14.0), 1, 3001),
        ],
        ("SOLUSDT", "okx", "trade"): [
            _fake_event(base_ts - 30.000, {"price": 20.0, "qty": 1.0, "side_or_signed_flow_proxy": 1.0}, 1, 4000),
        ],
    }
    if include_future_events:
        lanes[("BTCUSDT", "binance", "bbo")].append(
            _fake_event(base_ts + 0.100, _bbo_fields(mid=180.0, bid_size=0.1, ask_size=100.0), 1, 1015)
        )
    indexed = {
        lane_key: (np.asarray([event.event_time_ts for event in events], dtype=float), events)
        for lane_key, events in lanes.items()
    }
    writer = EventTokenCacheSplitWriter(
        directory=tmp_path,
        split_name="train",
        dataset_spec=dataset_spec,
        indexed=indexed,
        source_labels=["synthetic://r5"],
    )
    return writer, decision_time


def _fake_event(
    event_time_ts: float,
    fields: dict[str, object],
    source_label_id: int,
    source_event_index: int,
) -> _FakeCompactEvent:
    return _FakeCompactEvent(
        event_time_ts=event_time_ts,
        fields=fields,
        source_label_id=source_label_id,
        source_event_index=source_event_index,
    )


def _bbo_fields(
    *,
    mid: float,
    bid_size: object,
    ask_size: object,
    spread: float = 0.2,
) -> dict[str, object]:
    bid_price = mid - (spread / 2.0)
    ask_price = mid + (spread / 2.0)
    return {
        "bid_price": bid_price,
        "ask_price": ask_price,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "spread": spread,
        "mid": mid,
    }


def _candidate_snapshot(
    candidate: _EventCandidate,
) -> tuple[int, str, str, str, int, int, tuple[tuple[str, tuple[str, object]], ...]]:
    return (
        int(candidate.event_time_ts * 1000),
        candidate.exchange,
        candidate.symbol,
        candidate.stream,
        candidate.source_label_id,
        candidate.source_event_index,
        _canonical_payload_identity(candidate.event.fields),
    )


def _window_base_snapshot(window_base) -> dict[str, object]:
    return {
        "units": [
            (
                _candidate_snapshot(unit.candidate),
                unit.source_bucket,
                unit.lane_key,
                unit.burst_id,
                round(unit.salience, 12),
                sorted(unit.emission_tags),
                sorted(unit.matched_reasons),
                unit.canonical_significance_reason,
            )
            for unit in window_base.informative_units
        ],
        "supported_lane_count": window_base.supported_lane_count,
        "latest_reference_by_symbol": dict(window_base.latest_reference_by_symbol),
        "deduped_count": window_base.deduped_count,
        "duplicate_count": window_base.duplicate_count,
        "duplicate_dropped_by_stream": dict(window_base.duplicate_dropped_by_stream),
        "duplicate_dropped_by_venue": dict(window_base.duplicate_dropped_by_venue),
        "same_timestamp_tie_count": window_base.same_timestamp_tie_count,
        "source_order_inversion_count": window_base.source_order_inversion_count,
        "candidate_by_symbol": dict(window_base.candidate_by_symbol),
        "candidate_by_venue": dict(window_base.candidate_by_venue),
        "candidate_by_stream": dict(window_base.candidate_by_stream),
    }


def test_event_token_payload_formulas_are_frozen() -> None:
    previous_trade = _FakeCompactEvent(
        event_time_ts=1.0,
        fields={"price": 100.0, "qty": 2.0, "side_or_signed_flow_proxy": 2.0},
        source_label_id=0,
        source_event_index=0,
    )
    current_trade = _FakeCompactEvent(
        event_time_ts=1.1,
        fields={
            "price": 100.5,
            "qty": 3.0,
            "aggressor_side": "sell",
            "side_or_signed_flow_proxy": -3.0,
        },
        source_label_id=0,
        source_event_index=1,
    )
    trade_candidate = _EventCandidate(
        event_time_ts=current_trade.event_time_ts,
        exchange="binance",
        symbol="BTCUSDT",
        stream="trade",
        event=current_trade,
        lane_events=[previous_trade, current_trade],
        lane_position=1,
    )
    trade_values, trade_presence = _trade_payload(trade_candidate, 250)

    assert trade_values == [100.5, 3.0, -3.0, 0.5, 2.0]
    assert trade_presence == [True, True, True, True, True]

    current_bbo = _FakeCompactEvent(
        event_time_ts=1.2,
        fields={
            "bid_price": 100.4,
            "ask_price": 100.6,
            "bid_size": 2.5,
            "ask_size": 1.5,
        },
        source_label_id=0,
        source_event_index=2,
    )
    bbo_candidate = _EventCandidate(
        event_time_ts=current_bbo.event_time_ts,
        exchange="binance",
        symbol="BTCUSDT",
        stream="bbo",
        event=current_bbo,
        lane_events=[current_bbo],
        lane_position=0,
    )
    bbo_values, bbo_presence = _bbo_payload(bbo_candidate)

    assert bbo_values == [100.4, 100.6, 2.5, 1.5, 0.19999999999998863, 100.5, 0.25]
    assert bbo_presence == [True, True, True, True, True, True, True]
