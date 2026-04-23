from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from quantlab_ml.contracts import DatasetSpec, NormalizedMarketEvent
from quantlab_ml.trajectories import TrajectoryBuilder
from quantlab_ml.trajectories.event_token_cache import (
    EventTokenCacheSplitWriter,
    _EventCandidate,
    _InformativeUnit,
    _bbo_payload,
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


@dataclass(slots=True)
class _FakeCompactEvent:
    event_time_ts: float
    fields: dict[str, float | str | bool]
    source_label_id: int = 0
    source_event_index: int = 0


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
