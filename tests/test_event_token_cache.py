from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from quantlab_ml.contracts import DatasetSpec, NormalizedMarketEvent
from quantlab_ml.trajectories import TrajectoryBuilder
from quantlab_ml.trajectories.event_token_cache import (
    _EventCandidate,
    _bbo_payload,
    _trade_payload,
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


def test_event_token_cache_reports_overflow_and_symbol_fairness(
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
    assert first_row.selected_token_count == 256
    assert first_row.dropped_token_count > 0
    assert set(first_row.dropped_by_symbol) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    assert set(first_row.retained_by_symbol) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    assert first_row.target_symbol_retained_rate is not None
    assert 0.0 < first_row.target_symbol_retained_rate <= 1.0
    assert first_row.symbol_with_zero_retained_tokens_count == 0
    assert first_row.burst_count > 0
    assert first_row.burst_retention_rate is not None
    assert 0.0 < first_row.burst_retention_rate <= 1.0
    assert first_row.has_cross_venue_ordered_adjacency is True
    assert first_row.has_trade_to_bbo_ordered_adjacency is True

    assert train_diag.truncation_rate > 0.0
    assert train_diag.weighted_target_symbol_retained_rate is not None
    assert train_diag.weighted_burst_retention_rate is not None
    assert train_diag.cross_venue_ordered_adjacency_rate > 0.0
    assert train_diag.trade_to_bbo_ordered_adjacency_rate > 0.0
    assert set(train_diag.dropped_by_symbol) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


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
