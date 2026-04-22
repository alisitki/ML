from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from quantlab_ml.common import ensure_parent_dir, hash_payload
from quantlab_ml.contracts import (
    BBO_PAYLOAD_SCHEMA_ID,
    EVENT_TOKEN_CACHE_DIAGNOSTICS_FILENAME,
    EVENT_TOKEN_CACHE_DIAGNOSTICS_FORMAT_VERSION,
    EVENT_TOKEN_CACHE_DIRNAME,
    EVENT_TOKEN_CACHE_FORMAT_VERSION,
    EVENT_TOKEN_CACHE_MANIFEST_FILENAME,
    EVENT_TOKENIZER_VERSION,
    EVENT_WINDOW_CONTRACT_VERSION,
    DatasetSpec,
    EventTokenCacheDiagnosticsManifest,
    EventTokenCacheManifest,
    EventTokenCachePayloadStatus,
    EventTokenReplayRow,
    EventTokenRowWindowStats,
    EventTokenShardManifest,
    EventTokenSplitDiagnostics,
    EventTokenSplitManifest,
    TRADE_PAYLOAD_SCHEMA_ID,
    TrajectoryRecord,
    TrajectoryStep,
)

DEFAULT_EVENT_TOKEN_CACHE_SHARD_TARGET_BYTES = 512 * 1024 * 1024
DEFAULT_EVENT_WINDOW_LOOKBACK_SECONDS = 60
DEFAULT_EVENT_WINDOW_TOKEN_CAP = 256
DEFAULT_EVENT_WINDOW_RECENCY_RESERVE = 64
DEFAULT_EVENT_WINDOW_BURST_RESERVE = 48
DEFAULT_EVENT_WINDOW_BURST_GAP_MS = 250
DEFAULT_EVENT_WINDOW_STALE_AFTER_SECONDS = 180
_EVENT_STREAM_ORDER = ("trade", "bbo")


def event_token_cache_directory(directory: Path) -> Path:
    return directory / EVENT_TOKEN_CACHE_DIRNAME


def event_token_cache_manifest_path(directory: Path) -> Path:
    return event_token_cache_directory(directory) / EVENT_TOKEN_CACHE_MANIFEST_FILENAME


def event_token_cache_diagnostics_path(directory: Path) -> Path:
    return event_token_cache_directory(directory) / EVENT_TOKEN_CACHE_DIAGNOSTICS_FILENAME


def has_event_token_cache_manifest(directory: Path) -> bool:
    return event_token_cache_manifest_path(directory).exists()


@dataclass(slots=True)
class LoadedEventTokenCacheShard:
    row_offsets: np.ndarray
    event_time_ms: np.ndarray
    lag_ms: np.ndarray
    exchange_ids: np.ndarray
    symbol_ids: np.ndarray
    stream_ids: np.ndarray
    source_label_ids: np.ndarray
    source_event_indices: np.ndarray
    payload_schema_ids: np.ndarray
    payload_row_indices: np.ndarray
    trade_payload_values: np.ndarray
    trade_payload_presence: np.ndarray
    bbo_payload_values: np.ndarray
    bbo_payload_presence: np.ndarray
    replay_rows: list[EventTokenReplayRow]
    window_stats: list[EventTokenRowWindowStats]

    @property
    def row_count(self) -> int:
        return len(self.replay_rows)

    @property
    def token_count(self) -> int:
        return int(self.event_time_ms.shape[0])


@dataclass(slots=True)
class _EventCandidate:
    event_time_ts: float
    exchange: str
    symbol: str
    stream: str
    event: Any
    lane_events: list[Any]
    lane_position: int

    @property
    def source_label_id(self) -> int:
        return int(getattr(self.event, "source_label_id", 0))

    @property
    def source_event_index(self) -> int:
        return int(getattr(self.event, "source_event_index", 0))


class EventTokenCacheSplitWriter:
    def __init__(
        self,
        *,
        directory: Path,
        split_name: str,
        dataset_spec: DatasetSpec,
        indexed: dict[tuple[str, str, str], tuple[np.ndarray, list[Any]]],
        source_labels: list[str],
        shard_target_bytes: int = DEFAULT_EVENT_TOKEN_CACHE_SHARD_TARGET_BYTES,
        lookback_seconds: int = DEFAULT_EVENT_WINDOW_LOOKBACK_SECONDS,
        token_cap: int = DEFAULT_EVENT_WINDOW_TOKEN_CAP,
        recency_reserve_count: int = DEFAULT_EVENT_WINDOW_RECENCY_RESERVE,
        burst_reserve_count: int = DEFAULT_EVENT_WINDOW_BURST_RESERVE,
        burst_gap_ms: int = DEFAULT_EVENT_WINDOW_BURST_GAP_MS,
        stale_after_seconds: int = DEFAULT_EVENT_WINDOW_STALE_AFTER_SECONDS,
    ) -> None:
        if token_cap <= 0:
            raise ValueError("event token cap must be positive")
        self.directory = directory
        self.split_name = split_name
        self.dataset_spec = dataset_spec
        self.indexed = indexed
        self.source_labels = source_labels or ["<unknown>"]
        self.shard_target_bytes = shard_target_bytes
        self.lookback_seconds = lookback_seconds
        self.token_cap = token_cap
        self.recency_reserve_count = recency_reserve_count
        self.burst_reserve_count = burst_reserve_count
        self.burst_gap_ms = burst_gap_ms
        self.stale_after_seconds = stale_after_seconds
        self.exchange_to_id = {
            exchange: index for index, exchange in enumerate(self.dataset_spec.exchanges)
        }
        self.symbol_to_id = {
            symbol: index for index, symbol in enumerate(self.dataset_spec.symbols)
        }
        self.stream_to_id = {stream: index for index, stream in enumerate(_EVENT_STREAM_ORDER)}
        self.split_dir = event_token_cache_directory(directory) / split_name
        estimated_max_row_bytes = token_cap * 128
        self.rows_per_shard = max(1, shard_target_bytes // max(estimated_max_row_bytes, 1))
        self._reset_pending_buffers()
        self._shards: list[EventTokenShardManifest] = []
        self._pending_rows = 0
        self._total_rows = 0
        self._next_shard_index = 0
        self._candidate_token_total = 0
        self._selected_token_total = 0
        self._duplicate_event_count = 0
        self._same_timestamp_tie_count = 0
        self._source_order_inversion_count = 0
        self._dropped_by_stream: Counter[str] = Counter()
        self._dropped_by_venue: Counter[str] = Counter()
        self._dropped_by_symbol: Counter[str] = Counter()
        self._retained_by_symbol: Counter[str] = Counter()
        self._duplicate_dropped_by_stream: Counter[str] = Counter()
        self._duplicate_dropped_by_venue: Counter[str] = Counter()
        self._non_empty_row_count = 0
        self._empty_window_count = 0
        self._stale_window_count = 0
        self._truncated_row_count = 0
        self._target_symbol_retained_numerator = 0
        self._target_symbol_retained_denominator = 0
        self._retained_burst_count = 0
        self._burst_count = 0
        self._symbol_with_zero_retained_counts: list[int] = []
        self._cross_venue_ordered_adjacency_count = 0
        self._trade_to_bbo_ordered_adjacency_count = 0
        self._final_diagnostics: EventTokenSplitDiagnostics | None = None

    def consume_record(self, record: TrajectoryRecord) -> None:
        trajectory_start = True
        for step in record.steps:
            self._append_step(record=record, step=step, trajectory_start=trajectory_start)
            trajectory_start = False

    def finalize(self) -> EventTokenSplitManifest:
        if self._pending_rows > 0:
            self._flush()
        self._final_diagnostics = self._build_split_diagnostics()
        return EventTokenSplitManifest(
            split_name=self.split_name,
            row_count=self._total_rows,
            token_count=self._selected_token_total,
            shard_count=len(self._shards),
            shards=self._shards,
        )

    def diagnostics(self) -> EventTokenSplitDiagnostics:
        if self._final_diagnostics is None:
            raise ValueError("event token cache diagnostics requested before finalize()")
        return self._final_diagnostics

    def _reset_pending_buffers(self) -> None:
        self._row_offsets: list[int] = [0]
        self._event_time_ms: list[int] = []
        self._lag_ms: list[int] = []
        self._exchange_ids: list[int] = []
        self._symbol_ids: list[int] = []
        self._stream_ids: list[int] = []
        self._source_label_ids: list[int] = []
        self._source_event_indices: list[int] = []
        self._payload_schema_ids: list[int] = []
        self._payload_row_indices: list[int] = []
        self._trade_payload_values: list[list[float]] = []
        self._trade_payload_presence: list[list[bool]] = []
        self._bbo_payload_values: list[list[float]] = []
        self._bbo_payload_presence: list[list[bool]] = []
        self._replay_rows: list[EventTokenReplayRow] = []
        self._window_stats: list[EventTokenRowWindowStats] = []

    def _append_step(
        self,
        *,
        record: TrajectoryRecord,
        step: TrajectoryStep,
        trajectory_start: bool,
    ) -> None:
        _, selected, row_stats = self._build_window(
            target_symbol=record.target_symbol,
            decision_time=step.event_time,
            row_index=self._total_rows,
        )
        for candidate in selected:
            self._append_token(decision_time=step.event_time, candidate=candidate)
        self._row_offsets.append(len(self._event_time_ms))
        self._replay_rows.append(
            EventTokenReplayRow(
                decision_time=step.event_time,
                target_symbol=record.target_symbol,
                trajectory_id=record.trajectory_id,
                trajectory_start=trajectory_start,
                token_count=len(selected),
                truncated=row_stats.truncated,
                empty_window=row_stats.empty_window,
                stale_window=row_stats.stale_window,
            )
        )
        self._window_stats.append(row_stats)
        self._candidate_token_total += row_stats.candidate_token_count
        self._selected_token_total += row_stats.selected_token_count
        self._duplicate_event_count += row_stats.duplicate_event_count
        self._same_timestamp_tie_count += row_stats.same_timestamp_tie_count
        self._source_order_inversion_count += row_stats.source_order_inversion_count
        self._dropped_by_stream.update(row_stats.dropped_by_stream)
        self._dropped_by_venue.update(row_stats.dropped_by_venue)
        self._dropped_by_symbol.update(row_stats.dropped_by_symbol)
        self._retained_by_symbol.update(row_stats.retained_by_symbol)
        self._duplicate_dropped_by_stream.update(row_stats.duplicate_dropped_by_stream)
        self._duplicate_dropped_by_venue.update(row_stats.duplicate_dropped_by_venue)
        if row_stats.selected_token_count > 0:
            self._non_empty_row_count += 1
        if row_stats.empty_window:
            self._empty_window_count += 1
        if row_stats.stale_window:
            self._stale_window_count += 1
        if row_stats.truncated:
            self._truncated_row_count += 1
        self._symbol_with_zero_retained_counts.append(row_stats.symbol_with_zero_retained_tokens_count)
        self._retained_burst_count += row_stats.retained_burst_count
        self._burst_count += row_stats.burst_count
        if row_stats.has_cross_venue_ordered_adjacency:
            self._cross_venue_ordered_adjacency_count += 1
        if row_stats.has_trade_to_bbo_ordered_adjacency:
            self._trade_to_bbo_ordered_adjacency_count += 1
        target_candidates = row_stats.candidate_by_symbol.get(record.target_symbol, 0)
        if target_candidates > 0:
            self._target_symbol_retained_denominator += target_candidates
            self._target_symbol_retained_numerator += row_stats.retained_by_symbol.get(record.target_symbol, 0)
        self._pending_rows += 1
        self._total_rows += 1
        if self._pending_rows >= self.rows_per_shard:
            self._flush()

    def _flush(self) -> None:
        if self._pending_rows <= 0:
            return
        shard_index = self._next_shard_index
        self._next_shard_index += 1
        ensure_parent_dir(self.split_dir / "placeholder")
        shard_prefix = f"shard_{shard_index:05d}"
        row_offsets_path = self.split_dir / f"{shard_prefix}_row_offsets.pt"
        event_time_path = self.split_dir / f"{shard_prefix}_event_time.pt"
        lag_ms_path = self.split_dir / f"{shard_prefix}_lag_ms.pt"
        exchange_id_path = self.split_dir / f"{shard_prefix}_exchange_id.pt"
        symbol_id_path = self.split_dir / f"{shard_prefix}_symbol_id.pt"
        stream_id_path = self.split_dir / f"{shard_prefix}_stream_id.pt"
        source_label_id_path = self.split_dir / f"{shard_prefix}_source_label_id.pt"
        source_event_index_path = self.split_dir / f"{shard_prefix}_source_event_index.pt"
        payload_schema_id_path = self.split_dir / f"{shard_prefix}_payload_schema_id.pt"
        payload_row_index_path = self.split_dir / f"{shard_prefix}_payload_row_index.pt"
        replay_path = self.split_dir / f"{shard_prefix}_replay.jsonl"
        window_stats_path = self.split_dir / f"{shard_prefix}_window_stats.jsonl"
        trade_payload_values_path = self.split_dir / f"{shard_prefix}_trade_payload_values.pt"
        trade_payload_presence_path = self.split_dir / f"{shard_prefix}_trade_payload_presence.pt"
        bbo_payload_values_path = self.split_dir / f"{shard_prefix}_bbo_payload_values.pt"
        bbo_payload_presence_path = self.split_dir / f"{shard_prefix}_bbo_payload_presence.pt"

        _torch_save(row_offsets_path, np.asarray(self._row_offsets, dtype=np.int64))
        _torch_save(event_time_path, np.asarray(self._event_time_ms, dtype=np.int64))
        _torch_save(lag_ms_path, np.asarray(self._lag_ms, dtype=np.int64))
        _torch_save(exchange_id_path, np.asarray(self._exchange_ids, dtype=np.int64))
        _torch_save(symbol_id_path, np.asarray(self._symbol_ids, dtype=np.int64))
        _torch_save(stream_id_path, np.asarray(self._stream_ids, dtype=np.int64))
        _torch_save(source_label_id_path, np.asarray(self._source_label_ids, dtype=np.int64))
        _torch_save(source_event_index_path, np.asarray(self._source_event_indices, dtype=np.int64))
        _torch_save(payload_schema_id_path, np.asarray(self._payload_schema_ids, dtype=np.int64))
        _torch_save(payload_row_index_path, np.asarray(self._payload_row_indices, dtype=np.int64))
        _torch_save(
            trade_payload_values_path,
            np.asarray(self._trade_payload_values, dtype=np.float32).reshape(-1, 5),
        )
        _torch_save(
            trade_payload_presence_path,
            np.asarray(self._trade_payload_presence, dtype=np.bool_).reshape(-1, 5),
        )
        _torch_save(
            bbo_payload_values_path,
            np.asarray(self._bbo_payload_values, dtype=np.float32).reshape(-1, 7),
        )
        _torch_save(
            bbo_payload_presence_path,
            np.asarray(self._bbo_payload_presence, dtype=np.bool_).reshape(-1, 7),
        )
        _write_jsonl(replay_path, self._replay_rows)
        _write_jsonl(window_stats_path, self._window_stats)
        first_event_time = (
            epoch_millis_to_datetime(self._event_time_ms[0])
            if self._event_time_ms
            else self._replay_rows[0].decision_time
        )
        last_event_time = (
            epoch_millis_to_datetime(self._event_time_ms[-1])
            if self._event_time_ms
            else self._replay_rows[-1].decision_time
        )
        self._shards.append(
            EventTokenShardManifest(
                split_name=self.split_name,
                shard_index=shard_index,
                row_count=self._pending_rows,
                token_count=len(self._event_time_ms),
                trade_payload_count=len(self._trade_payload_values),
                bbo_payload_count=len(self._bbo_payload_values),
                first_event_time=first_event_time,
                last_event_time=last_event_time,
                row_offsets_path=_relative_cache_path(self.directory, row_offsets_path),
                event_time_path=_relative_cache_path(self.directory, event_time_path),
                lag_ms_path=_relative_cache_path(self.directory, lag_ms_path),
                exchange_id_path=_relative_cache_path(self.directory, exchange_id_path),
                symbol_id_path=_relative_cache_path(self.directory, symbol_id_path),
                stream_id_path=_relative_cache_path(self.directory, stream_id_path),
                source_label_id_path=_relative_cache_path(self.directory, source_label_id_path),
                source_event_index_path=_relative_cache_path(self.directory, source_event_index_path),
                payload_schema_id_path=_relative_cache_path(self.directory, payload_schema_id_path),
                payload_row_index_path=_relative_cache_path(self.directory, payload_row_index_path),
                replay_path=_relative_cache_path(self.directory, replay_path),
                window_stats_path=_relative_cache_path(self.directory, window_stats_path),
                trade_payload_values_path=_relative_cache_path(self.directory, trade_payload_values_path),
                trade_payload_presence_path=_relative_cache_path(self.directory, trade_payload_presence_path),
                bbo_payload_values_path=_relative_cache_path(self.directory, bbo_payload_values_path),
                bbo_payload_presence_path=_relative_cache_path(self.directory, bbo_payload_presence_path),
            )
        )
        self._pending_rows = 0
        self._reset_pending_buffers()

    def _build_split_diagnostics(self) -> EventTokenSplitDiagnostics:
        weighted_target_symbol_retained_rate = (
            self._target_symbol_retained_numerator / self._target_symbol_retained_denominator
            if self._target_symbol_retained_denominator > 0
            else None
        )
        weighted_burst_retention_rate = (
            self._retained_burst_count / self._burst_count
            if self._burst_count > 0
            else None
        )
        symbol_zero_p95 = (
            float(np.percentile(np.asarray(self._symbol_with_zero_retained_counts, dtype=np.float32), 95))
            if self._symbol_with_zero_retained_counts
            else 0.0
        )
        return EventTokenSplitDiagnostics(
            split_name=self.split_name,
            row_count=self._total_rows,
            token_count=self._selected_token_total,
            non_empty_row_count=self._non_empty_row_count,
            empty_window_count=self._empty_window_count,
            stale_window_count=self._stale_window_count,
            truncated_row_count=self._truncated_row_count,
            truncation_rate=(self._truncated_row_count / self._total_rows) if self._total_rows else 0.0,
            candidate_token_total=self._candidate_token_total,
            selected_token_total=self._selected_token_total,
            dropped_token_total=max(self._candidate_token_total - self._selected_token_total, 0),
            dropped_by_stream=dict(self._dropped_by_stream),
            dropped_by_venue=dict(self._dropped_by_venue),
            dropped_by_symbol=dict(self._dropped_by_symbol),
            retained_by_symbol=dict(self._retained_by_symbol),
            duplicate_event_count=self._duplicate_event_count,
            duplicate_dropped_by_stream=dict(self._duplicate_dropped_by_stream),
            duplicate_dropped_by_venue=dict(self._duplicate_dropped_by_venue),
            same_timestamp_tie_count=self._same_timestamp_tie_count,
            source_order_inversion_count=self._source_order_inversion_count,
            weighted_target_symbol_retained_rate=weighted_target_symbol_retained_rate,
            weighted_burst_retention_rate=weighted_burst_retention_rate,
            symbol_with_zero_retained_tokens_count_p95=symbol_zero_p95,
            cross_venue_ordered_adjacency_rate=(
                self._cross_venue_ordered_adjacency_count / self._non_empty_row_count
                if self._non_empty_row_count
                else 0.0
            ),
            trade_to_bbo_ordered_adjacency_rate=(
                self._trade_to_bbo_ordered_adjacency_count / self._non_empty_row_count
                if self._non_empty_row_count
                else 0.0
            ),
        )

    def _build_window(
        self,
        *,
        target_symbol: str,
        decision_time: datetime,
        row_index: int,
    ) -> tuple[list[_EventCandidate], list[_EventCandidate], EventTokenRowWindowStats]:
        decision_time_ts = decision_time.timestamp()
        window_start_ts = decision_time_ts - float(self.lookback_seconds)
        supported_lane_count = 0
        latest_target_reference_ts: float | None = None
        raw_candidates: list[_EventCandidate] = []
        source_order_inversion_count = 0
        for exchange in self.dataset_spec.exchanges:
            for symbol in self.dataset_spec.symbols:
                for stream in _EVENT_STREAM_ORDER:
                    if not self.dataset_spec.stream_available(exchange, stream):
                        continue
                    supported_lane_count += 1
                    bucket = self.indexed.get((symbol, exchange, stream))
                    if bucket is None:
                        continue
                    times_arr, lane_events = bucket
                    if times_arr.size <= 0:
                        continue
                    last_pos = int(np.searchsorted(times_arr, decision_time_ts, side="right")) - 1
                    if symbol == target_symbol and last_pos >= 0:
                        latest_target_reference_ts = times_arr[last_pos]
                    start_pos = int(np.searchsorted(times_arr, window_start_ts, side="left"))
                    end_pos = int(np.searchsorted(times_arr, decision_time_ts, side="right"))
                    if end_pos <= start_pos:
                        continue
                    lane_slice = lane_events[start_pos:end_pos]
                    previous_source_key: tuple[int, int] | None = None
                    for offset, event in enumerate(lane_slice, start=start_pos):
                        source_key = (int(getattr(event, "source_label_id", 0)), int(getattr(event, "source_event_index", 0)))
                        if previous_source_key is not None and source_key < previous_source_key:
                            source_order_inversion_count += 1
                        previous_source_key = source_key
                        raw_candidates.append(
                            _EventCandidate(
                                event_time_ts=float(getattr(event, "event_time_ts")),
                                exchange=exchange,
                                symbol=symbol,
                                stream=stream,
                                event=event,
                                lane_events=lane_events,
                                lane_position=offset,
                            )
                        )

        raw_candidates.sort(
            key=lambda item: (
                item.event_time_ts,
                item.source_label_id,
                item.source_event_index,
                item.symbol,
                item.exchange,
                item.stream,
            )
        )
        deduped_candidates: list[_EventCandidate] = []
        seen: set[tuple[str, str, str, int, str]] = set()
        duplicate_count = 0
        duplicate_dropped_by_stream: Counter[str] = Counter()
        duplicate_dropped_by_venue: Counter[str] = Counter()
        for candidate in raw_candidates:
            dedup_key = (
                candidate.exchange,
                candidate.symbol,
                candidate.stream,
                int(candidate.event_time_ts * 1000),
                _canonical_payload_hash(candidate.event.fields),
            )
            if dedup_key in seen:
                duplicate_count += 1
                duplicate_dropped_by_stream[candidate.stream] += 1
                duplicate_dropped_by_venue[candidate.exchange] += 1
                continue
            seen.add(dedup_key)
            deduped_candidates.append(candidate)

        same_timestamp_tie_count = 0
        for previous, current in zip(deduped_candidates, deduped_candidates[1:]):
            if int(previous.event_time_ts * 1000) == int(current.event_time_ts * 1000):
                same_timestamp_tie_count += 1

        selected_candidates = (
            list(deduped_candidates)
            if len(deduped_candidates) <= self.token_cap
            else self._select_candidates(deduped_candidates)
        )

        candidate_by_symbol = Counter(candidate.symbol for candidate in deduped_candidates)
        retained_by_symbol = Counter(candidate.symbol for candidate in selected_candidates)
        dropped_by_symbol = Counter(
            {
                symbol: candidate_by_symbol[symbol] - retained_by_symbol.get(symbol, 0)
                for symbol in candidate_by_symbol
            }
        )
        dropped_by_stream = Counter(candidate.stream for candidate in deduped_candidates)
        dropped_by_stream.subtract(Counter(candidate.stream for candidate in selected_candidates))
        dropped_by_venue = Counter(candidate.exchange for candidate in deduped_candidates)
        dropped_by_venue.subtract(Counter(candidate.exchange for candidate in selected_candidates))
        burst_groups = _burst_groups(deduped_candidates, self.burst_gap_ms)
        selected_lookup = {
            (
                candidate.event_time_ts,
                candidate.source_label_id,
                candidate.source_event_index,
                candidate.symbol,
                candidate.exchange,
                candidate.stream,
            )
            for candidate in selected_candidates
        }
        retained_burst_count = 0
        for burst in burst_groups:
            if any(
                (
                    member.event_time_ts,
                    member.source_label_id,
                    member.source_event_index,
                    member.symbol,
                    member.exchange,
                    member.stream,
                )
                in selected_lookup
                for member in burst
            ):
                retained_burst_count += 1

        target_candidate_count = candidate_by_symbol.get(target_symbol, 0)
        target_symbol_retained_rate = (
            retained_by_symbol.get(target_symbol, 0) / target_candidate_count
            if target_candidate_count > 0
            else None
        )
        empty_window = len(deduped_candidates) == 0
        stale_window = bool(
            latest_target_reference_ts is not None
            and (decision_time_ts - latest_target_reference_ts) > float(self.stale_after_seconds)
        )
        has_cross_venue_ordered_adjacency = _has_ordered_pair(
            selected_candidates,
            lambda left, right: (
                left.symbol == right.symbol
                and left.stream == right.stream
                and left.exchange != right.exchange
            ),
        )
        has_trade_to_bbo_ordered_adjacency = _has_ordered_pair(
            selected_candidates,
            lambda left, right: (
                left.symbol == right.symbol
                and left.exchange == right.exchange
                and left.stream == "trade"
                and right.stream == "bbo"
            ),
        )
        row_stats = EventTokenRowWindowStats(
            row_index=row_index,
            decision_time=decision_time,
            target_symbol=target_symbol,
            candidate_token_count=len(deduped_candidates),
            selected_token_count=len(selected_candidates),
            dropped_token_count=max(len(deduped_candidates) - len(selected_candidates), 0),
            truncated=len(deduped_candidates) > self.token_cap,
            candidate_by_symbol=dict(candidate_by_symbol),
            retained_by_symbol=dict(retained_by_symbol),
            dropped_by_symbol={key: value for key, value in dropped_by_symbol.items() if value > 0},
            dropped_by_stream={key: value for key, value in dropped_by_stream.items() if value > 0},
            dropped_by_venue={key: value for key, value in dropped_by_venue.items() if value > 0},
            target_symbol_retained_rate=target_symbol_retained_rate,
            target_symbol_candidate_empty=target_candidate_count == 0,
            symbol_with_zero_retained_tokens_count=sum(
                1
                for symbol, count in candidate_by_symbol.items()
                if count > 0 and retained_by_symbol.get(symbol, 0) == 0
            ),
            burst_count=len(burst_groups),
            retained_burst_count=retained_burst_count,
            burst_retention_rate=(retained_burst_count / len(burst_groups)) if burst_groups else None,
            duplicate_event_count=duplicate_count,
            duplicate_dropped_by_stream=dict(duplicate_dropped_by_stream),
            duplicate_dropped_by_venue=dict(duplicate_dropped_by_venue),
            same_timestamp_tie_count=same_timestamp_tie_count,
            source_order_inversion_count=source_order_inversion_count,
            supported_lane_count=supported_lane_count,
            empty_window=empty_window,
            stale_window=stale_window,
            has_cross_venue_ordered_adjacency=has_cross_venue_ordered_adjacency,
            has_trade_to_bbo_ordered_adjacency=has_trade_to_bbo_ordered_adjacency,
        )
        return deduped_candidates, selected_candidates, row_stats

    def _select_candidates(self, candidates: list[_EventCandidate]) -> list[_EventCandidate]:
        selected_keys: set[tuple[float, int, int, str, str, str]] = set()

        def _candidate_key(candidate: _EventCandidate) -> tuple[float, int, int, str, str, str]:
            return (
                candidate.event_time_ts,
                candidate.source_label_id,
                candidate.source_event_index,
                candidate.symbol,
                candidate.exchange,
                candidate.stream,
            )

        newest_candidates = list(reversed(candidates))
        for candidate in newest_candidates[: self.recency_reserve_count]:
            selected_keys.add(_candidate_key(candidate))

        burst_tail_candidates: list[_EventCandidate] = []
        by_lane: dict[tuple[str, str, str], list[_EventCandidate]] = defaultdict(list)
        for candidate in candidates:
            by_lane[(candidate.exchange, candidate.symbol, candidate.stream)].append(candidate)
        for lane_candidates in by_lane.values():
            bursts = _burst_groups(lane_candidates, self.burst_gap_ms)
            for burst in bursts[-2:]:
                if burst:
                    burst_tail_candidates.append(burst[-1])
        burst_tail_candidates.sort(key=lambda item: _candidate_key(item), reverse=True)
        for candidate in burst_tail_candidates:
            if len(selected_keys) >= self.recency_reserve_count + self.burst_reserve_count:
                break
            selected_keys.add(_candidate_key(candidate))
        bucket_order = [
            ("trade", "binance"),
            ("trade", "bybit"),
            ("trade", "okx"),
            ("bbo", "binance"),
            ("bbo", "bybit"),
            ("bbo", "okx"),
        ]
        bucket_candidates: dict[tuple[str, str], list[_EventCandidate]] = {}
        bucket_positions: dict[tuple[str, str], int] = {}
        for stream, exchange in bucket_order:
            bucket = [
                candidate
                for candidate in reversed(candidates)
                if candidate.stream == stream and candidate.exchange == exchange
                and _candidate_key(candidate) not in selected_keys
            ]
            bucket_candidates[(stream, exchange)] = bucket
            bucket_positions[(stream, exchange)] = 0
        while len(selected_keys) < self.token_cap:
            made_progress = False
            for bucket_key in bucket_order:
                bucket = bucket_candidates[bucket_key]
                position = bucket_positions[bucket_key]
                while position < len(bucket) and _candidate_key(bucket[position]) in selected_keys:
                    position += 1
                bucket_positions[bucket_key] = position
                if position >= len(bucket):
                    continue
                selected_keys.add(_candidate_key(bucket[position]))
                bucket_positions[bucket_key] += 1
                made_progress = True
                if len(selected_keys) >= self.token_cap:
                    break
            if not made_progress:
                break
        return [candidate for candidate in candidates if _candidate_key(candidate) in selected_keys]

    def _append_token(self, *, decision_time: datetime, candidate: _EventCandidate) -> None:
        event_time_ms = datetime_to_epoch_millis(epoch_seconds_to_datetime(candidate.event_time_ts))
        lag_ms = max(datetime_to_epoch_millis(decision_time) - event_time_ms, 0)
        self._event_time_ms.append(event_time_ms)
        self._lag_ms.append(lag_ms)
        self._exchange_ids.append(self.exchange_to_id[candidate.exchange])
        self._symbol_ids.append(self.symbol_to_id[candidate.symbol])
        self._stream_ids.append(self.stream_to_id[candidate.stream])
        self._source_label_ids.append(candidate.source_label_id)
        self._source_event_indices.append(candidate.source_event_index)
        if candidate.stream == "trade":
            payload_values, payload_presence = _trade_payload(candidate, self.burst_gap_ms)
            payload_row_index = len(self._trade_payload_values)
            self._trade_payload_values.append(payload_values)
            self._trade_payload_presence.append(payload_presence)
            self._payload_schema_ids.append(TRADE_PAYLOAD_SCHEMA_ID)
        elif candidate.stream == "bbo":
            payload_values, payload_presence = _bbo_payload(candidate)
            payload_row_index = len(self._bbo_payload_values)
            self._bbo_payload_values.append(payload_values)
            self._bbo_payload_presence.append(payload_presence)
            self._payload_schema_ids.append(BBO_PAYLOAD_SCHEMA_ID)
        else:  # pragma: no cover - guarded by builder stream scope
            raise ValueError(f"unsupported event token stream: {candidate.stream}")
        self._payload_row_indices.append(payload_row_index)


def write_event_token_cache_manifest_atomic(directory: Path, manifest: EventTokenCacheManifest) -> None:
    path = event_token_cache_manifest_path(directory)
    ensure_parent_dir(path)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.replace(path)


def read_event_token_cache_manifest(directory: Path) -> EventTokenCacheManifest:
    return EventTokenCacheManifest.model_validate_json(
        event_token_cache_manifest_path(directory).read_text(encoding="utf-8")
    )


def event_token_cache_payload_status(directory: Path) -> EventTokenCachePayloadStatus:
    manifest_path = event_token_cache_manifest_path(directory)
    if not manifest_path.exists():
        return EventTokenCachePayloadStatus(
            manifest_present=False,
            payload_complete=False,
        )
    manifest = read_event_token_cache_manifest(directory)
    referenced_paths: list[str] = []
    for split_manifest in manifest.splits.values():
        for shard in split_manifest.shards:
            referenced_paths.extend(
                [
                    shard.row_offsets_path,
                    shard.event_time_path,
                    shard.lag_ms_path,
                    shard.exchange_id_path,
                    shard.symbol_id_path,
                    shard.stream_id_path,
                    shard.source_label_id_path,
                    shard.source_event_index_path,
                    shard.payload_schema_id_path,
                    shard.payload_row_index_path,
                    shard.replay_path,
                    shard.window_stats_path,
                    shard.trade_payload_values_path,
                    shard.trade_payload_presence_path,
                    shard.bbo_payload_values_path,
                    shard.bbo_payload_presence_path,
                ]
            )
    missing_payloads = sorted(
        {
            relative_path
            for relative_path in referenced_paths
            if not (directory / relative_path).exists()
        }
    )
    referenced_payload_count = len(referenced_paths)
    existing_payload_count = referenced_payload_count - len(missing_payloads)
    return EventTokenCachePayloadStatus(
        manifest_present=True,
        payload_complete=len(missing_payloads) == 0,
        referenced_payload_count=referenced_payload_count,
        existing_payload_count=existing_payload_count,
        missing_payload_count=len(missing_payloads),
        missing_payloads=missing_payloads,
    )


def has_event_token_cache(directory: Path) -> bool:
    return event_token_cache_payload_status(directory).payload_complete


def write_event_token_cache_diagnostics_atomic(
    directory: Path,
    diagnostics: EventTokenCacheDiagnosticsManifest,
) -> None:
    path = event_token_cache_diagnostics_path(directory)
    ensure_parent_dir(path)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(diagnostics.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.replace(path)


def read_event_token_cache_diagnostics(directory: Path) -> EventTokenCacheDiagnosticsManifest:
    return EventTokenCacheDiagnosticsManifest.model_validate_json(
        event_token_cache_diagnostics_path(directory).read_text(encoding="utf-8")
    )


def load_event_token_cache_shard(directory: Path, shard: EventTokenShardManifest) -> LoadedEventTokenCacheShard:
    row_offsets = _torch_load_numpy(directory / shard.row_offsets_path, np.int64)
    event_time_ms = _torch_load_numpy(directory / shard.event_time_path, np.int64)
    lag_ms = _torch_load_numpy(directory / shard.lag_ms_path, np.int64)
    exchange_ids = _torch_load_numpy(directory / shard.exchange_id_path, np.int64)
    symbol_ids = _torch_load_numpy(directory / shard.symbol_id_path, np.int64)
    stream_ids = _torch_load_numpy(directory / shard.stream_id_path, np.int64)
    source_label_ids = _torch_load_numpy(directory / shard.source_label_id_path, np.int64)
    source_event_indices = _torch_load_numpy(directory / shard.source_event_index_path, np.int64)
    payload_schema_ids = _torch_load_numpy(directory / shard.payload_schema_id_path, np.int64)
    payload_row_indices = _torch_load_numpy(directory / shard.payload_row_index_path, np.int64)
    trade_payload_values = _torch_load_numpy(directory / shard.trade_payload_values_path, np.float32).reshape(-1, 5)
    trade_payload_presence = _torch_load_numpy(directory / shard.trade_payload_presence_path, np.bool_).reshape(-1, 5)
    bbo_payload_values = _torch_load_numpy(directory / shard.bbo_payload_values_path, np.float32).reshape(-1, 7)
    bbo_payload_presence = _torch_load_numpy(directory / shard.bbo_payload_presence_path, np.bool_).reshape(-1, 7)
    replay_rows = list(_read_jsonl(directory / shard.replay_path, EventTokenReplayRow))
    window_stats = list(_read_jsonl(directory / shard.window_stats_path, EventTokenRowWindowStats))
    if row_offsets.shape[0] != len(replay_rows) + 1:
        raise ValueError(
            f"event token row offset mismatch for split={shard.split_name!r} shard={shard.shard_index}: "
            f"row_offsets={row_offsets.shape[0]}, replay_rows={len(replay_rows)}"
        )
    if len(window_stats) != len(replay_rows):
        raise ValueError(
            f"event token window stat mismatch for split={shard.split_name!r} shard={shard.shard_index}: "
            f"window_stats={len(window_stats)}, replay_rows={len(replay_rows)}"
        )
    if int(row_offsets[-1]) != int(event_time_ms.shape[0]):
        raise ValueError(
            f"event token count mismatch for split={shard.split_name!r} shard={shard.shard_index}: "
            f"row_offsets_last={int(row_offsets[-1])}, token_rows={int(event_time_ms.shape[0])}"
        )
    return LoadedEventTokenCacheShard(
        row_offsets=row_offsets,
        event_time_ms=event_time_ms,
        lag_ms=lag_ms,
        exchange_ids=exchange_ids,
        symbol_ids=symbol_ids,
        stream_ids=stream_ids,
        source_label_ids=source_label_ids,
        source_event_indices=source_event_indices,
        payload_schema_ids=payload_schema_ids,
        payload_row_indices=payload_row_indices,
        trade_payload_values=trade_payload_values,
        trade_payload_presence=trade_payload_presence,
        bbo_payload_values=bbo_payload_values,
        bbo_payload_presence=bbo_payload_presence,
        replay_rows=replay_rows,
        window_stats=window_stats,
    )


def datetime_to_epoch_millis(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def epoch_millis_to_datetime(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000.0, tz=UTC)


def epoch_seconds_to_datetime(value: float) -> datetime:
    return datetime.fromtimestamp(value, tz=UTC)


def _trade_payload(candidate: _EventCandidate, burst_gap_ms: int) -> tuple[list[float], list[bool]]:
    fields = candidate.event.fields
    price = _field_float(fields, "price")
    qty = _field_float(fields, "qty")
    side_sign = _trade_side_sign(fields, qty)
    signed_flow = qty * side_sign if qty is not None and side_sign is not None else 0.0
    signed_flow_present = qty is not None and side_sign is not None
    prev_price = None
    if candidate.lane_position > 0:
        prev_price = _field_float(candidate.lane_events[candidate.lane_position - 1].fields, "price")
    event_delta = (price - prev_price) if price is not None and prev_price is not None else 0.0
    event_delta_present = price is not None and prev_price is not None
    values = [
        price or 0.0,
        qty or 0.0,
        signed_flow,
        event_delta,
        float(_burst_length(candidate.lane_events, candidate.lane_position, burst_gap_ms)),
    ]
    presence = [
        price is not None,
        qty is not None,
        signed_flow_present,
        event_delta_present,
        True,
    ]
    return values, presence


def _bbo_payload(candidate: _EventCandidate) -> tuple[list[float], list[bool]]:
    fields = candidate.event.fields
    bid_price = _field_float(fields, "bid_price", "bid")
    ask_price = _field_float(fields, "ask_price", "ask")
    bid_size = _field_float(fields, "bid_size", "bid_qty")
    ask_size = _field_float(fields, "ask_size", "ask_qty")
    spread = (ask_price - bid_price) if ask_price is not None and bid_price is not None else 0.0
    spread_present = ask_price is not None and bid_price is not None
    mid = ((bid_price + ask_price) / 2.0) if ask_price is not None and bid_price is not None else 0.0
    mid_present = ask_price is not None and bid_price is not None
    imbalance = 0.0
    imbalance_present = False
    if bid_size is not None and ask_size is not None:
        denominator = bid_size + ask_size
        if abs(denominator) > 1e-12:
            imbalance = (bid_size - ask_size) / max(denominator, 1e-12)
            imbalance_present = True
    values = [
        bid_price or 0.0,
        ask_price or 0.0,
        bid_size or 0.0,
        ask_size or 0.0,
        spread,
        mid,
        imbalance,
    ]
    presence = [
        bid_price is not None,
        ask_price is not None,
        bid_size is not None,
        ask_size is not None,
        spread_present,
        mid_present,
        imbalance_present,
    ]
    return values, presence


def _trade_side_sign(fields: dict[str, Any], qty: float | None) -> float | None:
    string_side = None
    for key in ("aggressor_side", "side"):
        value = fields.get(key)
        if isinstance(value, str) and value.strip():
            string_side = value.strip().lower()
            break
    if string_side in {"buy", "bid", "b", "1"}:
        return 1.0
    if string_side in {"sell", "ask", "s", "-1"}:
        return -1.0
    buyer_maker = fields.get("is_buyer_maker")
    if isinstance(buyer_maker, bool):
        return -1.0 if buyer_maker else 1.0
    if qty is not None and qty > 0.0:
        proxy = _field_float(fields, "side_or_signed_flow_proxy")
        if proxy is not None:
            if proxy > 0.0:
                return 1.0
            if proxy < 0.0:
                return -1.0
    return None


def _field_float(fields: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in fields:
            continue
        value = fields[key]
        if value is None:
            continue
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(coerced):
            continue
        return coerced
    return None


def _burst_length(lane_events: list[Any], position: int, burst_gap_ms: int) -> int:
    length = 1
    current_index = position
    while current_index > 0:
        current_time = float(getattr(lane_events[current_index], "event_time_ts"))
        previous_time = float(getattr(lane_events[current_index - 1], "event_time_ts"))
        if ((current_time - previous_time) * 1000.0) > float(burst_gap_ms):
            break
        length += 1
        current_index -= 1
    return length


def _canonical_payload_hash(fields: dict[str, Any]) -> str:
    normalized = {
        key: value
        for key, value in sorted(fields.items())
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    return hash_payload(normalized)


def _burst_groups(candidates: list[_EventCandidate], burst_gap_ms: int) -> list[list[_EventCandidate]]:
    if not candidates:
        return []
    grouped: dict[tuple[str, str, str], list[_EventCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.exchange, candidate.symbol, candidate.stream)].append(candidate)
    bursts: list[list[_EventCandidate]] = []
    for lane_candidates in grouped.values():
        current_burst: list[_EventCandidate] = []
        previous_time_ts: float | None = None
        for candidate in lane_candidates:
            if previous_time_ts is None or ((candidate.event_time_ts - previous_time_ts) * 1000.0) > float(burst_gap_ms):
                if current_burst:
                    bursts.append(current_burst)
                current_burst = [candidate]
            else:
                current_burst.append(candidate)
            previous_time_ts = candidate.event_time_ts
        if current_burst:
            bursts.append(current_burst)
    return bursts


def _has_ordered_pair(
    candidates: list[_EventCandidate],
    predicate: Any,
) -> bool:
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            if predicate(left, right):
                return True
    return False


def _relative_cache_path(directory: Path, path: Path) -> str:
    return str(path.relative_to(directory))


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json())
            handle.write("\n")


def _read_jsonl(path: Path, model_type: Any) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            yield model_type.model_validate_json(stripped)


def _torch_save(path: Path, values: np.ndarray) -> None:
    ensure_parent_dir(path)
    torch_module = _require_torch()
    tensor = torch_module.from_numpy(np.ascontiguousarray(values))
    torch_module.save(tensor, path)


def _torch_load_numpy(path: Path, dtype: type[np.generic]) -> np.ndarray:
    torch_module = _require_torch()
    loaded = torch_module.load(path, map_location="cpu")
    if hasattr(loaded, "detach"):
        array = loaded.detach().cpu().numpy()
    else:
        array = np.asarray(loaded)
    return np.asarray(array, dtype=dtype)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("torch is required for event token cache support") from exc
    return torch


__all__ = [
    "DEFAULT_EVENT_TOKEN_CACHE_SHARD_TARGET_BYTES",
    "DEFAULT_EVENT_WINDOW_BURST_GAP_MS",
    "DEFAULT_EVENT_WINDOW_BURST_RESERVE",
    "DEFAULT_EVENT_WINDOW_LOOKBACK_SECONDS",
    "DEFAULT_EVENT_WINDOW_RECENCY_RESERVE",
    "DEFAULT_EVENT_WINDOW_STALE_AFTER_SECONDS",
    "DEFAULT_EVENT_WINDOW_TOKEN_CAP",
    "EVENT_TOKEN_CACHE_DIAGNOSTICS_FILENAME",
    "EVENT_TOKEN_CACHE_DIAGNOSTICS_FORMAT_VERSION",
    "EVENT_TOKEN_CACHE_DIRNAME",
    "EVENT_TOKEN_CACHE_FORMAT_VERSION",
    "EVENT_TOKEN_CACHE_MANIFEST_FILENAME",
    "EVENT_TOKENIZER_VERSION",
    "EVENT_WINDOW_CONTRACT_VERSION",
    "EventTokenCacheSplitWriter",
    "LoadedEventTokenCacheShard",
    "datetime_to_epoch_millis",
    "epoch_millis_to_datetime",
    "event_token_cache_diagnostics_path",
    "event_token_cache_directory",
    "event_token_cache_manifest_path",
    "event_token_cache_payload_status",
    "has_event_token_cache",
    "has_event_token_cache_manifest",
    "load_event_token_cache_shard",
    "read_event_token_cache_diagnostics",
    "read_event_token_cache_manifest",
    "write_event_token_cache_diagnostics_atomic",
    "write_event_token_cache_manifest_atomic",
]
