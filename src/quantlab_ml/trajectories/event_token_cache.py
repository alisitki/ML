from __future__ import annotations

import json
import math
import time
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from quantlab_ml.common import ensure_parent_dir, hash_payload
from quantlab_ml.contracts import (
    BBO_PAYLOAD_SCHEMA_ID,
    EVENT_TOKEN_CACHE_DIAGNOSTICS_FILENAME,
    EVENT_TOKEN_CACHE_DIAGNOSTICS_FORMAT_VERSION,
    EVENT_TOKEN_CACHE_DIRNAME,
    EVENT_TOKEN_CACHE_FORMAT_VERSION,
    EVENT_TOKEN_CACHE_MANIFEST_FILENAME,
    EVENT_TOKEN_CACHE_RETENTION_RECEIPT_FILENAME,
    EVENT_TOKEN_SELECTION_POLICY_ID,
    EVENT_TOKENIZER_VERSION,
    EVENT_WINDOW_CONTRACT_VERSION,
    DatasetSpec,
    EventTokenCacheDiagnosticsManifest,
    EventTokenCacheManifest,
    EventTokenCachePayloadStatus,
    EventTokenCacheRetentionReceipt,
    EventTokenReplayRow,
    EventTokenRowWindowStats,
    EventTokenSelectionHyperparameters,
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
DEFAULT_EVENT_WINDOW_RECENCY_RESERVE = 0
DEFAULT_EVENT_WINDOW_BURST_RESERVE = 0
DEFAULT_EVENT_WINDOW_BURST_GAP_MS = 250
DEFAULT_EVENT_WINDOW_STALE_AFTER_SECONDS = 180
DEFAULT_EVENT_WINDOW_RECENT_HIGH_FIDELITY_SECONDS = 5
DEFAULT_EVENT_WINDOW_CAUSAL_HORIZON_MS = 1000
DEFAULT_EVENT_WINDOW_RECENT_BBO_EXTRA_SIGNIFICANT_LIMIT = 3
DEFAULT_BBO_MID_EXCURSION_THRESHOLD = 5e-5
DEFAULT_BBO_SPREAD_REGIME_JUMP_THRESHOLD = 1.5
DEFAULT_BBO_IMBALANCE_REGIME_FLIP_THRESHOLD = 0.5
DEFAULT_BBO_LIQUIDITY_VACUUM_THRESHOLD = 3.0
_EVENT_STREAM_ORDER = ("trade", "bbo")
_SIGNIFICANCE_REASON_PRECEDENCE = (
    "liquidity_vacuum",
    "spread_regime_jump",
    "mid_excursion",
    "imbalance_regime_flip",
    "burst_boundary",
)
_TIER_ORDER = ("T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7")
_TIER_CAP_GROUPS = (
    ("T0", ("T0",), 64),
    ("T1", ("T1",), 32),
    ("T2_T3", ("T2", "T3"), 48),
    ("T4", ("T4",), 64),
    ("T5_T6", ("T5", "T6"), 32),
    ("T7_GLOBAL_FLEX", ("T7",), 16),
)


def event_token_cache_directory(directory: Path) -> Path:
    return directory / EVENT_TOKEN_CACHE_DIRNAME


def event_token_cache_manifest_path(directory: Path) -> Path:
    return event_token_cache_directory(directory) / EVENT_TOKEN_CACHE_MANIFEST_FILENAME


def event_token_cache_diagnostics_path(directory: Path) -> Path:
    return event_token_cache_directory(directory) / EVENT_TOKEN_CACHE_DIAGNOSTICS_FILENAME


def event_token_cache_retention_receipt_path(directory: Path) -> Path:
    return event_token_cache_directory(directory) / EVENT_TOKEN_CACHE_RETENTION_RECEIPT_FILENAME


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


@dataclass(slots=True)
class _InformativeUnit:
    candidate: _EventCandidate
    lag_ms: int
    source_bucket: Literal["trade_recent_raw", "trade_older_summary", "bbo_recent_sig", "bbo_older_sig"]
    lane_key: tuple[str, str, str]
    burst_id: tuple[object, ...]
    salience: float = 0.0
    emission_tags: set[str] = field(default_factory=set)
    matched_reasons: set[str] = field(default_factory=set)
    canonical_significance_reason: str | None = None
    priority_tier: str | None = None
    best_anchor_key: tuple[float, int, int, str, str, str] | None = None
    best_anchor_tier: str | None = None
    best_anchor_delta_ms: int | None = None

    @property
    def event_time_ts(self) -> float:
        return self.candidate.event_time_ts

    @property
    def exchange(self) -> str:
        return self.candidate.exchange

    @property
    def symbol(self) -> str:
        return self.candidate.symbol

    @property
    def stream(self) -> str:
        return self.candidate.stream

    @property
    def source_label_id(self) -> int:
        return self.candidate.source_label_id

    @property
    def source_event_index(self) -> int:
        return self.candidate.source_event_index

    @property
    def is_bbo_significant(self) -> bool:
        return self.stream == "bbo" and self.canonical_significance_reason is not None


@dataclass(slots=True)
class _SelectorProfile:
    bbo_significance_wall_sec: float = 0.0
    t4_resolution_wall_sec: float = 0.0
    quota_fill_wall_sec: float = 0.0
    diagnostics_serialization_wall_sec: float = 0.0
    total_selector_wall_sec: float = 0.0
    t4_anchor_count: int = 0


@dataclass(slots=True)
class _WindowBase:
    informative_units: list[_InformativeUnit]
    supported_lane_count: int
    latest_reference_by_symbol: dict[str, float]
    deduped_count: int
    duplicate_count: int
    duplicate_dropped_by_stream: Counter[str]
    duplicate_dropped_by_venue: Counter[str]
    same_timestamp_tie_count: int
    source_order_inversion_count: int
    candidate_by_symbol: Counter[str]
    candidate_by_venue: Counter[str]
    candidate_by_stream: Counter[str]
    bbo_significance_wall_sec: float


@dataclass(slots=True)
class _WindowBaseComputation:
    window_base: _WindowBase
    raw_candidates: list[_EventCandidate]
    deduped_candidates: list[_EventCandidate]


def _candidate_key(item: _EventCandidate | _InformativeUnit) -> tuple[float, int, int, str, str, str]:
    return (
        item.event_time_ts,
        item.source_label_id,
        item.source_event_index,
        item.symbol,
        item.exchange,
        item.stream,
    )


def _burst_identifier(
    lane_key: tuple[str, str, str],
    burst: list[_EventCandidate],
) -> tuple[object, ...]:
    return (lane_key, _candidate_key(burst[0]), _candidate_key(burst[-1]))


def _tier_rank(tier: str | None) -> int:
    if tier is None:
        return len(_TIER_ORDER)
    return _TIER_ORDER.index(tier)


def _priority_group_name(tier: str | None) -> str:
    if tier in {"T0", "T1", "T4"}:
        return tier
    if tier in {"T2", "T3"}:
        return "T2_T3"
    if tier in {"T5", "T6"}:
        return "T5_T6"
    return "T7_GLOBAL_FLEX"


def _canonical_significance_reason(matched_reasons: set[str]) -> str | None:
    for reason in _SIGNIFICANCE_REASON_PRECEDENCE:
        if reason in matched_reasons:
            return reason
    return None


def _lag_bucket_rank(lag_ms: int, recent_high_fidelity_seconds: int) -> int:
    recent_boundary_ms = recent_high_fidelity_seconds * 1000
    if lag_ms <= recent_boundary_ms:
        return 0
    if lag_ms <= 15_000:
        return 1
    return 2


def _base_priority_tier(*, unit: _InformativeUnit, target_symbol: str) -> str:
    if unit.symbol == target_symbol:
        if unit.source_bucket == "trade_recent_raw":
            return "T0"
        if unit.source_bucket == "bbo_recent_sig":
            return "T1"
        if unit.source_bucket == "trade_older_summary":
            return "T2"
        return "T3"
    if unit.source_bucket == "trade_recent_raw":
        return "T5"
    if unit.source_bucket == "bbo_recent_sig":
        return "T6"
    return "T7"


def _anchor_match_key(
    unit: _InformativeUnit,
    anchor: _InformativeUnit,
) -> tuple[int, int, float, float, int, int, _InformativeUnit]:
    return (
        abs(int((unit.event_time_ts - anchor.event_time_ts) * 1000)),
        _tier_rank(anchor.priority_tier),
        -anchor.salience,
        -anchor.event_time_ts,
        anchor.source_label_id,
        anchor.source_event_index,
        anchor,
    )


def _canonical_payload_identity(fields: dict[str, Any]) -> tuple[tuple[str, tuple[str, object]], ...]:
    values: list[tuple[str, tuple[str, object]]] = []
    for key, value in sorted(fields.items()):
        if value is None:
            values.append((key, ("none", None)))
        elif isinstance(value, bool):
            values.append((key, ("bool", value)))
        elif isinstance(value, int):
            values.append((key, ("int", value)))
        elif isinstance(value, float):
            values.append((key, ("float", repr(value))))
        elif isinstance(value, str):
            values.append((key, ("str", value)))
    return tuple(values)


def _candidate_dedup_key(
    candidate: _EventCandidate,
) -> tuple[str, str, str, int, tuple[tuple[str, tuple[str, object]], ...]]:
    return (
        candidate.exchange,
        candidate.symbol,
        candidate.stream,
        int(candidate.event_time_ts * 1000),
        _canonical_payload_identity(candidate.event.fields),
    )


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
        recent_high_fidelity_seconds: int = DEFAULT_EVENT_WINDOW_RECENT_HIGH_FIDELITY_SECONDS,
        causal_horizon_ms: int = DEFAULT_EVENT_WINDOW_CAUSAL_HORIZON_MS,
        recent_bbo_extra_significant_limit: int = DEFAULT_EVENT_WINDOW_RECENT_BBO_EXTRA_SIGNIFICANT_LIMIT,
        bbo_mid_excursion_threshold: float = DEFAULT_BBO_MID_EXCURSION_THRESHOLD,
        bbo_spread_regime_jump_threshold: float = DEFAULT_BBO_SPREAD_REGIME_JUMP_THRESHOLD,
        bbo_imbalance_regime_flip_threshold: float = DEFAULT_BBO_IMBALANCE_REGIME_FLIP_THRESHOLD,
        bbo_liquidity_vacuum_threshold: float = DEFAULT_BBO_LIQUIDITY_VACUUM_THRESHOLD,
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
        self.selection_policy_id = EVENT_TOKEN_SELECTION_POLICY_ID
        self.selection_hyperparameters = EventTokenSelectionHyperparameters(
            recent_high_fidelity_seconds=recent_high_fidelity_seconds,
            burst_gap_ms=burst_gap_ms,
            causal_horizon_ms=causal_horizon_ms,
            recent_bbo_extra_significant_limit=recent_bbo_extra_significant_limit,
            bbo_mid_excursion_threshold=bbo_mid_excursion_threshold,
            bbo_spread_regime_jump_threshold=bbo_spread_regime_jump_threshold,
            bbo_imbalance_regime_flip_threshold=bbo_imbalance_regime_flip_threshold,
            bbo_liquidity_vacuum_threshold=bbo_liquidity_vacuum_threshold,
        )
        self.selector_params_hash = hash_payload(self.selection_hyperparameters.model_dump(mode="json"))
        self.exchange_to_id = {
            exchange: index for index, exchange in enumerate(self.dataset_spec.exchanges)
        }
        self.symbol_to_id = {
            symbol: index for index, symbol in enumerate(self.dataset_spec.symbols)
        }
        self.stream_to_id = {stream: index for index, stream in enumerate(_EVENT_STREAM_ORDER)}
        self.split_dir = event_token_cache_directory(directory) / split_name
        self.partial_selector_profile_path = (
            event_token_cache_directory(directory) / f"{split_name}_partial_selector_profile.json"
        )
        estimated_max_row_bytes = token_cap * 128
        self.rows_per_shard = max(1, shard_target_bytes // max(estimated_max_row_bytes, 1))
        self._window_base_cache: dict[int, _WindowBase] = {}
        self._window_base_cache_hit_count = 0
        self._window_base_cache_miss_count = 0
        self._window_base_precompute_wall_sec = 0.0
        self._partial_profile_write_wall_sec = 0.0
        self._last_decision_time: datetime | None = None
        self._reset_pending_buffers()
        self._shards: list[EventTokenShardManifest] = []
        self._pending_rows = 0
        self._total_rows = 0
        self._next_shard_index = 0
        self._candidate_token_total = 0
        self._informative_candidate_total = 0
        self._selected_token_total = 0
        self._duplicate_event_count = 0
        self._same_timestamp_tie_count = 0
        self._source_order_inversion_count = 0
        self._dropped_by_stream: Counter[str] = Counter()
        self._dropped_by_venue: Counter[str] = Counter()
        self._dropped_by_symbol: Counter[str] = Counter()
        self._candidate_by_stream: Counter[str] = Counter()
        self._candidate_by_venue: Counter[str] = Counter()
        self._informative_candidate_by_stream: Counter[str] = Counter()
        self._informative_candidate_by_venue: Counter[str] = Counter()
        self._selected_by_stream: Counter[str] = Counter()
        self._selected_by_venue: Counter[str] = Counter()
        self._retained_by_symbol: Counter[str] = Counter()
        self._duplicate_dropped_by_stream: Counter[str] = Counter()
        self._duplicate_dropped_by_venue: Counter[str] = Counter()
        self._non_empty_row_count = 0
        self._empty_window_count = 0
        self._stale_window_count = 0
        self._truncated_row_count = 0
        self._token_budget_pressure_row_count = 0
        self._target_symbol_retained_numerator = 0
        self._target_symbol_retained_denominator = 0
        self._raw_target_symbol_retained_numerator = 0
        self._raw_target_symbol_retained_denominator = 0
        self._target_trade_retained_numerator = 0
        self._target_trade_retained_denominator = 0
        self._target_bbo_sig_retained_numerator = 0
        self._target_bbo_sig_retained_denominator = 0
        self._target_selected_token_total = 0
        self._retained_burst_count = 0
        self._burst_count = 0
        self._symbol_with_zero_retained_counts: list[int] = []
        self._symbol_candidate_row_count: Counter[str] = Counter()
        self._symbol_zero_retained_row_count: Counter[str] = Counter()
        self._cross_venue_ordered_adjacency_preserved_count = 0
        self._cross_venue_ordered_adjacency_raw_count = 0
        self._trade_to_bbo_ordered_adjacency_preserved_count = 0
        self._trade_to_bbo_ordered_adjacency_raw_count = 0
        self._significant_bbo_emitted_count_by_reason: Counter[str] = Counter()
        self._significant_bbo_retained_count_by_reason: Counter[str] = Counter()
        self._informative_candidate_by_tier: Counter[str] = Counter()
        self._t4_candidate_total = 0
        self._t4_anchor_total = 0
        self._t4_resolution_wall_sec = 0.0
        self._bbo_significance_wall_sec = 0.0
        self._quota_fill_wall_sec = 0.0
        self._diagnostics_serialization_wall_sec = 0.0
        self._total_selector_wall_sec = 0.0
        self._budget_fill_by_tier: Counter[str] = Counter()
        self._drop_reason_counts_by_tier: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self._lane_cap_hit_row_count = 0
        self._bbo_cap_hit_row_count = 0
        self._symbol_cap_hit_row_count = 0
        self._final_diagnostics: EventTokenSplitDiagnostics | None = None
        self._write_partial_selector_profile(status="initialized")

    def consume_record(self, record: TrajectoryRecord) -> None:
        trajectory_start = True
        for step in record.steps:
            self._append_step(record=record, step=step, trajectory_start=trajectory_start)
            trajectory_start = False

    def finalize(self) -> EventTokenSplitManifest:
        if self._pending_rows > 0:
            self._flush()
        self._final_diagnostics = self._build_split_diagnostics()
        self._write_partial_selector_profile(status="complete")
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
        _, _, selected_units, row_stats = self._build_window(
            target_symbol=record.target_symbol,
            decision_time=step.event_time,
            row_index=self._total_rows,
        )
        for unit in selected_units:
            self._append_token(decision_time=step.event_time, candidate=unit.candidate)
        self._row_offsets.append(len(self._event_time_ms))
        self._replay_rows.append(
            EventTokenReplayRow(
                decision_time=step.event_time,
                target_symbol=record.target_symbol,
                trajectory_id=record.trajectory_id,
                trajectory_start=trajectory_start,
                token_count=len(selected_units),
                truncated=row_stats.truncated,
                empty_window=row_stats.empty_window,
                stale_window=row_stats.stale_window,
            )
        )
        self._window_stats.append(row_stats)
        self._candidate_token_total += row_stats.candidate_token_count
        self._informative_candidate_total += row_stats.informative_candidate_count
        self._selected_token_total += row_stats.selected_token_count
        self._duplicate_event_count += row_stats.duplicate_event_count
        self._same_timestamp_tie_count += row_stats.same_timestamp_tie_count
        self._source_order_inversion_count += row_stats.source_order_inversion_count
        self._dropped_by_stream.update(row_stats.dropped_by_stream)
        self._dropped_by_venue.update(row_stats.dropped_by_venue)
        self._dropped_by_symbol.update(row_stats.dropped_by_symbol)
        self._candidate_by_stream.update(row_stats.candidate_by_stream)
        self._candidate_by_venue.update(row_stats.candidate_by_venue)
        self._informative_candidate_by_stream.update(row_stats.informative_candidate_by_stream)
        self._informative_candidate_by_venue.update(row_stats.informative_candidate_by_venue)
        self._selected_by_stream.update(row_stats.retained_by_stream)
        self._selected_by_venue.update(row_stats.retained_by_venue)
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
        if row_stats.token_budget_pressure:
            self._token_budget_pressure_row_count += 1
        self._symbol_with_zero_retained_counts.append(row_stats.symbol_with_zero_retained_tokens_count)
        self._retained_burst_count += row_stats.retained_burst_count
        self._burst_count += row_stats.burst_count
        if row_stats.raw_has_target_cross_venue_ordered_adjacency:
            self._cross_venue_ordered_adjacency_raw_count += 1
            if row_stats.retained_has_target_cross_venue_ordered_adjacency:
                self._cross_venue_ordered_adjacency_preserved_count += 1
        if row_stats.raw_has_target_trade_to_bbo_ordered_adjacency:
            self._trade_to_bbo_ordered_adjacency_raw_count += 1
            if row_stats.retained_has_target_trade_to_bbo_ordered_adjacency:
                self._trade_to_bbo_ordered_adjacency_preserved_count += 1
        target_candidates = row_stats.informative_candidate_by_symbol.get(record.target_symbol, 0)
        if target_candidates > 0:
            self._target_symbol_retained_numerator += row_stats.retained_by_symbol.get(record.target_symbol, 0)
            self._target_symbol_retained_denominator += target_candidates
        raw_target_candidates = row_stats.candidate_by_symbol.get(record.target_symbol, 0)
        if raw_target_candidates > 0:
            self._raw_target_symbol_retained_numerator += row_stats.retained_by_symbol.get(record.target_symbol, 0)
            self._raw_target_symbol_retained_denominator += raw_target_candidates
        if row_stats.target_trade_candidate_count > 0:
            self._target_trade_retained_numerator += row_stats.retained_target_trade_count
            self._target_trade_retained_denominator += row_stats.target_trade_candidate_count
        if row_stats.target_bbo_sig_candidate_count > 0:
            self._target_bbo_sig_retained_numerator += row_stats.retained_target_bbo_sig_count
            self._target_bbo_sig_retained_denominator += row_stats.target_bbo_sig_candidate_count
        self._target_selected_token_total += row_stats.retained_by_symbol.get(record.target_symbol, 0)
        for symbol, count in row_stats.informative_candidate_by_symbol.items():
            if count > 0:
                self._symbol_candidate_row_count[symbol] += 1
                if row_stats.retained_by_symbol.get(symbol, 0) == 0:
                    self._symbol_zero_retained_row_count[symbol] += 1
        self._significant_bbo_emitted_count_by_reason.update(row_stats.significant_bbo_emitted_count_by_reason)
        self._significant_bbo_retained_count_by_reason.update(row_stats.significant_bbo_retained_count_by_reason)
        self._informative_candidate_by_tier.update(row_stats.informative_candidate_by_tier)
        self._t4_candidate_total += row_stats.t4_candidate_count
        self._t4_anchor_total += row_stats.t4_anchor_count
        self._t4_resolution_wall_sec += row_stats.t4_resolution_wall_sec
        self._bbo_significance_wall_sec += row_stats.bbo_significance_wall_sec
        self._quota_fill_wall_sec += row_stats.quota_fill_wall_sec
        self._diagnostics_serialization_wall_sec += row_stats.diagnostics_serialization_wall_sec
        self._total_selector_wall_sec += row_stats.total_selector_wall_sec
        self._budget_fill_by_tier.update(
            {
                key: value
                for key, value in row_stats.budget_fill_by_tier.items()
                if key in {"T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"}
            }
        )
        for tier, counts in row_stats.drop_reason_counts_by_tier.items():
            self._drop_reason_counts_by_tier[tier].update(counts)
        if row_stats.lane_cap_hit_count > 0:
            self._lane_cap_hit_row_count += 1
        if row_stats.bbo_cap_hit_count > 0:
            self._bbo_cap_hit_row_count += 1
        if row_stats.symbol_cap_hit_count > 0:
            self._symbol_cap_hit_row_count += 1
        self._pending_rows += 1
        self._total_rows += 1
        if self._pending_rows >= self.rows_per_shard:
            self._flush()
        self._write_partial_selector_profile(status="in_progress")

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
        diagnostics_start = time.perf_counter()
        _write_jsonl(window_stats_path, self._window_stats)
        self._diagnostics_serialization_wall_sec += time.perf_counter() - diagnostics_start
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

    def _write_partial_selector_profile(self, *, status: str) -> None:
        started = time.perf_counter()
        tier_counts = {tier: int(self._informative_candidate_by_tier.get(tier, 0)) for tier in _TIER_ORDER}
        payload = {
            "format_version": "event_token_partial_selector_profile_v1",
            "selection_policy_id": self.selection_policy_id,
            "selector_params_hash": self.selector_params_hash,
            "split_name": self.split_name,
            "partial_split_completion_status": status,
            "rows_processed": self._total_rows,
            "pending_rows": self._pending_rows,
            "shard_count": len(self._shards),
            "raw_candidate_count": self._candidate_token_total,
            "post_compression_informative_unit_count": self._informative_candidate_total,
            "tier_counts": tier_counts,
            "t4_candidate_count": self._t4_candidate_total,
            "t4_anchor_count": self._t4_anchor_total,
            "t4_resolution_wall_sec": self._t4_resolution_wall_sec,
            "bbo_significance_wall_sec": self._bbo_significance_wall_sec,
            "quota_fill_wall_sec": self._quota_fill_wall_sec,
            "diagnostics_serialization_wall_sec": self._diagnostics_serialization_wall_sec,
            "per_split_total_selector_wall_sec": (
                self._total_selector_wall_sec + self._diagnostics_serialization_wall_sec
            ),
            "window_base_cache_hit_count": self._window_base_cache_hit_count,
            "window_base_cache_miss_count": self._window_base_cache_miss_count,
            "window_base_precompute_wall_sec": self._window_base_precompute_wall_sec,
            "partial_profile_write_wall_sec": self._partial_profile_write_wall_sec,
            "last_decision_time": self._last_decision_time.isoformat() if self._last_decision_time else None,
        }
        ensure_parent_dir(self.partial_selector_profile_path)
        tmp_path = self.partial_selector_profile_path.with_suffix(
            f"{self.partial_selector_profile_path.suffix}.tmp"
        )
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(self.partial_selector_profile_path)
        self._partial_profile_write_wall_sec += time.perf_counter() - started

    def _build_split_diagnostics(self) -> EventTokenSplitDiagnostics:
        weighted_target_symbol_retained_rate = (
            self._target_symbol_retained_numerator / self._target_symbol_retained_denominator
            if self._target_symbol_retained_denominator > 0
            else None
        )
        weighted_raw_target_symbol_retained_rate = (
            self._raw_target_symbol_retained_numerator / self._raw_target_symbol_retained_denominator
            if self._raw_target_symbol_retained_denominator > 0
            else None
        )
        weighted_target_trade_retained_rate = (
            self._target_trade_retained_numerator / self._target_trade_retained_denominator
            if self._target_trade_retained_denominator > 0
            else None
        )
        weighted_target_bbo_sig_retained_rate = (
            self._target_bbo_sig_retained_numerator / self._target_bbo_sig_retained_denominator
            if self._target_bbo_sig_retained_denominator > 0
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
        venue_candidate_total = sum(self._candidate_by_venue.values())
        venue_selected_total = sum(self._selected_by_venue.values())
        significant_bbo_emitted_total = sum(self._significant_bbo_emitted_count_by_reason.values())
        significant_bbo_retained_total = sum(self._significant_bbo_retained_count_by_reason.values())
        per_symbol_starvation_rate = {
            symbol: (
                self._symbol_zero_retained_row_count[symbol] / row_count
                if row_count > 0
                else 0.0
            )
            for symbol, row_count in sorted(self._symbol_candidate_row_count.items())
        }
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
            informative_candidate_total=self._informative_candidate_total,
            selected_token_total=self._selected_token_total,
            dropped_token_total=max(self._informative_candidate_total - self._selected_token_total, 0),
            dropped_by_stream=dict(self._dropped_by_stream),
            dropped_by_venue=dict(self._dropped_by_venue),
            dropped_by_symbol=dict(self._dropped_by_symbol),
            retained_by_symbol=dict(self._retained_by_symbol),
            candidate_by_venue=dict(self._candidate_by_venue),
            candidate_by_stream=dict(self._candidate_by_stream),
            selected_by_venue=dict(self._selected_by_venue),
            selected_by_stream=dict(self._selected_by_stream),
            duplicate_event_count=self._duplicate_event_count,
            duplicate_dropped_by_stream=dict(self._duplicate_dropped_by_stream),
            duplicate_dropped_by_venue=dict(self._duplicate_dropped_by_venue),
            same_timestamp_tie_count=self._same_timestamp_tie_count,
            source_order_inversion_count=self._source_order_inversion_count,
            weighted_target_symbol_retained_rate=weighted_target_symbol_retained_rate,
            weighted_raw_target_symbol_retained_rate=weighted_raw_target_symbol_retained_rate,
            weighted_target_trade_retained_rate=weighted_target_trade_retained_rate,
            weighted_target_bbo_sig_retained_rate=weighted_target_bbo_sig_retained_rate,
            weighted_burst_retention_rate=weighted_burst_retention_rate,
            weighted_target_selected_share=(
                self._target_selected_token_total / self._selected_token_total
                if self._selected_token_total > 0
                else None
            ),
            symbol_with_zero_retained_tokens_count_p95=symbol_zero_p95,
            per_symbol_starvation_rate=per_symbol_starvation_rate,
            venue_candidate_share_by_venue={
                venue: (count / venue_candidate_total) if venue_candidate_total > 0 else 0.0
                for venue, count in sorted(self._candidate_by_venue.items())
            },
            venue_selected_share_by_venue={
                venue: (count / venue_selected_total) if venue_selected_total > 0 else 0.0
                for venue, count in sorted(self._selected_by_venue.items())
            },
            venue_overrepresentation_ratio={
                venue: (
                    ((self._selected_by_venue.get(venue, 0) / venue_selected_total) / (count / venue_candidate_total))
                    if venue_selected_total > 0 and venue_candidate_total > 0 and count > 0
                    else 0.0
                )
                for venue, count in sorted(self._candidate_by_venue.items())
            },
            significant_bbo_emitted_count_by_reason=dict(self._significant_bbo_emitted_count_by_reason),
            significant_bbo_retained_count_by_reason=dict(self._significant_bbo_retained_count_by_reason),
            significant_bbo_preservation_rate=(
                significant_bbo_retained_total / significant_bbo_emitted_total
                if significant_bbo_emitted_total > 0
                else None
            ),
            informative_candidate_by_tier=dict(self._informative_candidate_by_tier),
            t4_candidate_total=self._t4_candidate_total,
            t4_anchor_total=self._t4_anchor_total,
            t4_resolution_wall_sec=self._t4_resolution_wall_sec,
            bbo_significance_wall_sec=self._bbo_significance_wall_sec,
            quota_fill_wall_sec=self._quota_fill_wall_sec,
            diagnostics_serialization_wall_sec=self._diagnostics_serialization_wall_sec,
            total_selector_wall_sec=(
                self._total_selector_wall_sec + self._diagnostics_serialization_wall_sec
            ),
            token_budget_pressure_row_count=self._token_budget_pressure_row_count,
            token_budget_pressure_rate=(
                self._token_budget_pressure_row_count / self._total_rows
                if self._total_rows
                else 0.0
            ),
            budget_fill_by_tier=dict(self._budget_fill_by_tier),
            drop_reason_counts_by_tier={
                tier: dict(counter)
                for tier, counter in sorted(self._drop_reason_counts_by_tier.items())
            },
            compression_ratio_by_family={
                "raw_bbo_to_significant_bbo": (
                    self._informative_candidate_by_stream.get("bbo", 0) / self._candidate_by_stream.get("bbo", 1)
                    if self._candidate_by_stream.get("bbo", 0) > 0
                    else 0.0
                ),
                "raw_trade_to_retained_trade_units": (
                    self._informative_candidate_by_stream.get("trade", 0) / self._candidate_by_stream.get("trade", 1)
                    if self._candidate_by_stream.get("trade", 0) > 0
                    else 0.0
                ),
                "significant_bbo_to_retained_bbo": (
                    self._selected_by_stream.get("bbo", 0) / self._informative_candidate_by_stream.get("bbo", 1)
                    if self._informative_candidate_by_stream.get("bbo", 0) > 0
                    else 0.0
                ),
                "total_raw_to_retained": (
                    self._selected_token_total / self._candidate_token_total
                    if self._candidate_token_total > 0
                    else 0.0
                ),
            },
            lane_cap_hit_rate=(self._lane_cap_hit_row_count / self._total_rows) if self._total_rows else 0.0,
            bbo_cap_hit_rate=(self._bbo_cap_hit_row_count / self._total_rows) if self._total_rows else 0.0,
            symbol_cap_hit_rate=(self._symbol_cap_hit_row_count / self._total_rows) if self._total_rows else 0.0,
            cross_venue_ordered_adjacency_rate=(
                self._cross_venue_ordered_adjacency_preserved_count / self._cross_venue_ordered_adjacency_raw_count
                if self._cross_venue_ordered_adjacency_raw_count
                else 0.0
            ),
            trade_to_bbo_ordered_adjacency_rate=(
                self._trade_to_bbo_ordered_adjacency_preserved_count / self._trade_to_bbo_ordered_adjacency_raw_count
                if self._trade_to_bbo_ordered_adjacency_raw_count
                else 0.0
            ),
        )

    def _build_window(
        self,
        *,
        target_symbol: str,
        decision_time: datetime,
        row_index: int,
    ) -> tuple[list[_EventCandidate], list[_InformativeUnit], list[_InformativeUnit], EventTokenRowWindowStats]:
        selector_start = time.perf_counter()
        decision_time_ts = decision_time.timestamp()
        window_start_ts = decision_time_ts - float(self.lookback_seconds)
        window_base, cache_hit = self._window_base(decision_time=decision_time)
        self._last_decision_time = decision_time

        profile = _SelectorProfile(
            bbo_significance_wall_sec=0.0 if cache_hit else window_base.bbo_significance_wall_sec
        )
        informative_units = list(window_base.informative_units)
        self._reset_target_priority_annotations(informative_units)
        self._assign_priority_tiers(
            informative_units=informative_units,
            target_symbol=target_symbol,
            profile=profile,
        )
        selected_units, selection_meta = self._select_informative_units(
            informative_units=informative_units,
            target_symbol=target_symbol,
            profile=profile,
        )
        informative_candidate_by_tier = Counter(
            unit.priority_tier or "UNKNOWN" for unit in informative_units
        )
        informative_candidate_by_symbol = Counter(unit.symbol for unit in informative_units)
        informative_candidate_by_venue = Counter(unit.exchange for unit in informative_units)
        informative_candidate_by_stream = Counter(unit.stream for unit in informative_units)
        retained_by_symbol = Counter(unit.symbol for unit in selected_units)
        retained_by_venue = Counter(unit.exchange for unit in selected_units)
        retained_by_stream = Counter(unit.stream for unit in selected_units)
        dropped_by_symbol = Counter(
            {
                symbol: informative_candidate_by_symbol[symbol] - retained_by_symbol.get(symbol, 0)
                for symbol in informative_candidate_by_symbol
            }
        )
        dropped_by_stream = Counter(informative_candidate_by_stream)
        dropped_by_stream.subtract(retained_by_stream)
        dropped_by_venue = Counter(informative_candidate_by_venue)
        dropped_by_venue.subtract(retained_by_venue)
        informative_burst_ids = {unit.burst_id for unit in informative_units}
        retained_burst_ids = {unit.burst_id for unit in selected_units}
        retained_burst_count = len(retained_burst_ids)
        target_candidate_count = informative_candidate_by_symbol.get(target_symbol, 0)
        raw_target_candidate_count = window_base.candidate_by_symbol.get(target_symbol, 0)
        retained_target_count = retained_by_symbol.get(target_symbol, 0)
        target_trade_candidate_count = sum(
            1 for unit in informative_units if unit.symbol == target_symbol and unit.stream == "trade"
        )
        retained_target_trade_count = sum(
            1 for unit in selected_units if unit.symbol == target_symbol and unit.stream == "trade"
        )
        target_bbo_sig_candidate_count = sum(
            1 for unit in informative_units if unit.symbol == target_symbol and unit.is_bbo_significant
        )
        retained_target_bbo_sig_count = sum(
            1 for unit in selected_units if unit.symbol == target_symbol and unit.is_bbo_significant
        )
        target_symbol_retained_rate = (
            retained_target_count / target_candidate_count
            if target_candidate_count > 0
            else None
        )
        raw_target_symbol_retained_rate = (
            retained_target_count / raw_target_candidate_count
            if raw_target_candidate_count > 0
            else None
        )
        empty_window = window_base.deduped_count == 0
        latest_target_reference_ts = window_base.latest_reference_by_symbol.get(target_symbol)
        stale_window = bool(
            latest_target_reference_ts is not None
            and (decision_time_ts - latest_target_reference_ts) > float(self.stale_after_seconds)
        )
        raw_target_candidates = self._target_deduped_candidates(
            target_symbol=target_symbol,
            decision_time_ts=decision_time_ts,
            window_start_ts=window_start_ts,
        )
        raw_has_cross_venue_ordered_adjacency = _has_ordered_pair(
            raw_target_candidates,
            lambda left, right: (
                left.symbol == right.symbol
                and left.stream == right.stream
                and left.exchange != right.exchange
            ),
        )
        retained_has_cross_venue_ordered_adjacency = (
            raw_has_cross_venue_ordered_adjacency
            and _has_ordered_pair(
                [unit for unit in selected_units if unit.symbol == target_symbol],
                lambda left, right: (
                    left.symbol == right.symbol
                    and left.stream == right.stream
                    and left.exchange != right.exchange
                ),
            )
        )
        raw_trade_to_bbo_source = sorted(
            [candidate for candidate in raw_target_candidates if candidate.stream == "trade"]
            + [unit for unit in informative_units if unit.symbol == target_symbol and unit.is_bbo_significant],
            key=_candidate_key,
        )
        raw_has_trade_to_bbo_ordered_adjacency = _has_ordered_pair(
            raw_trade_to_bbo_source,
            lambda left, right: (
                left.symbol == right.symbol
                and left.exchange == right.exchange
                and left.stream == "trade"
                and right.stream == "bbo"
            ),
        )
        retained_has_trade_to_bbo_ordered_adjacency = (
            raw_has_trade_to_bbo_ordered_adjacency
            and _has_ordered_pair(
                [unit for unit in selected_units if unit.symbol == target_symbol],
                lambda left, right: (
                    left.symbol == right.symbol
                    and left.exchange == right.exchange
                    and left.stream == "trade"
                    and right.stream == "bbo"
                ),
            )
        )
        significant_bbo_emitted_count_by_reason = Counter(
            unit.canonical_significance_reason
            for unit in informative_units
            if unit.is_bbo_significant and unit.canonical_significance_reason is not None
        )
        significant_bbo_retained_count_by_reason = Counter(
            unit.canonical_significance_reason
            for unit in selected_units
            if unit.is_bbo_significant and unit.canonical_significance_reason is not None
        )
        lost_after_compression_count = max(window_base.deduped_count - len(informative_units), 0)
        drop_reason_counts_by_tier = {
            tier: dict(counter)
            for tier, counter in selection_meta["drop_reason_counts_by_tier"].items()
        }
        if lost_after_compression_count > 0:
            drop_reason_counts_by_tier.setdefault("COMPRESSION", {})
            drop_reason_counts_by_tier["COMPRESSION"]["lost_after_compression"] = lost_after_compression_count
        profile.total_selector_wall_sec = time.perf_counter() - selector_start
        row_stats = EventTokenRowWindowStats(
            row_index=row_index,
            decision_time=decision_time,
            target_symbol=target_symbol,
            candidate_token_count=window_base.deduped_count,
            informative_candidate_count=len(informative_units),
            selected_token_count=len(selected_units),
            dropped_token_count=max(len(informative_units) - len(selected_units), 0),
            truncated=len(informative_units) > self.token_cap,
            token_budget_pressure=len(informative_units) > self.token_cap,
            candidate_by_symbol=dict(window_base.candidate_by_symbol),
            candidate_by_venue=dict(window_base.candidate_by_venue),
            candidate_by_stream=dict(window_base.candidate_by_stream),
            informative_candidate_by_symbol=dict(informative_candidate_by_symbol),
            informative_candidate_by_venue=dict(informative_candidate_by_venue),
            informative_candidate_by_stream=dict(informative_candidate_by_stream),
            retained_by_symbol=dict(retained_by_symbol),
            retained_by_venue=dict(retained_by_venue),
            retained_by_stream=dict(retained_by_stream),
            dropped_by_symbol={key: value for key, value in dropped_by_symbol.items() if value > 0},
            dropped_by_stream={key: value for key, value in dropped_by_stream.items() if value > 0},
            dropped_by_venue={key: value for key, value in dropped_by_venue.items() if value > 0},
            target_symbol_retained_rate=target_symbol_retained_rate,
            raw_target_symbol_retained_rate=raw_target_symbol_retained_rate,
            target_trade_retained_rate=(
                retained_target_trade_count / target_trade_candidate_count
                if target_trade_candidate_count > 0
                else None
            ),
            target_bbo_sig_retained_rate=(
                retained_target_bbo_sig_count / target_bbo_sig_candidate_count
                if target_bbo_sig_candidate_count > 0
                else None
            ),
            target_selected_share=(
                retained_target_count / len(selected_units)
                if selected_units
                else None
            ),
            target_trade_candidate_count=target_trade_candidate_count,
            retained_target_trade_count=retained_target_trade_count,
            target_bbo_sig_candidate_count=target_bbo_sig_candidate_count,
            retained_target_bbo_sig_count=retained_target_bbo_sig_count,
            target_symbol_candidate_empty=target_candidate_count == 0,
            symbol_with_zero_retained_tokens_count=sum(
                1
                for symbol, count in informative_candidate_by_symbol.items()
                if count > 0 and retained_by_symbol.get(symbol, 0) == 0
            ),
            burst_count=len(informative_burst_ids),
            retained_burst_count=retained_burst_count,
            burst_retention_rate=(
                retained_burst_count / len(informative_burst_ids)
                if informative_burst_ids
                else None
            ),
            significant_bbo_emitted_count_by_reason=dict(significant_bbo_emitted_count_by_reason),
            significant_bbo_retained_count_by_reason=dict(significant_bbo_retained_count_by_reason),
            significant_bbo_preservation_rate=(
                sum(significant_bbo_retained_count_by_reason.values()) / sum(significant_bbo_emitted_count_by_reason.values())
                if significant_bbo_emitted_count_by_reason
                else None
            ),
            informative_candidate_by_tier=dict(informative_candidate_by_tier),
            t4_candidate_count=informative_candidate_by_tier.get("T4", 0),
            t4_anchor_count=profile.t4_anchor_count,
            t4_resolution_wall_sec=profile.t4_resolution_wall_sec,
            bbo_significance_wall_sec=profile.bbo_significance_wall_sec,
            quota_fill_wall_sec=profile.quota_fill_wall_sec,
            diagnostics_serialization_wall_sec=profile.diagnostics_serialization_wall_sec,
            total_selector_wall_sec=profile.total_selector_wall_sec,
            budget_fill_by_tier=dict(selection_meta["budget_fill_by_tier"]),
            drop_reason_counts_by_tier=drop_reason_counts_by_tier,
            lane_cap_hit_count=1 if selection_meta["lane_cap_hit"] else 0,
            bbo_cap_hit_count=1 if selection_meta["bbo_cap_hit"] else 0,
            symbol_cap_hit_count=1 if selection_meta["symbol_cap_hit"] else 0,
            duplicate_event_count=window_base.duplicate_count,
            duplicate_dropped_by_stream=dict(window_base.duplicate_dropped_by_stream),
            duplicate_dropped_by_venue=dict(window_base.duplicate_dropped_by_venue),
            same_timestamp_tie_count=window_base.same_timestamp_tie_count,
            source_order_inversion_count=window_base.source_order_inversion_count,
            supported_lane_count=window_base.supported_lane_count,
            empty_window=empty_window,
            stale_window=stale_window,
            raw_has_target_cross_venue_ordered_adjacency=raw_has_cross_venue_ordered_adjacency,
            retained_has_target_cross_venue_ordered_adjacency=retained_has_cross_venue_ordered_adjacency,
            raw_has_target_trade_to_bbo_ordered_adjacency=raw_has_trade_to_bbo_ordered_adjacency,
            retained_has_target_trade_to_bbo_ordered_adjacency=retained_has_trade_to_bbo_ordered_adjacency,
            has_cross_venue_ordered_adjacency=retained_has_cross_venue_ordered_adjacency,
            has_trade_to_bbo_ordered_adjacency=retained_has_trade_to_bbo_ordered_adjacency,
        )
        return raw_target_candidates, informative_units, selected_units, row_stats

    def _window_base(self, *, decision_time: datetime) -> tuple[_WindowBase, bool]:
        decision_time_ms = datetime_to_epoch_millis(decision_time)
        cached = self._window_base_cache.get(decision_time_ms)
        if cached is not None:
            self._window_base_cache_hit_count += 1
            return cached, True

        self._window_base_cache_miss_count += 1
        precompute_start = time.perf_counter()
        computation = self._compute_window_base_optimized(decision_time=decision_time)
        window_base = computation.window_base
        self._window_base_cache[decision_time_ms] = window_base
        self._window_base_precompute_wall_sec += time.perf_counter() - precompute_start
        return window_base, False

    def _compute_window_base_slow_reference(
        self,
        *,
        decision_time: datetime,
    ) -> _WindowBaseComputation:
        decision_time_ts = decision_time.timestamp()
        window_start_ts = decision_time_ts - float(self.lookback_seconds)
        supported_lane_count = 0
        latest_reference_by_symbol: dict[str, float] = {}
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
                    if last_pos >= 0:
                        latest_reference_by_symbol[symbol] = times_arr[last_pos]
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

        raw_candidates.sort(key=_candidate_key)
        deduped_candidates: list[_EventCandidate] = []
        seen: set[tuple[str, str, str, int, tuple[tuple[str, tuple[str, object]], ...]]] = set()
        duplicate_count = 0
        duplicate_dropped_by_stream: Counter[str] = Counter()
        duplicate_dropped_by_venue: Counter[str] = Counter()
        for candidate in raw_candidates:
            dedup_key = _candidate_dedup_key(candidate)
            if dedup_key in seen:
                duplicate_count += 1
                duplicate_dropped_by_stream[candidate.stream] += 1
                duplicate_dropped_by_venue[candidate.exchange] += 1
                continue
            seen.add(dedup_key)
            deduped_candidates.append(candidate)
        return self._finish_window_base_computation(
            raw_candidates=raw_candidates,
            deduped_candidates=deduped_candidates,
            supported_lane_count=supported_lane_count,
            latest_reference_by_symbol=latest_reference_by_symbol,
            duplicate_count=duplicate_count,
            duplicate_dropped_by_stream=duplicate_dropped_by_stream,
            duplicate_dropped_by_venue=duplicate_dropped_by_venue,
            source_order_inversion_count=source_order_inversion_count,
            decision_time=decision_time,
        )

    def _compute_window_base_optimized(self, *, decision_time: datetime) -> _WindowBaseComputation:
        return self._compute_window_base_slow_reference(decision_time=decision_time)

    def _finish_window_base_computation(
        self,
        *,
        raw_candidates: list[_EventCandidate],
        deduped_candidates: list[_EventCandidate],
        supported_lane_count: int,
        latest_reference_by_symbol: dict[str, float],
        duplicate_count: int,
        duplicate_dropped_by_stream: Counter[str],
        duplicate_dropped_by_venue: Counter[str],
        source_order_inversion_count: int,
        decision_time: datetime,
    ) -> _WindowBaseComputation:
        same_timestamp_tie_count = 0
        for previous, current in zip(deduped_candidates, deduped_candidates[1:]):
            if int(previous.event_time_ts * 1000) == int(current.event_time_ts * 1000):
                same_timestamp_tie_count += 1

        candidate_by_symbol = Counter(candidate.symbol for candidate in deduped_candidates)
        candidate_by_venue = Counter(candidate.exchange for candidate in deduped_candidates)
        candidate_by_stream = Counter(candidate.stream for candidate in deduped_candidates)
        profile = _SelectorProfile()
        informative_units = self._emit_informative_units(
            candidates=deduped_candidates,
            decision_time=decision_time,
            profile=profile,
        )
        window_base = _WindowBase(
            informative_units=informative_units,
            supported_lane_count=supported_lane_count,
            latest_reference_by_symbol=latest_reference_by_symbol,
            deduped_count=len(deduped_candidates),
            duplicate_count=duplicate_count,
            duplicate_dropped_by_stream=duplicate_dropped_by_stream,
            duplicate_dropped_by_venue=duplicate_dropped_by_venue,
            same_timestamp_tie_count=same_timestamp_tie_count,
            source_order_inversion_count=source_order_inversion_count,
            candidate_by_symbol=candidate_by_symbol,
            candidate_by_venue=candidate_by_venue,
            candidate_by_stream=candidate_by_stream,
            bbo_significance_wall_sec=profile.bbo_significance_wall_sec,
        )
        return _WindowBaseComputation(
            window_base=window_base,
            raw_candidates=raw_candidates,
            deduped_candidates=deduped_candidates,
        )

    def _target_deduped_candidates(
        self,
        *,
        target_symbol: str,
        decision_time_ts: float,
        window_start_ts: float,
    ) -> list[_EventCandidate]:
        raw_candidates: list[_EventCandidate] = []
        for exchange in self.dataset_spec.exchanges:
            for stream in _EVENT_STREAM_ORDER:
                if not self.dataset_spec.stream_available(exchange, stream):
                    continue
                bucket = self.indexed.get((target_symbol, exchange, stream))
                if bucket is None:
                    continue
                times_arr, lane_events = bucket
                if times_arr.size <= 0:
                    continue
                start_pos = int(np.searchsorted(times_arr, window_start_ts, side="left"))
                end_pos = int(np.searchsorted(times_arr, decision_time_ts, side="right"))
                if end_pos <= start_pos:
                    continue
                for offset, event in enumerate(lane_events[start_pos:end_pos], start=start_pos):
                    raw_candidates.append(
                        _EventCandidate(
                            event_time_ts=float(getattr(event, "event_time_ts")),
                            exchange=exchange,
                            symbol=target_symbol,
                            stream=stream,
                            event=event,
                            lane_events=lane_events,
                            lane_position=offset,
                        )
                    )
        raw_candidates.sort(key=_candidate_key)
        deduped_candidates: list[_EventCandidate] = []
        seen: set[tuple[str, str, str, int, tuple[tuple[str, tuple[str, object]], ...]]] = set()
        for candidate in raw_candidates:
            dedup_key = _candidate_dedup_key(candidate)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            deduped_candidates.append(candidate)
        return deduped_candidates

    @staticmethod
    def _reset_target_priority_annotations(informative_units: list[_InformativeUnit]) -> None:
        for unit in informative_units:
            unit.priority_tier = None
            unit.best_anchor_key = None
            unit.best_anchor_tier = None
            unit.best_anchor_delta_ms = None

    def _emit_informative_units(
        self,
        *,
        candidates: list[_EventCandidate],
        decision_time: datetime,
        profile: _SelectorProfile | None = None,
    ) -> list[_InformativeUnit]:
        decision_time_ms = datetime_to_epoch_millis(decision_time)
        grouped: dict[tuple[str, str, str], list[_EventCandidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[(candidate.exchange, candidate.symbol, candidate.stream)].append(candidate)
        units_by_key: dict[tuple[float, int, int, str, str, str], _InformativeUnit] = {}
        for lane_key, lane_candidates in grouped.items():
            for burst in _burst_groups(lane_candidates, self.burst_gap_ms):
                burst_id = _burst_identifier(lane_key, burst)
                burst_end = burst[-1]
                burst_end_lag_ms = max(decision_time_ms - int(burst_end.event_time_ts * 1000), 0)
                is_recent_burst = burst_end_lag_ms <= self.selection_hyperparameters.recent_high_fidelity_seconds * 1000
                if lane_key[2] == "trade":
                    if is_recent_burst:
                        for candidate in burst:
                            self._upsert_informative_unit(
                                units_by_key=units_by_key,
                                candidate=candidate,
                                source_bucket="trade_recent_raw",
                                decision_time_ms=decision_time_ms,
                                lane_key=lane_key,
                                burst_id=burst_id,
                                salience=_trade_salience(candidate),
                                emission_tag="recent_raw",
                            )
                    else:
                        self._upsert_informative_unit(
                            units_by_key=units_by_key,
                            candidate=burst_end,
                            source_bucket="trade_older_summary",
                            decision_time_ms=decision_time_ms,
                            lane_key=lane_key,
                            burst_id=burst_id,
                            salience=_trade_salience(burst_end),
                            emission_tag="burst_end",
                        )
                        max_abs_flow_candidate = max(
                            burst,
                            key=lambda item: (_trade_salience(item), _candidate_key(item)),
                        )
                        self._upsert_informative_unit(
                            units_by_key=units_by_key,
                            candidate=max_abs_flow_candidate,
                            source_bucket="trade_older_summary",
                            decision_time_ms=decision_time_ms,
                            lane_key=lane_key,
                            burst_id=burst_id,
                            salience=_trade_salience(max_abs_flow_candidate),
                            emission_tag="max_abs_signed_flow",
                        )
                    continue

                burst_start = burst[0]
                bbo_start = time.perf_counter()
                bbo_assessments = [
                    (candidate, *self._bbo_significance_assessment(burst_start=burst_start, candidate=candidate))
                    for candidate in burst
                ]
                if profile is not None:
                    profile.bbo_significance_wall_sec += time.perf_counter() - bbo_start
                burst_end_reasons, burst_end_salience = next(
                    (set(reasons), salience)
                    for candidate, reasons, salience in bbo_assessments
                    if candidate is burst_end
                )
                burst_end_reasons.add("burst_boundary")
                self._upsert_informative_unit(
                    units_by_key=units_by_key,
                    candidate=burst_end,
                    source_bucket="bbo_recent_sig" if is_recent_burst else "bbo_older_sig",
                    decision_time_ms=decision_time_ms,
                    lane_key=lane_key,
                    burst_id=burst_id,
                    salience=burst_end_salience,
                    emission_tag="burst_end",
                    matched_reasons=burst_end_reasons,
                )
                significant_candidates: list[tuple[float, _EventCandidate, set[str]]] = []
                for candidate, matched_reasons, salience in bbo_assessments:
                    if matched_reasons:
                        significant_candidates.append((salience, candidate, matched_reasons))
                significant_candidates.sort(
                    key=lambda item: (-item[0], -item[1].event_time_ts, item[1].source_label_id, item[1].source_event_index)
                )
                extra_limit = (
                    self.selection_hyperparameters.recent_bbo_extra_significant_limit
                    if is_recent_burst
                    else 1
                )
                for salience, candidate, matched_reasons in significant_candidates[:extra_limit]:
                    self._upsert_informative_unit(
                        units_by_key=units_by_key,
                        candidate=candidate,
                        source_bucket="bbo_recent_sig" if is_recent_burst else "bbo_older_sig",
                        decision_time_ms=decision_time_ms,
                        lane_key=lane_key,
                        burst_id=burst_id,
                        salience=salience,
                        emission_tag="significant",
                        matched_reasons=matched_reasons,
                    )
        return sorted(units_by_key.values(), key=_candidate_key)

    def _assign_priority_tiers(
        self,
        *,
        informative_units: list[_InformativeUnit],
        target_symbol: str,
        profile: _SelectorProfile | None = None,
    ) -> None:
        for unit in informative_units:
            unit.priority_tier = _base_priority_tier(unit=unit, target_symbol=target_symbol)
        target_anchors = [
            unit
            for unit in informative_units
            if unit.symbol == target_symbol and unit.priority_tier in {"T0", "T1", "T2", "T3"}
        ]
        if profile is not None:
            profile.t4_anchor_count = len(target_anchors)
        anchors_by_exchange: dict[str, list[_InformativeUnit]] = defaultdict(list)
        anchors_by_symbol: dict[str, list[_InformativeUnit]] = defaultdict(list)
        for anchor in target_anchors:
            anchors_by_exchange[anchor.exchange].append(anchor)
            anchors_by_symbol[anchor.symbol].append(anchor)
        sorted_anchors_by_exchange = {
            key: sorted(anchors, key=_candidate_key) for key, anchors in anchors_by_exchange.items()
        }
        sorted_anchors_by_symbol = {
            key: sorted(anchors, key=_candidate_key) for key, anchors in anchors_by_symbol.items()
        }
        exchange_indexes = {
            key: (tuple(anchor.event_time_ts for anchor in anchors), anchors)
            for key, anchors in sorted_anchors_by_exchange.items()
        }
        symbol_indexes = {
            key: (tuple(anchor.event_time_ts for anchor in anchors), anchors)
            for key, anchors in sorted_anchors_by_symbol.items()
        }
        resolution_start = time.perf_counter()
        for unit in informative_units:
            if unit.priority_tier in {"T0", "T1", "T2", "T3"}:
                continue
            match = self._best_t4_anchor_match(
                unit=unit,
                exchange_indexes=exchange_indexes,
                symbol_indexes=symbol_indexes,
            )
            if match is not None:
                best_anchor = match[-1]
                unit.priority_tier = "T4"
                unit.best_anchor_key = _candidate_key(best_anchor)
                unit.best_anchor_tier = best_anchor.priority_tier
                unit.best_anchor_delta_ms = abs(int((unit.event_time_ts - best_anchor.event_time_ts) * 1000))
        if profile is not None:
            profile.t4_resolution_wall_sec += time.perf_counter() - resolution_start

    def _best_t4_anchor_match(
        self,
        *,
        unit: _InformativeUnit,
        exchange_indexes: dict[str, tuple[tuple[float, ...], list[_InformativeUnit]]],
        symbol_indexes: dict[str, tuple[tuple[float, ...], list[_InformativeUnit]]],
    ) -> tuple[int, int, float, float, int, int, _InformativeUnit] | None:
        horizon_seconds = self.selection_hyperparameters.causal_horizon_ms / 1000.0
        best: tuple[int, int, float, float, int, int, _InformativeUnit] | None = None
        seen_anchor_keys: set[tuple[float, int, int, str, str, str]] = set()
        candidate_indexes = (
            exchange_indexes.get(unit.exchange),
            symbol_indexes.get(unit.symbol),
        )
        for index in candidate_indexes:
            if index is None:
                continue
            match = self._best_t4_anchor_match_in_index(
                unit=unit,
                times=index[0],
                anchors=index[1],
                horizon_seconds=horizon_seconds,
                seen_anchor_keys=seen_anchor_keys,
            )
            if match is not None and (best is None or match < best):
                best = match
        return best

    def _best_t4_anchor_match_in_index(
        self,
        *,
        unit: _InformativeUnit,
        times: tuple[float, ...],
        anchors: list[_InformativeUnit],
        horizon_seconds: float,
        seen_anchor_keys: set[tuple[float, int, int, str, str, str]],
    ) -> tuple[int, int, float, float, int, int, _InformativeUnit] | None:
        center = bisect_left(times, unit.event_time_ts)
        left = center - 1
        right = center
        best: tuple[int, int, float, float, int, int, _InformativeUnit] | None = None

        while left >= 0 or right < len(anchors):
            left_delta = abs(unit.event_time_ts - times[left]) if left >= 0 else math.inf
            right_delta = abs(times[right] - unit.event_time_ts) if right < len(anchors) else math.inf
            next_delta = min(left_delta, right_delta)
            if next_delta > horizon_seconds:
                break
            next_delta_ms = int(next_delta * 1000)
            if best is not None and next_delta_ms > best[0]:
                break

            if left_delta <= right_delta:
                anchor = anchors[left]
                left -= 1
                match = self._t4_anchor_match_if_valid(
                    unit=unit,
                    anchor=anchor,
                    horizon_seconds=horizon_seconds,
                    seen_anchor_keys=seen_anchor_keys,
                )
                if match is not None and (best is None or match < best):
                    best = match
                continue

            anchor = anchors[right]
            right += 1
            match = self._t4_anchor_match_if_valid(
                unit=unit,
                anchor=anchor,
                horizon_seconds=horizon_seconds,
                seen_anchor_keys=seen_anchor_keys,
            )
            if match is not None and (best is None or match < best):
                best = match

        return best

    def _t4_anchor_match_if_valid(
        self,
        *,
        unit: _InformativeUnit,
        anchor: _InformativeUnit,
        horizon_seconds: float,
        seen_anchor_keys: set[tuple[float, int, int, str, str, str]],
    ) -> tuple[int, int, float, float, int, int, _InformativeUnit] | None:
        anchor_key = _candidate_key(anchor)
        if anchor_key in seen_anchor_keys:
            return None
        seen_anchor_keys.add(anchor_key)
        if abs(unit.event_time_ts - anchor.event_time_ts) > horizon_seconds:
            return None
        if not (
            (unit.symbol == anchor.symbol and unit.exchange != anchor.exchange)
            or (unit.exchange == anchor.exchange and unit.symbol != anchor.symbol)
        ):
            return None
        delta_ms = abs(int((unit.event_time_ts - anchor.event_time_ts) * 1000))
        if delta_ms > self.selection_hyperparameters.causal_horizon_ms:
            return None
        return _anchor_match_key(unit, anchor)

    def _t4_anchor_matches(
        self,
        *,
        unit: _InformativeUnit,
        exchange_indexes: dict[str, tuple[tuple[float, ...], list[_InformativeUnit]]],
        symbol_indexes: dict[str, tuple[tuple[float, ...], list[_InformativeUnit]]],
    ) -> list[tuple[int, int, float, float, int, int, _InformativeUnit]]:
        horizon_seconds = self.selection_hyperparameters.causal_horizon_ms / 1000.0
        lower = unit.event_time_ts - horizon_seconds
        upper = unit.event_time_ts + horizon_seconds
        matches: list[tuple[int, int, float, float, int, int, _InformativeUnit]] = []
        seen_anchor_keys: set[tuple[float, int, int, str, str, str]] = set()
        candidate_indexes = (
            exchange_indexes.get(unit.exchange),
            symbol_indexes.get(unit.symbol),
        )
        for index in candidate_indexes:
            if index is None:
                continue
            times, anchors = index
            start = bisect_left(times, lower)
            stop = bisect_right(times, upper)
            for anchor in anchors[start:stop]:
                anchor_key = _candidate_key(anchor)
                if anchor_key in seen_anchor_keys:
                    continue
                seen_anchor_keys.add(anchor_key)
                if not (
                    (unit.symbol == anchor.symbol and unit.exchange != anchor.exchange)
                    or (unit.exchange == anchor.exchange and unit.symbol != anchor.symbol)
                ):
                    continue
                delta_ms = abs(int((unit.event_time_ts - anchor.event_time_ts) * 1000))
                if delta_ms > self.selection_hyperparameters.causal_horizon_ms:
                    continue
                matches.append(_anchor_match_key(unit, anchor))
        return matches

    def _select_informative_units(
        self,
        *,
        informative_units: list[_InformativeUnit],
        target_symbol: str,
        profile: _SelectorProfile | None = None,
    ) -> tuple[list[_InformativeUnit], dict[str, Any]]:
        quota_start = time.perf_counter()
        selected_keys: set[tuple[float, int, int, str, str, str]] = set()
        selected_units: list[_InformativeUnit] = []
        state = {
            "total": 0,
            "bbo": 0,
            "lane": Counter(),
            "symbol": Counter(),
        }
        selection_meta: dict[str, Any] = {
            "budget_fill_by_tier": Counter(),
            "drop_reason_counts_by_tier": defaultdict(Counter),
            "lane_cap_hit": False,
            "bbo_cap_hit": False,
            "symbol_cap_hit": False,
        }
        carry_over = 0
        group_caps = self.selection_hyperparameters.tier_token_caps
        for group_name, tiers, _ in _TIER_CAP_GROUPS:
            capacity = group_caps[group_name] + carry_over
            group_units = [unit for unit in informative_units if unit.priority_tier in tiers]
            selected_count_before = len(selected_units)
            self._select_group_units(
                group_name=group_name,
                group_units=group_units,
                capacity=capacity,
                selected_keys=selected_keys,
                selected_units=selected_units,
                state=state,
                selection_meta=selection_meta,
                target_symbol=target_symbol,
            )
            selected_count_after = len(selected_units)
            carry_over = max(capacity - (selected_count_after - selected_count_before), 0)
            if carry_over > 0:
                selection_meta["drop_reason_counts_by_tier"][group_name]["deferred_to_lower_tier"] += carry_over

        final_selected = sorted(selected_units, key=_candidate_key)
        for unit in informative_units:
            if _candidate_key(unit) in selected_keys:
                continue
            tier_name = unit.priority_tier or _priority_group_name(unit.priority_tier)
            drop_reason = self._final_drop_reason(unit=unit, state=state)
            selection_meta["drop_reason_counts_by_tier"][tier_name][drop_reason] += 1
            if drop_reason in {"lane_cap", "bbo_cap", "symbol_cap"}:
                selection_meta[f"{drop_reason}_hit"] = True
        if profile is not None:
            profile.quota_fill_wall_sec += time.perf_counter() - quota_start
        return final_selected, selection_meta

    def _select_group_units(
        self,
        *,
        group_name: str,
        group_units: list[_InformativeUnit],
        capacity: int,
        selected_keys: set[tuple[float, int, int, str, str, str]],
        selected_units: list[_InformativeUnit],
        state: dict[str, Any],
        selection_meta: dict[str, Any],
        target_symbol: str,
    ) -> None:
        if capacity <= 0 or not group_units:
            return
        group_selected_count = 0
        sorted_group_units = sorted(
            group_units,
            key=lambda unit: (
                _tier_rank(unit.priority_tier),
                _lag_bucket_rank(unit.lag_ms, self.selection_hyperparameters.recent_high_fidelity_seconds),
                -unit.salience,
                -unit.event_time_ts,
                unit.source_label_id,
                unit.source_event_index,
                unit.symbol,
                unit.exchange,
                unit.stream,
            ),
        )
        local_exchange_counts: Counter[str] = Counter()
        local_symbol_counts: Counter[str] = Counter()
        exchange_max: int | None = None
        floor_targets: list[tuple[str, str, int]] = []
        if group_name == "T0":
            exchange_max = self.selection_hyperparameters.target_floor_by_exchange["T0_max"]
            floor_targets = [
                ("exchange", exchange, self.selection_hyperparameters.target_floor_by_exchange["T0_min"])
                for exchange in self.dataset_spec.exchanges
            ]
        elif group_name == "T1":
            exchange_max = self.selection_hyperparameters.target_floor_by_exchange["T1_max"]
            floor_targets = [
                ("exchange", exchange, self.selection_hyperparameters.target_floor_by_exchange["T1_min"])
                for exchange in self.dataset_spec.exchanges
            ]
        elif group_name == "T2_T3":
            exchange_max = self.selection_hyperparameters.target_floor_by_exchange["T2_T3_max"]
            floor_targets = [
                ("exchange", exchange, self.selection_hyperparameters.target_floor_by_exchange["T2_T3_min"])
                for exchange in self.dataset_spec.exchanges
            ]
        elif group_name == "T4":
            floor_targets = [
                ("symbol", symbol, self.selection_hyperparameters.t4_symbol_floor)
                for symbol in self.dataset_spec.symbols
                if symbol != target_symbol
            ]

        for attr_name, value, target_count in floor_targets:
            while (
                (local_exchange_counts[value] if attr_name == "exchange" else local_symbol_counts[value]) < target_count
                and len(selected_units) < self.token_cap
                and group_selected_count < capacity
            ):
                picked_any = False
                for unit in sorted_group_units:
                    if len(selected_units) >= self.token_cap:
                        break
                    if group_selected_count >= capacity:
                        break
                    if _candidate_key(unit) in selected_keys:
                        continue
                    if getattr(unit, attr_name) != value:
                        continue
                    if attr_name == "exchange" and exchange_max is not None and local_exchange_counts[value] >= exchange_max:
                        break
                    block_reason = self._selection_block_reason(
                        unit=unit,
                        state=state,
                    )
                    if block_reason is not None:
                        if block_reason in {"lane_cap", "bbo_cap", "symbol_cap"}:
                            selection_meta[f"{block_reason}_hit"] = True
                        continue
                    self._register_selected_unit(
                        unit=unit,
                        selected_keys=selected_keys,
                        selected_units=selected_units,
                        state=state,
                        selection_meta=selection_meta,
                    )
                    group_selected_count += 1
                    if attr_name == "exchange":
                        local_exchange_counts[value] += 1
                    else:
                        local_symbol_counts[value] += 1
                    picked_any = True
                    break
                if not picked_any:
                    break

        for unit in sorted_group_units:
            if len(selected_units) >= self.token_cap:
                break
            if group_selected_count >= capacity:
                break
            if _candidate_key(unit) in selected_keys:
                continue
            if exchange_max is not None and local_exchange_counts[unit.exchange] >= exchange_max:
                continue
            block_reason = self._selection_block_reason(
                unit=unit,
                state=state,
            )
            if block_reason is not None:
                if block_reason in {"lane_cap", "bbo_cap", "symbol_cap"}:
                    selection_meta[f"{block_reason}_hit"] = True
                continue
            self._register_selected_unit(
                unit=unit,
                selected_keys=selected_keys,
                selected_units=selected_units,
                state=state,
                selection_meta=selection_meta,
            )
            group_selected_count += 1
            if exchange_max is not None:
                local_exchange_counts[unit.exchange] += 1
            if group_name == "T4":
                local_symbol_counts[unit.symbol] += 1

    def _selection_block_reason(
        self,
        *,
        unit: _InformativeUnit,
        state: dict[str, Any],
    ) -> str | None:
        if state["total"] >= self.token_cap:
            return "bucket_overflow"
        if unit.stream == "bbo" and state["bbo"] >= self.selection_hyperparameters.bbo_total_cap:
            return "bbo_cap"
        if state["lane"][unit.lane_key] >= self.selection_hyperparameters.single_lane_cap:
            return "lane_cap"
        if state["symbol"][unit.symbol] >= self.selection_hyperparameters.single_symbol_cap:
            return "symbol_cap"
        return None

    def _register_selected_unit(
        self,
        *,
        unit: _InformativeUnit,
        selected_keys: set[tuple[float, int, int, str, str, str]],
        selected_units: list[_InformativeUnit],
        state: dict[str, Any],
        selection_meta: dict[str, Any],
    ) -> None:
        selected_keys.add(_candidate_key(unit))
        selected_units.append(unit)
        state["total"] += 1
        if unit.stream == "bbo":
            state["bbo"] += 1
        state["lane"][unit.lane_key] += 1
        state["symbol"][unit.symbol] += 1
        if unit.priority_tier is not None:
            selection_meta["budget_fill_by_tier"][unit.priority_tier] += 1

    def _final_drop_reason(
        self,
        *,
        unit: _InformativeUnit,
        state: dict[str, Any],
    ) -> str:
        if unit.stream == "bbo" and state["bbo"] >= self.selection_hyperparameters.bbo_total_cap:
            return "bbo_cap"
        if state["lane"][unit.lane_key] >= self.selection_hyperparameters.single_lane_cap:
            return "lane_cap"
        if state["symbol"][unit.symbol] >= self.selection_hyperparameters.single_symbol_cap:
            return "symbol_cap"
        return "bucket_overflow"

    def _upsert_informative_unit(
        self,
        *,
        units_by_key: dict[tuple[float, int, int, str, str, str], _InformativeUnit],
        candidate: _EventCandidate,
        source_bucket: Literal["trade_recent_raw", "trade_older_summary", "bbo_recent_sig", "bbo_older_sig"],
        decision_time_ms: int,
        lane_key: tuple[str, str, str],
        burst_id: tuple[object, ...],
        salience: float,
        emission_tag: str,
        matched_reasons: set[str] | None = None,
    ) -> None:
        key = _candidate_key(candidate)
        unit = units_by_key.get(key)
        if unit is None:
            unit = _InformativeUnit(
                candidate=candidate,
                lag_ms=max(decision_time_ms - int(candidate.event_time_ts * 1000), 0),
                source_bucket=source_bucket,
                lane_key=lane_key,
                burst_id=burst_id,
                salience=salience,
            )
            units_by_key[key] = unit
        unit.salience = max(unit.salience, salience)
        unit.emission_tags.add(emission_tag)
        if matched_reasons:
            unit.matched_reasons.update(matched_reasons)
        unit.canonical_significance_reason = _canonical_significance_reason(unit.matched_reasons)

    def _bbo_significance_assessment(
        self,
        *,
        burst_start: _EventCandidate,
        candidate: _EventCandidate,
    ) -> tuple[set[str], float]:
        return self._bbo_significance_assessment_from_values(
            start_values=_bbo_state_tuple(burst_start),
            current_values=_bbo_state_tuple(candidate),
        )

    def _bbo_significance_assessment_from_values(
        self,
        *,
        start_values: tuple[float, float, float | None, float | None] | None,
        current_values: tuple[float, float, float | None, float | None] | None,
    ) -> tuple[set[str], float]:
        if start_values is None or current_values is None:
            return set(), 0.0
        start_mid, start_spread, start_imbalance, start_min_side = start_values
        current_mid, current_spread, current_imbalance, current_min_side = current_values
        matched_reasons: set[str] = set()
        mid_move_score = abs(current_mid - start_mid) / max(abs(start_mid), 1e-12)
        spread_score = current_spread / max(start_spread, 1e-12) if current_spread > 0.0 else 0.0
        imbalance_score = (
            abs(current_imbalance - start_imbalance)
            if start_imbalance is not None and current_imbalance is not None
            else 0.0
        )
        liquidity_vacuum_score = 0.0
        if start_min_side is not None and current_min_side is not None and start_min_side > 0.0:
            liquidity_vacuum_score = start_min_side / max(current_min_side, 1e-12)
        if liquidity_vacuum_score >= self.selection_hyperparameters.bbo_liquidity_vacuum_threshold:
            matched_reasons.add("liquidity_vacuum")
        if spread_score >= self.selection_hyperparameters.bbo_spread_regime_jump_threshold:
            matched_reasons.add("spread_regime_jump")
        if mid_move_score >= self.selection_hyperparameters.bbo_mid_excursion_threshold:
            matched_reasons.add("mid_excursion")
        sign_flip = (
            start_imbalance is not None
            and current_imbalance is not None
            and (
                (start_imbalance > 0.0 and current_imbalance < 0.0)
                or (start_imbalance < 0.0 and current_imbalance > 0.0)
            )
        )
        if sign_flip and imbalance_score >= self.selection_hyperparameters.bbo_imbalance_regime_flip_threshold:
            matched_reasons.add("imbalance_regime_flip")
        return matched_reasons, max(mid_move_score, spread_score, imbalance_score, liquidity_vacuum_score)

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


def build_event_token_cache_retention_receipt(directory: Path) -> EventTokenCacheRetentionReceipt:
    payload_status = event_token_cache_payload_status(directory)
    return EventTokenCacheRetentionReceipt(
        source_manifest_path=_relative_cache_path(directory, event_token_cache_manifest_path(directory)),
        retained_shard_count=payload_status.complete_shard_count,
        missing_shard_count=payload_status.incomplete_shard_count,
        retained_payload_count=payload_status.existing_payload_count,
        missing_payload_count=payload_status.missing_payload_count,
        missing_payloads=payload_status.missing_payloads,
    )


def write_event_token_cache_retention_receipt_atomic(
    directory: Path,
    receipt: EventTokenCacheRetentionReceipt,
) -> None:
    path = event_token_cache_retention_receipt_path(directory)
    ensure_parent_dir(path)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.replace(path)


def read_event_token_cache_retention_receipt(directory: Path) -> EventTokenCacheRetentionReceipt:
    return EventTokenCacheRetentionReceipt.model_validate_json(
        event_token_cache_retention_receipt_path(directory).read_text(encoding="utf-8")
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
    shard_count = 0
    complete_shard_count = 0
    incomplete_shard_count = 0
    for split_manifest in manifest.splits.values():
        for shard in split_manifest.shards:
            shard_count += 1
            shard_paths = [
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
            referenced_paths.extend(shard_paths)
            if all((directory / relative_path).exists() for relative_path in shard_paths):
                complete_shard_count += 1
            else:
                incomplete_shard_count += 1
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
        shard_count=shard_count,
        complete_shard_count=complete_shard_count,
        incomplete_shard_count=incomplete_shard_count,
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


def _trade_salience(candidate: _EventCandidate) -> float:
    fields = candidate.event.fields
    proxy = _field_float(fields, "side_or_signed_flow_proxy")
    if proxy is None:
        qty = _field_float(fields, "qty")
        side_sign = _trade_side_sign(fields, qty)
        if qty is not None and side_sign is not None:
            proxy = qty * side_sign
    event_delta = _field_float(fields, "event_delta")
    if event_delta is None and candidate.lane_position > 0:
        previous_fields = candidate.lane_events[candidate.lane_position - 1].fields
        previous_price = _field_float(previous_fields, "price")
        current_price = _field_float(fields, "price")
        if previous_price is not None and current_price is not None:
            event_delta = current_price - previous_price
    return max(abs(proxy or 0.0), abs(event_delta or 0.0))


def _bbo_state_tuple(
    candidate: _EventCandidate,
) -> tuple[float, float, float | None, float | None] | None:
    fields = candidate.event.fields
    bid_price = _field_float(fields, "bid_price", "bid")
    ask_price = _field_float(fields, "ask_price", "ask")
    spread = _field_float(fields, "spread")
    if spread is None and bid_price is not None and ask_price is not None:
        spread = ask_price - bid_price
    mid = _field_float(fields, "mid")
    if mid is None and bid_price is not None and ask_price is not None:
        mid = (bid_price + ask_price) / 2.0
    if mid is None or spread is None:
        return None
    bid_size = _field_float(fields, "bid_size", "bid_qty")
    ask_size = _field_float(fields, "ask_size", "ask_qty")
    imbalance = _field_float(fields, "imbalance_inputs", "imbalance")
    if imbalance is None and bid_size is not None and ask_size is not None:
        denominator = bid_size + ask_size
        if abs(denominator) > 1e-12:
            imbalance = (bid_size - ask_size) / denominator
    min_side = min(bid_size, ask_size) if bid_size is not None and ask_size is not None else None
    return mid, spread, imbalance, min_side


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
    candidates: list[_EventCandidate] | list[_InformativeUnit],
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
    "EVENT_TOKEN_CACHE_RETENTION_RECEIPT_FILENAME",
    "EVENT_TOKEN_SELECTION_POLICY_ID",
    "EVENT_TOKENIZER_VERSION",
    "EVENT_WINDOW_CONTRACT_VERSION",
    "EventTokenCacheSplitWriter",
    "LoadedEventTokenCacheShard",
    "build_event_token_cache_retention_receipt",
    "datetime_to_epoch_millis",
    "epoch_millis_to_datetime",
    "event_token_cache_diagnostics_path",
    "event_token_cache_directory",
    "event_token_cache_manifest_path",
    "event_token_cache_payload_status",
    "event_token_cache_retention_receipt_path",
    "has_event_token_cache",
    "has_event_token_cache_manifest",
    "load_event_token_cache_shard",
    "read_event_token_cache_diagnostics",
    "read_event_token_cache_manifest",
    "read_event_token_cache_retention_receipt",
    "write_event_token_cache_diagnostics_atomic",
    "write_event_token_cache_manifest_atomic",
    "write_event_token_cache_retention_receipt_atomic",
]
