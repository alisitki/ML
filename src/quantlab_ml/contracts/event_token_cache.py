from __future__ import annotations

from datetime import datetime

from pydantic import Field

from quantlab_ml.contracts.common import QuantBaseModel

EVENT_TOKEN_CACHE_FORMAT_VERSION = "event_token_cache_v1"
EVENT_TOKEN_CACHE_DIRNAME = EVENT_TOKEN_CACHE_FORMAT_VERSION
EVENT_TOKEN_CACHE_MANIFEST_FILENAME = "event_token_cache_manifest.json"
EVENT_TOKEN_CACHE_SUMMARY_FILENAME = "event_token_cache_manifest.summary.json"
EVENT_TOKEN_CACHE_DIAGNOSTICS_FORMAT_VERSION = "event_token_cache_diagnostics_v1"
EVENT_TOKEN_CACHE_DIAGNOSTICS_FILENAME = "event_token_cache_diagnostics.json"
EVENT_WINDOW_CONTRACT_VERSION = "event_window_contract_v1"
EVENT_TOKENIZER_VERSION = "event_tokenizer_contract_v1"
TRADE_PAYLOAD_SCHEMA_ID = 1
BBO_PAYLOAD_SCHEMA_ID = 2


class EventTokenReplayRow(QuantBaseModel):
    decision_time: datetime
    target_symbol: str
    trajectory_id: str
    trajectory_start: bool
    token_count: int
    truncated: bool
    empty_window: bool
    stale_window: bool


class EventTokenRowWindowStats(QuantBaseModel):
    row_index: int
    decision_time: datetime
    target_symbol: str
    candidate_token_count: int
    selected_token_count: int
    dropped_token_count: int
    truncated: bool
    candidate_by_symbol: dict[str, int] = Field(default_factory=dict)
    retained_by_symbol: dict[str, int] = Field(default_factory=dict)
    dropped_by_symbol: dict[str, int] = Field(default_factory=dict)
    dropped_by_stream: dict[str, int] = Field(default_factory=dict)
    dropped_by_venue: dict[str, int] = Field(default_factory=dict)
    target_symbol_retained_rate: float | None = None
    target_symbol_candidate_empty: bool = False
    symbol_with_zero_retained_tokens_count: int = 0
    burst_count: int = 0
    retained_burst_count: int = 0
    burst_retention_rate: float | None = None
    duplicate_event_count: int = 0
    duplicate_dropped_by_stream: dict[str, int] = Field(default_factory=dict)
    duplicate_dropped_by_venue: dict[str, int] = Field(default_factory=dict)
    same_timestamp_tie_count: int = 0
    source_order_inversion_count: int = 0
    supported_lane_count: int = 0
    empty_window: bool = False
    stale_window: bool = False
    has_cross_venue_ordered_adjacency: bool = False
    has_trade_to_bbo_ordered_adjacency: bool = False


class EventTokenShardManifest(QuantBaseModel):
    split_name: str
    shard_index: int
    row_count: int
    token_count: int
    trade_payload_count: int
    bbo_payload_count: int
    first_event_time: datetime
    last_event_time: datetime
    row_offsets_path: str
    event_time_path: str
    lag_ms_path: str
    exchange_id_path: str
    symbol_id_path: str
    stream_id_path: str
    source_label_id_path: str
    source_event_index_path: str
    payload_schema_id_path: str
    payload_row_index_path: str
    replay_path: str
    window_stats_path: str
    trade_payload_values_path: str
    trade_payload_presence_path: str
    bbo_payload_values_path: str
    bbo_payload_presence_path: str


class EventTokenSplitManifest(QuantBaseModel):
    split_name: str
    row_count: int
    token_count: int
    shard_count: int
    shards: list[EventTokenShardManifest] = Field(default_factory=list)


class EventTokenCacheManifest(QuantBaseModel):
    format_version: str = EVENT_TOKEN_CACHE_FORMAT_VERSION
    event_window_contract_version: str = EVENT_WINDOW_CONTRACT_VERSION
    tokenizer_version: str = EVENT_TOKENIZER_VERSION
    trajectory_manifest_hash: str
    tensor_cache_manifest_hash: str
    dataset_hash: str
    lookback_seconds: int
    token_cap: int
    recency_reserve_count: int
    burst_reserve_count: int
    burst_gap_ms: int
    stale_after_seconds: int
    stream_order: list[str] = Field(default_factory=list)
    exchange_order: list[str] = Field(default_factory=list)
    symbol_order: list[str] = Field(default_factory=list)
    source_labels: list[str] = Field(default_factory=list)
    payload_schema_catalog: dict[str, dict[str, object]] = Field(default_factory=dict)
    splits: dict[str, EventTokenSplitManifest] = Field(default_factory=dict)


class EventTokenCachePayloadStatus(QuantBaseModel):
    manifest_present: bool
    payload_complete: bool
    referenced_payload_count: int = 0
    existing_payload_count: int = 0
    missing_payload_count: int = 0
    missing_payloads: list[str] = Field(default_factory=list)


class EventTokenSplitDiagnostics(QuantBaseModel):
    split_name: str
    row_count: int
    token_count: int
    non_empty_row_count: int
    empty_window_count: int
    stale_window_count: int
    truncated_row_count: int
    truncation_rate: float
    candidate_token_total: int
    selected_token_total: int
    dropped_token_total: int
    dropped_by_stream: dict[str, int] = Field(default_factory=dict)
    dropped_by_venue: dict[str, int] = Field(default_factory=dict)
    dropped_by_symbol: dict[str, int] = Field(default_factory=dict)
    retained_by_symbol: dict[str, int] = Field(default_factory=dict)
    duplicate_event_count: int = 0
    duplicate_dropped_by_stream: dict[str, int] = Field(default_factory=dict)
    duplicate_dropped_by_venue: dict[str, int] = Field(default_factory=dict)
    same_timestamp_tie_count: int = 0
    source_order_inversion_count: int = 0
    weighted_target_symbol_retained_rate: float | None = None
    weighted_burst_retention_rate: float | None = None
    symbol_with_zero_retained_tokens_count_p95: float = 0.0
    cross_venue_ordered_adjacency_rate: float = 0.0
    trade_to_bbo_ordered_adjacency_rate: float = 0.0


class EventTokenCacheDiagnosticsManifest(QuantBaseModel):
    format_version: str = EVENT_TOKEN_CACHE_DIAGNOSTICS_FORMAT_VERSION
    splits: dict[str, EventTokenSplitDiagnostics] = Field(default_factory=dict)


class EventTokenCacheRetentionReceipt(QuantBaseModel):
    format_version: str = EVENT_TOKEN_CACHE_FORMAT_VERSION
    source_manifest_path: str
    retained_shard_count: int
    missing_shard_count: int
    retained_payload_count: int
    missing_payload_count: int


__all__ = [
    "BBO_PAYLOAD_SCHEMA_ID",
    "EVENT_TOKEN_CACHE_DIAGNOSTICS_FILENAME",
    "EVENT_TOKEN_CACHE_DIAGNOSTICS_FORMAT_VERSION",
    "EVENT_TOKEN_CACHE_DIRNAME",
    "EVENT_TOKEN_CACHE_FORMAT_VERSION",
    "EVENT_TOKEN_CACHE_MANIFEST_FILENAME",
    "EVENT_TOKEN_CACHE_SUMMARY_FILENAME",
    "EVENT_TOKENIZER_VERSION",
    "EVENT_WINDOW_CONTRACT_VERSION",
    "TRADE_PAYLOAD_SCHEMA_ID",
    "EventTokenCacheDiagnosticsManifest",
    "EventTokenCacheManifest",
    "EventTokenCachePayloadStatus",
    "EventTokenCacheRetentionReceipt",
    "EventTokenReplayRow",
    "EventTokenRowWindowStats",
    "EventTokenShardManifest",
    "EventTokenSplitDiagnostics",
    "EventTokenSplitManifest",
]
