from __future__ import annotations

from datetime import datetime

from pydantic import Field

from quantlab_ml.contracts.common import QuantBaseModel

EVENT_TOKEN_CACHE_FORMAT_VERSION = "event_token_cache_v1"
EVENT_TOKEN_CACHE_DIRNAME = EVENT_TOKEN_CACHE_FORMAT_VERSION
EVENT_TOKEN_CACHE_MANIFEST_FILENAME = "event_token_cache_manifest.json"
EVENT_TOKEN_CACHE_SUMMARY_FILENAME = "event_token_cache_manifest.summary.json"
EVENT_TOKEN_CACHE_DIAGNOSTICS_FORMAT_VERSION = "event_token_cache_diagnostics_v2"
EVENT_TOKEN_CACHE_DIAGNOSTICS_FILENAME = "event_token_cache_diagnostics.json"
EVENT_TOKEN_CACHE_RETENTION_RECEIPT_FILENAME = "event_token_cache_retention_receipt.json"
EVENT_WINDOW_CONTRACT_VERSION = "event_window_contract_v2"
EVENT_TOKENIZER_VERSION = "event_tokenizer_contract_v2"
EVENT_TOKEN_SELECTION_POLICY_ID = "significant_bbo_priority_window_v2"
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


class EventTokenSelectionHyperparameters(QuantBaseModel):
    recent_high_fidelity_seconds: int = 5
    burst_gap_ms: int = 250
    causal_horizon_ms: int = 1000
    recent_bbo_extra_significant_limit: int = 3
    bbo_mid_excursion_threshold: float = 5e-5
    bbo_spread_regime_jump_threshold: float = 1.5
    bbo_imbalance_regime_flip_threshold: float = 0.5
    bbo_liquidity_vacuum_threshold: float = 3.0
    tier_token_caps: dict[str, int] = Field(
        default_factory=lambda: {
            "T0": 64,
            "T1": 32,
            "T2_T3": 48,
            "T4": 64,
            "T5_T6": 32,
            "T7_GLOBAL_FLEX": 16,
        }
    )
    bbo_total_cap: int = 96
    single_lane_cap: int = 40
    single_symbol_cap: int = 160
    target_floor_by_exchange: dict[str, int] = Field(
        default_factory=lambda: {
            "T0_min": 16,
            "T0_max": 32,
            "T1_min": 8,
            "T1_max": 16,
            "T2_T3_min": 8,
            "T2_T3_max": 24,
        }
    )
    t4_symbol_floor: int = 4


class EventTokenRowWindowStats(QuantBaseModel):
    row_index: int
    decision_time: datetime
    target_symbol: str
    candidate_token_count: int
    informative_candidate_count: int = 0
    selected_token_count: int
    dropped_token_count: int
    truncated: bool
    token_budget_pressure: bool = False
    candidate_by_symbol: dict[str, int] = Field(default_factory=dict)
    candidate_by_venue: dict[str, int] = Field(default_factory=dict)
    candidate_by_stream: dict[str, int] = Field(default_factory=dict)
    informative_candidate_by_symbol: dict[str, int] = Field(default_factory=dict)
    informative_candidate_by_venue: dict[str, int] = Field(default_factory=dict)
    informative_candidate_by_stream: dict[str, int] = Field(default_factory=dict)
    retained_by_symbol: dict[str, int] = Field(default_factory=dict)
    retained_by_venue: dict[str, int] = Field(default_factory=dict)
    retained_by_stream: dict[str, int] = Field(default_factory=dict)
    dropped_by_symbol: dict[str, int] = Field(default_factory=dict)
    dropped_by_stream: dict[str, int] = Field(default_factory=dict)
    dropped_by_venue: dict[str, int] = Field(default_factory=dict)
    target_symbol_retained_rate: float | None = None
    raw_target_symbol_retained_rate: float | None = None
    target_trade_retained_rate: float | None = None
    target_bbo_sig_retained_rate: float | None = None
    target_selected_share: float | None = None
    target_trade_candidate_count: int = 0
    retained_target_trade_count: int = 0
    target_bbo_sig_candidate_count: int = 0
    retained_target_bbo_sig_count: int = 0
    target_symbol_candidate_empty: bool = False
    symbol_with_zero_retained_tokens_count: int = 0
    burst_count: int = 0
    retained_burst_count: int = 0
    burst_retention_rate: float | None = None
    significant_bbo_emitted_count_by_reason: dict[str, int] = Field(default_factory=dict)
    significant_bbo_retained_count_by_reason: dict[str, int] = Field(default_factory=dict)
    significant_bbo_preservation_rate: float | None = None
    informative_candidate_by_tier: dict[str, int] = Field(default_factory=dict)
    t4_candidate_count: int = 0
    t4_anchor_count: int = 0
    t4_resolution_wall_sec: float = Field(default=0.0, exclude=True)
    bbo_significance_wall_sec: float = Field(default=0.0, exclude=True)
    quota_fill_wall_sec: float = Field(default=0.0, exclude=True)
    diagnostics_serialization_wall_sec: float = Field(default=0.0, exclude=True)
    total_selector_wall_sec: float = Field(default=0.0, exclude=True)
    budget_fill_by_tier: dict[str, int] = Field(default_factory=dict)
    drop_reason_counts_by_tier: dict[str, dict[str, int]] = Field(default_factory=dict)
    lane_cap_hit_count: int = 0
    bbo_cap_hit_count: int = 0
    symbol_cap_hit_count: int = 0
    duplicate_event_count: int = 0
    duplicate_dropped_by_stream: dict[str, int] = Field(default_factory=dict)
    duplicate_dropped_by_venue: dict[str, int] = Field(default_factory=dict)
    same_timestamp_tie_count: int = 0
    source_order_inversion_count: int = 0
    supported_lane_count: int = 0
    empty_window: bool = False
    stale_window: bool = False
    raw_has_target_cross_venue_ordered_adjacency: bool = False
    retained_has_target_cross_venue_ordered_adjacency: bool = False
    raw_has_target_trade_to_bbo_ordered_adjacency: bool = False
    retained_has_target_trade_to_bbo_ordered_adjacency: bool = False
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
    selection_policy_id: str = EVENT_TOKEN_SELECTION_POLICY_ID
    selection_hyperparameters: EventTokenSelectionHyperparameters = Field(
        default_factory=EventTokenSelectionHyperparameters
    )
    selector_params_hash: str | None = None
    selector_audit_artifact_path: str | None = None
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
    shard_count: int = 0
    complete_shard_count: int = 0
    incomplete_shard_count: int = 0
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
    informative_candidate_total: int = 0
    selected_token_total: int
    dropped_token_total: int
    dropped_by_stream: dict[str, int] = Field(default_factory=dict)
    dropped_by_venue: dict[str, int] = Field(default_factory=dict)
    dropped_by_symbol: dict[str, int] = Field(default_factory=dict)
    retained_by_symbol: dict[str, int] = Field(default_factory=dict)
    candidate_by_venue: dict[str, int] = Field(default_factory=dict)
    candidate_by_stream: dict[str, int] = Field(default_factory=dict)
    selected_by_venue: dict[str, int] = Field(default_factory=dict)
    selected_by_stream: dict[str, int] = Field(default_factory=dict)
    duplicate_event_count: int = 0
    duplicate_dropped_by_stream: dict[str, int] = Field(default_factory=dict)
    duplicate_dropped_by_venue: dict[str, int] = Field(default_factory=dict)
    same_timestamp_tie_count: int = 0
    source_order_inversion_count: int = 0
    weighted_target_symbol_retained_rate: float | None = None
    weighted_raw_target_symbol_retained_rate: float | None = None
    weighted_target_trade_retained_rate: float | None = None
    weighted_target_bbo_sig_retained_rate: float | None = None
    weighted_burst_retention_rate: float | None = None
    weighted_target_selected_share: float | None = None
    symbol_with_zero_retained_tokens_count_p95: float = 0.0
    per_symbol_starvation_rate: dict[str, float] = Field(default_factory=dict)
    venue_candidate_share_by_venue: dict[str, float] = Field(default_factory=dict)
    venue_selected_share_by_venue: dict[str, float] = Field(default_factory=dict)
    venue_overrepresentation_ratio: dict[str, float] = Field(default_factory=dict)
    significant_bbo_emitted_count_by_reason: dict[str, int] = Field(default_factory=dict)
    significant_bbo_retained_count_by_reason: dict[str, int] = Field(default_factory=dict)
    significant_bbo_preservation_rate: float | None = None
    informative_candidate_by_tier: dict[str, int] = Field(default_factory=dict)
    t4_candidate_total: int = 0
    t4_anchor_total: int = 0
    t4_resolution_wall_sec: float = 0.0
    bbo_significance_wall_sec: float = 0.0
    quota_fill_wall_sec: float = 0.0
    diagnostics_serialization_wall_sec: float = 0.0
    total_selector_wall_sec: float = 0.0
    token_budget_pressure_row_count: int = 0
    token_budget_pressure_rate: float = 0.0
    budget_fill_by_tier: dict[str, int] = Field(default_factory=dict)
    drop_reason_counts_by_tier: dict[str, dict[str, int]] = Field(default_factory=dict)
    compression_ratio_by_family: dict[str, float] = Field(default_factory=dict)
    lane_cap_hit_rate: float = 0.0
    bbo_cap_hit_rate: float = 0.0
    symbol_cap_hit_rate: float = 0.0
    cross_venue_ordered_adjacency_rate: float = 0.0
    trade_to_bbo_ordered_adjacency_rate: float = 0.0


class EventTokenCacheDiagnosticsManifest(QuantBaseModel):
    format_version: str = EVENT_TOKEN_CACHE_DIAGNOSTICS_FORMAT_VERSION
    selection_policy_id: str = EVENT_TOKEN_SELECTION_POLICY_ID
    selector_params_hash: str | None = None
    audit_row_count: int = 0
    audit_row_ids: list[int] = Field(default_factory=list)
    audit_selector_policy_id: str | None = None
    audit_selector_params_hash: str | None = None
    audit_artifact_relative_path: str | None = None
    splits: dict[str, EventTokenSplitDiagnostics] = Field(default_factory=dict)


class EventTokenCacheRetentionReceipt(QuantBaseModel):
    format_version: str = EVENT_TOKEN_CACHE_FORMAT_VERSION
    source_manifest_path: str
    retained_shard_count: int
    missing_shard_count: int
    retained_payload_count: int
    missing_payload_count: int
    missing_payloads: list[str] = Field(default_factory=list)


__all__ = [
    "BBO_PAYLOAD_SCHEMA_ID",
    "EVENT_TOKEN_CACHE_DIAGNOSTICS_FILENAME",
    "EVENT_TOKEN_CACHE_DIAGNOSTICS_FORMAT_VERSION",
    "EVENT_TOKEN_CACHE_DIRNAME",
    "EVENT_TOKEN_CACHE_FORMAT_VERSION",
    "EVENT_TOKEN_CACHE_MANIFEST_FILENAME",
    "EVENT_TOKEN_CACHE_RETENTION_RECEIPT_FILENAME",
    "EVENT_TOKEN_CACHE_SUMMARY_FILENAME",
    "EVENT_TOKEN_SELECTION_POLICY_ID",
    "EVENT_TOKENIZER_VERSION",
    "EVENT_WINDOW_CONTRACT_VERSION",
    "EventTokenSelectionHyperparameters",
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
