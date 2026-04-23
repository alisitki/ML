# QL-033 Event-Depth Redesign Plan

## Scope status

Current implemented scope on `HEAD`:

- `event_token_cache_v1` remains an offline-only sibling artifact beside `tensor_cache_v1`
- it remains row-aligned to existing decision rows
- it still covers `trade` + `bbo` only on the first redesign surface
- it does not replace `observation_schema_v1`
- it does not start `Phase C`, runtime work, or model adaptation

Current gate:

- `Phase A` and `Phase B` are implemented
- `Phase C` remains gated until the retained event structure is informative enough on proof-slice evidence

This redesign is current-phase hardening inside `QL-033`, not a competing workstream to `QL-031` and not a live/runtime claim.

## Event window contract

Canonical version ids:

- event window contract -> `event_window_contract_v2`
- tokenizer contract -> `event_tokenizer_contract_v2`
- selection policy id -> `significant_bbo_priority_window_v2`

Row anchor:

- one event window per existing decision row
- anchor timestamp = row decision timestamp
- window = closed interval `[decision_timestamp - 60s, decision_timestamp]`
- first redesign surface remains `trade`, `bbo`
- unsupported lanes remain excluded by contract and stay authoritative only in snapshot masks

## Selector parameter rule

The following numbers are versioned proof-slice hyperparameters for `significant_bbo_priority_window_v2`, not universal system truths:

- recent high-fidelity window length
- burst gap
- causal horizon
- tier quotas and caps
- `bbo_total_cap`
- `single_lane_cap`
- `single_symbol_cap`

Current proof-slice defaults:

- `recent_high_fidelity_seconds = 5`
- `burst_gap_ms = 250`
- `causal_horizon_ms = 1000`
- `token_cap = 256`
- `recent_bbo_extra_significant_limit = 3`
- `T0 = 64`
- `T1 = 32`
- `T2_T3 = 48`
- `T4 = 64`
- `T5_T6 = 32`
- `T7_GLOBAL_FLEX = 16`
- `bbo_total_cap = 96`
- `single_lane_cap = 40`
- `single_symbol_cap = 160`

These values must be written into the manifest and diagnostics and may be re-tuned later only by evidence.

## Retention policy

### Stage 0: ordering and deduplication

- authoritative time source remains `event_time`
- deterministic tie-break remains `(event_time, source_label, source_event_index)`
- duplicate key remains `(exchange, symbol, stream, event_time, canonical_payload_hash)`
- replayability, row alignment, and deterministic ordering remain non-negotiable

### Stage 1: informative-unit emission

Per `(exchange, symbol, stream)` lane:

- define bursts where consecutive inter-arrival gaps are `<= burst_gap_ms`
- compress each burst into informative units before token-cap selection

Trade emission:

- `trade_recent_raw`
  - if burst end lag `<= recent_high_fidelity_seconds`
  - keep all burst trades
- `trade_older_summary`
  - if burst end lag `> recent_high_fidelity_seconds`
  - keep only:
    - `burst_end`
    - `max_abs_signed_flow`
  - if both point to the same raw event, emit one unit only

BBO emission:

- `bbo_recent_sig`
  - always emit `burst_end`
  - additionally emit up to `recent_bbo_extra_significant_limit` highest-salience significant quote events
- `bbo_older_sig`
  - always emit `burst_end`
  - additionally emit at most one highest-salience significant quote event

### BBO significance reasons

Supported reasons:

- `liquidity_vacuum`
- `spread_regime_jump`
- `mid_excursion`
- `imbalance_regime_flip`
- `burst_boundary`

An event may satisfy multiple significance conditions, but it emits only one canonical reason for per-reason accounting.

Canonical precedence:

1. `liquidity_vacuum`
2. `spread_regime_jump`
3. `mid_excursion`
4. `imbalance_regime_flip`
5. `burst_boundary`

Telemetry rule:

- `matched_reasons[]` keeps all matched reasons for debug/audit use
- `canonical_significance_reason` is the only field used for emitted and retained per-reason counts
- this avoids double-counting in split-level preservation metrics

### Salience formulas

Trade:

- `trade_salience = max(abs(side_or_signed_flow_proxy), abs(event_delta))`

BBO:

- `bbo_mid_move_score = abs(mid_t - mid_burst_start) / max(abs(mid_burst_start), 1e-12)`
- `bbo_spread_score = spread_t / max(spread_burst_start, 1e-12)`
- `bbo_imbalance_score = abs(imbalance_t - imbalance_burst_start)`
- `bbo_liquidity_vacuum_score = min_side_size_burst_start / max(min(bid_size_t, ask_size_t), 1e-12)`
- `bbo_salience = max(mid_move_score, spread_score, imbalance_score, liquidity_vacuum_score)`

## Priority tiers

Base tiers:

- `T0`: target-symbol `trade_recent_raw`
- `T1`: target-symbol `bbo_recent_sig`
- `T2`: target-symbol `trade_older_summary`
- `T3`: target-symbol `bbo_older_sig`
- `T5`: non-target `trade_recent_raw`
- `T6`: non-target `bbo_recent_sig`
- `T7`: non-target older summaries

### T4 target-causal candidate rule

`T4` is a proximity heuristic for target-relevant trigger candidates. It is not a claim of true causality.

Anchor set:

- only target-symbol informative units already emitted into `T0-T3`
- raw target events that were dropped during compression do not become anchors

Eligibility:

- non-target event must be within `± causal_horizon_ms` of at least one target anchor
- and must satisfy one of:
  - `same_symbol + different_venue`
  - `same_venue + different_symbol`

If one non-target event matches multiple anchors:

- emit it once only
- choose `best_anchor` by:
  1. smallest absolute time delta
  2. higher anchor tier priority (`T0 > T1 > T2 > T3`)
  3. higher anchor salience
  4. newer anchor event time
  5. deterministic `(source_label_id, source_event_index)`

## Quotas and caps

Tier-group quotas:

- `T0 = 64`
- `T1 = 32`
- `T2_T3 = 48`
- `T4 = 64`
- `T5_T6 = 32`
- `T7_GLOBAL_FLEX = 16`

Global caps:

- `token_cap = 256`
- `bbo_total_cap = 96`
- `single_lane_cap = 40`
- `single_symbol_cap = 160`

Floors:

- `T0`
  - venue min `16`
  - venue max `32`
- `T1`
  - venue min `8`
  - venue max `16`
- `T2_T3`
  - venue min `8`
  - venue max `24`
- `T4`
  - initial floor `4` per eligible non-target symbol

Tie-break inside a tier:

1. `priority_tier`
2. lag bucket: `<=5s`, `5-15s`, `15-60s`
3. higher salience
4. newer `event_time`
5. deterministic `(source_label_id, source_event_index)`

Overflow behavior:

- if a tier-group has unused budget, it may spill only downward
- if a unit is blocked, its final drop reason must be explicit:
  - `bucket_overflow`
  - `bbo_cap`
  - `lane_cap`
  - `symbol_cap`
  - `lost_after_compression`

## Required telemetry

Row level:

- `candidate_token_count`
- `informative_candidate_count`
- `selected_token_count`
- `dropped_token_count`
- `truncated`
- `token_budget_pressure`
- `candidate_by_symbol`
- `candidate_by_venue`
- `candidate_by_stream`
- `informative_candidate_by_symbol`
- `informative_candidate_by_venue`
- `informative_candidate_by_stream`
- `retained_by_symbol`
- `retained_by_venue`
- `retained_by_stream`
- `dropped_by_symbol`
- `dropped_by_venue`
- `dropped_by_stream`
- `target_symbol_retained_rate`
- `raw_target_symbol_retained_rate`
- `target_trade_retained_rate`
- `target_bbo_sig_retained_rate`
- `target_selected_share`
- `burst_count`
- `retained_burst_count`
- `burst_retention_rate`
- `significant_bbo_emitted_count_by_reason`
- `significant_bbo_retained_count_by_reason`
- `significant_bbo_preservation_rate`
- `budget_fill_by_tier`
- `drop_reason_counts_by_tier`
- `raw_has_target_cross_venue_ordered_adjacency`
- `retained_has_target_cross_venue_ordered_adjacency`
- `raw_has_target_trade_to_bbo_ordered_adjacency`
- `retained_has_target_trade_to_bbo_ordered_adjacency`

Split level:

- `truncation_rate`
- `weighted_target_symbol_retained_rate`
- `weighted_raw_target_symbol_retained_rate`
- `weighted_target_trade_retained_rate`
- `weighted_target_bbo_sig_retained_rate`
- `weighted_burst_retention_rate`
- `weighted_target_selected_share`
- `per_symbol_starvation_rate`
- `venue_candidate_share_by_venue`
- `venue_selected_share_by_venue`
- `venue_overrepresentation_ratio`
- `significant_bbo_emitted_count_by_reason`
- `significant_bbo_retained_count_by_reason`
- `significant_bbo_preservation_rate`
- `token_budget_pressure_rate`
- `budget_fill_by_tier`
- `drop_reason_counts_by_tier`
- `compression_ratio_by_family`
- `lane_cap_hit_rate`
- `bbo_cap_hit_rate`
- `symbol_cap_hit_rate`
- `cross_venue_ordered_adjacency_rate`
- `trade_to_bbo_ordered_adjacency_rate`

## Stream-typed payload schemas

Payload schemas remain unchanged on current `HEAD`:

- `trade_payload_v1`
- `bbo_payload_v1`

The redesign changes retention semantics and diagnostics, not the payload field order.

## Survivability rules

`full` retained behavior:

- canonical `event_token_cache_manifest.json` is present
- all referenced shard payloads are present

`slim` retained behavior:

- canonical manifest is absent when payload completeness is not guaranteed
- `event_token_cache_manifest.summary.json` may remain for diagnostics only

`partial retain` behavior:

- if only a subset of event-token payloads is retained, write:
  - `event_token_cache_manifest.summary.json`
  - `event_token_cache_retention_receipt.json`
- do not keep a dangling canonical manifest

Hash-link rules:

- manifest must carry:
  - `trajectory_manifest_hash`
  - `tensor_cache_manifest_hash`
  - `dataset_hash`
  - `event_window_contract_version`
  - `tokenizer_version`
  - `selection_policy_id`
  - `selection_hyperparameters`
  - `selector_params_hash`

## Proof-slice note

Current target for the next proof-slice is still:

- keep `60s` cadence
- keep `trade + bbo` surface only
- keep `token_cap = 256`
- prove that retention policy v2 preserves useful market structure before any Phase C work begins

Current `HEAD` does not start `Phase C`.
