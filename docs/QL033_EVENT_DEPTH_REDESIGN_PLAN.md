# QL-033 Event-Depth Redesign Plan

## Scope status

Current implemented scope on `HEAD`:

- `event_token_cache_v1` is an offline-only sibling artifact written next to `tensor_cache_v1`
- it preserves row-aligned `trade` + `bbo` event windows for each existing decision row
- it does not replace `observation_schema_v1`
- it does not yet change Phase 1A objectives, reward semantics, runtime export, or live-path behavior

Current gate:

- only `Phase A` and `Phase B` are implemented on current `HEAD`
- `Phase C` stays gated until the proof slice shows that `event_token_cache_v1` is replayable, stable, and informative

This is the first serious data-depth step, not the final ceiling.
Decision rows remain anchored to the existing `sampling_interval_seconds` cadence.
Cadence redesign is a separate later research question.

## Event window contract

Canonical version id:

- event window contract -> `event_window_contract_v1`

Row anchor:

- one event window per existing decision row
- anchor timestamp = row decision timestamp
- window = closed interval `[decision_timestamp - 60s, decision_timestamp]`
- first implemented streams = `trade`, `bbo`
- unsupported lanes are excluded by contract and remain authoritative only in snapshot masks

## Token overflow policy

Hard cap:

- `token_cap = 256`

Selection order:

1. Build the candidate set from all eligible, deduplicated `trade` and `bbo` events in the window.
2. If `candidate_count <= 256`, keep all tokens.
3. If `candidate_count > 256`, apply the mixed policy below.

Mixed policy:

- global recency reserve:
  - keep the newest `64` candidates first
- burst reserve:
  - define a burst per `(exchange, symbol, stream)` lane when consecutive inter-arrival gaps are `<= 250ms`
  - for each lane, take the last event of the most recent two bursts if not already selected
  - from that pool, keep the newest remaining candidates up to `48` tokens
- round-robin fill:
  - fill remaining capacity in this fixed order:
    - `trade@binance`
    - `trade@bybit`
    - `trade@okx`
    - `bbo@binance`
    - `bbo@bybit`
    - `bbo@okx`
  - within each bucket, pick the newest unselected event first

Required overflow telemetry:

- row level:
  - `candidate_token_count`
  - `selected_token_count`
  - `dropped_token_count`
  - `truncated`
  - `dropped_by_stream`
  - `dropped_by_venue`
  - `dropped_by_symbol`
  - `retained_by_symbol`
  - `target_symbol_retained_rate`
  - `symbol_with_zero_retained_tokens_count`
  - `burst_count`
  - `retained_burst_count`
  - `burst_retention_rate`
- split level:
  - `truncation_rate`
  - aggregate `dropped_by_stream`
  - aggregate `dropped_by_venue`
  - aggregate `dropped_by_symbol`
  - weighted `target_symbol_retained_rate`
  - weighted `burst_retention_rate`

Symbol fairness rules:

- symbol starvation must be measurable explicitly
- `target_symbol_retained_rate` is only defined when the target symbol has candidates
- `symbol_with_zero_retained_tokens_count` counts symbols with candidates but zero selected tokens

## Stream-typed payload schemas

### `trade_payload_v1`

Field order:

1. `price`
2. `qty`
3. `side_or_signed_flow_proxy`
4. `event_delta`
5. `count_or_burst`

Formula freeze:

- `price`
  - formula: current trade execution price
  - reference quantity: current trade
  - unit: quote-currency per base-unit
  - status: raw, unnormalized
  - missing behavior: write `0.0`, `presence=false`
- `qty`
  - formula: current matched trade quantity
  - reference quantity: current trade
  - unit: base-unit quantity
  - status: raw, unnormalized
  - missing behavior: write `0.0`, `presence=false`
- `side_or_signed_flow_proxy`
  - formula: `qty_t * side_sign_t`
  - reference quantity: current trade `qty`
  - unit: signed base-unit quantity
  - status: engineered, unnormalized
  - missing behavior: write `0.0`, `presence=false`
- `event_delta`
  - formula: `price_t - price_{t-1}` on the same `(exchange, symbol, trade)` lane after dedup and authoritative ordering
  - reference quantity: previous same-lane trade price
  - unit: quote-currency per base-unit
  - status: engineered, unnormalized
  - missing behavior: write `0.0`, `presence=false`
- `count_or_burst`
  - formula: running size of the contiguous same-lane burst ending at `t`, where the burst resets when inter-arrival gap `> 250ms`
  - reference quantity: previous same-lane event time
  - unit: event count
  - status: engineered, unnormalized integer-as-float
  - missing behavior: always present, minimum `1.0`

### `bbo_payload_v1`

Field order:

1. `bid_price`
2. `ask_price`
3. `bid_size`
4. `ask_size`
5. `spread`
6. `mid`
7. `imbalance_inputs`

Formula freeze:

- `bid_price`
  - formula: current best bid
  - reference quantity: current quote
  - unit: quote-currency per base-unit
  - status: raw, unnormalized
  - missing behavior: write `0.0`, `presence=false`
- `ask_price`
  - formula: current best ask
  - reference quantity: current quote
  - unit: quote-currency per base-unit
  - status: raw, unnormalized
  - missing behavior: write `0.0`, `presence=false`
- `bid_size`
  - formula: current displayed top-of-book bid size
  - reference quantity: current quote
  - unit: base-unit quantity
  - status: raw, unnormalized
  - missing behavior: write `0.0`, `presence=false`
- `ask_size`
  - formula: current displayed top-of-book ask size
  - reference quantity: current quote
  - unit: base-unit quantity
  - status: raw, unnormalized
  - missing behavior: write `0.0`, `presence=false`
- `spread`
  - formula: `ask_price - bid_price`
  - reference quantity: same-event top-of-book prices
  - unit: quote-currency per base-unit
  - status: engineered, unnormalized
  - missing behavior: write `0.0`, `presence=false`
- `mid`
  - formula: `(bid_price + ask_price) / 2`
  - reference quantity: same-event top-of-book prices
  - unit: quote-currency per base-unit
  - status: engineered, unnormalized
  - missing behavior: write `0.0`, `presence=false`
- `imbalance_inputs`
  - formula: `(bid_size - ask_size) / max(bid_size + ask_size, 1e-12)`
  - reference quantity: same-event total top-of-book size
  - unit: dimensionless ratio in `[-1, 1]`
  - status: engineered, normalized
  - missing behavior: write `0.0`, `presence=false`

## Ordering semantics

Authoritative time source:

- `event_time`
- `ts_event` may be coerced into `event_time`
- `ingest_time` is diagnostics-only

Same-timestamp ties:

- deterministic order key = `(event_time, source_label, source_event_index)`
- exact timestamp equality does not assert causal priority beyond deterministic ordering

Duplicate events:

- duplicate key = `(exchange, symbol, stream, event_time, canonical_payload_hash)`
- keep the first event under authoritative ordering
- drop the rest and report duplicate telemetry

Late / out-of-order events:

- offline build re-sorts by authoritative ordering key
- any event with `event_time <= decision_timestamp` remains eligible
- source-order inversions are measured and reported

Unsupported vs empty vs stale:

- `unsupported`
  - lane is absent by contract
  - event window never fabricates synthetic emptiness for that lane
- `empty_window`
  - supported lanes exist but the `[t-60s, t]` window has no eligible event after dedup
- `stale_window`
  - supported target-symbol reference lanes exist but the last event age exceeds `180s`

## Survivability rules

`full` retained behavior:

- canonical `event_token_cache_manifest.json` is present
- all referenced shard payloads are present
- replay receipt is complete

`slim` retained behavior:

- canonical manifest is absent when payload completeness is not guaranteed
- a summary artifact may remain for diagnostics only

`partial retain` behavior:

- defaults to slim
- if only a subset of shards is retained, use `event_token_cache_retention_receipt.json`
- do not keep a dangling canonical manifest

Hash-link rules:

- `event_token_cache_manifest` must carry:
  - `trajectory_manifest_hash`
  - `tensor_cache_manifest_hash`
  - `dataset_hash`
  - `tokenizer_version`
  - `event_window_contract_version`

## Phase C gate

When `C1` begins later:

- keep the same Phase 1A semantics and objective baseline
- change only representation, not action/reward/objective semantics

Current `HEAD` does not start `Phase C`.
