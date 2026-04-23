# Observation Schema

## Purpose

This document defines the canonical observation surface consumed by QuantLab learning and runtime inference systems.

This is the canonical reference for:
- observation axes
- tensor layout
- mask semantics
- scale semantics
- derived-surface attachment

Canonical version id:
- observation schema version -> `observation_schema_v1`
- event window contract version -> `event_window_contract_v2`

## 1. Core principle

The observation surface must preserve market structure without collapsing:
- exchange
- symbol
- stream
- field
- time scale

The core observation surface is not allowed to use exchange averaging or stream scalar collapse.

## 2. Canonical axes

The canonical observation layout is:

`[window_scale, time_bucket, symbol, exchange, stream, field]`

This is the conceptual layout.
Implementation may store blocks per scale or equivalent structured tensors, but semantics must remain identical.

## 3. Required parallel tensors

For every coordinate, the observation surface must preserve:

- `values`
- `age`
- `missing`
- `stale`
- `padding`
- `availability_by_contract`

These may be stored as separate tensors or blocks, but semantics must remain explicit.

## 4. Scale axis

The canonical v1 production preset is:

- `1m × 8`
- `5m × 8`
- `15m × 8`
- `60m × 12`

The canonical fixture/smoke preset is:

- `1m × 4`

Scale-specific bucket semantics must remain independent.
A scale may not silently overwrite or flatten another scale.

## 5. Time buckets

Each time bucket represents a deterministic aggregation or latest-known-state snapshot under the configured resolution.

Bucket generation must be:
- causal
- deterministic
- versioned

## 6. Target asset and context

Every sample has:
- one `target_symbol`
- one `decision_timestamp`

But the observation context may include:
- all configured symbols
- all configured exchanges
- all configured streams
- all configured fields

Cross-symbol context is allowed.
Action ownership remains single-asset.

## 7. Required stream/field presence

The observation surface must include field families specified in the canonical market data contract.

A stream may not be reduced to a single scalar in the core observation surface.

## 8. Derived surface

Derived channels are allowed, but only as augmentation.

The observation surface may include:
- `raw_surface`
- `derived_surface`

Rules:
- raw surface is the core/default path
- raw surface is primary
- derived surface must not replace raw channels
- derived surface is augmentation only
- derived channels must be versioned
- derived channels must be documented
- derived channels must declare whether they are target-centric or broader
- derived channels remain optional experiment paths unless explicitly promoted by evidence and an explicit decision/state update
- derived feature support in one model or runtime does not make handcrafted derived channels a future requirement
- default assumption is raw-first; derived only when ablation/evidence justifies it

## 9. V1 derived-surface scope

The v1 derived-surface scope is target-centric.

Allowed v1 derived families include:
- target-symbol bid-ask spread
- target-symbol orderbook imbalance
- target-symbol signed trade-flow proxy
- target-symbol OI delta
- target-symbol funding delta
- freshness / latency channels
- target-symbol venue-pair price spread
- target-symbol relative move versus each other configured symbol

Venue-pair spread in v1 is full pairwise across available venues for the target symbol.

Allowed derived families do not imply default dependence.
Support for these channels does not elevate them to the core/default observation path by itself.

## 9. Parallel event window

`observation_schema_v1` remains the authoritative coarse snapshot surface.

Current `HEAD` also supports a parallel offline-only sibling artifact:

- `event_token_cache_v1`

Rules:

- it is row-aligned to existing decision rows
- it does not replace `observation_schema_v1`
- it does not change current decision cadence
- it is the first serious data-depth step, not the final ceiling
- cadence redesign remains a separate later research question

Current implemented event-window scope:

- window: `[decision_timestamp - 60s, decision_timestamp]`
- token cap: `256`
- streams: `trade`, `bbo`
- selection policy: `significant_bbo_priority_window_v2`
- selector numbers are proof-slice hyperparameters, not universal system truths
- event retention now happens on emitted informative units rather than raw recency-only overflow
- recent target-relevant trades remain highest-fidelity; older bursts and BBO lanes are compressed before token-cap selection
- BBO lanes emit canonical significant quote events with explicit precedence:
  - `liquidity_vacuum`
  - `spread_regime_jump`
  - `mid_excursion`
  - `imbalance_regime_flip`
  - `burst_boundary`
- non-target `T4` units are proximity-based target-causal candidates only; they are not declared true causal edges
- unsupported lanes stay authoritative in snapshot masks rather than being synthesized into fake event emptiness
- ordering is anchored on `event_time` with deterministic tie-break `(event_time, source_label, source_event_index)`

## 10. Mask semantics

### `availability_by_contract`
True if the exchange-stream-field coordinate is expected by design.

### `missing`
True if the coordinate is contract-available but no valid value exists at the required point.

### `stale`
True if a valid value exists but exceeds freshness bound.

### `padding`
True if insufficient history exists to populate the requested lookback slot.

These states are mutually exclusive.
Construction precedence is:
`padding -> contract-unavailable (availability_by_contract = False) -> missing -> stale -> valid`

These states must never be collapsed.

## 11. Causality rule

Observation may depend only on information available at or before the decision timestamp.

Future timestamps, future prices, future masks, or future feasibility information are forbidden in observation construction.

## 12. Compatibility rule

Legacy scalar reductions may exist only in compatibility adapters.

Runtime must reject observations before inference if the live observation contract diverges from the artifact's required contract, including:
- observation schema version mismatch
- scale-spec mismatch
- raw tensor shape mismatch
- derived-surface contract/version mismatch
- derived channel identity/order mismatch

The canonical observation schema is not allowed to regress into:
- target-only mark-price series
- exchange-averaged series
- stream-first-value collapse
- flattened scalar-only surfaces
