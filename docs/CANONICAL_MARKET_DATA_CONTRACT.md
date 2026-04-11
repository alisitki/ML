# Canonical Market Data Contract

## Purpose

This document defines the canonical market-data contract used by QuantLab.

It governs:
- canonical event schema
- timestamp semantics
- exchange / symbol / stream identity
- field availability
- missing / stale / padding semantics
- learning-surface preservation rules

This is the canonical source for market-data semantics.

## 1. Core principle

QuantLab does not simplify away structural market information in the core data contract.

The core data contract must preserve:
- symbol separation
- exchange separation
- stream separation
- field-level information
- event-time semantics
- contract availability semantics

Compatibility reductions may exist only outside the core contract.

## 2. Event identity

Every canonical event must carry:

- `event_time`
- `ingest_time` if available
- `exchange`
- `symbol`
- `stream`
- `fields`

Optional:
- raw source metadata
- source file metadata
- partition metadata

Canonical event payloads use `fields` as the primary field carrier.
Any legacy scalar payload carrier may exist only in explicit compatibility paths and must not replace `fields` in canonical processing.

## 3. Timestamp semantics

### `event_time`
Canonical event timestamp for market semantics.

Use:
- ordering
- learning-surface alignment
- reward timeline construction
- split generation

### `ingest_time`
Collection or ingestion timestamp if available.

Use:
- operational diagnostics
- lag / freshness diagnostics
- not as the canonical market-time replacement

### Rule
Learning surface and split policy are anchored on `event_time`, not `ingest_time`.

## 4. Identity axes

The core learning surface must preserve these axes:

- `time`
- `symbol`
- `exchange`
- `stream`
- `field`

Canonical observation layout is conceptually:

`[window_scale, time_bucket, symbol, exchange, stream, field]`

Parallel masks/tensors must preserve:
- age
- missing
- stale
- padding
- availability_by_contract

## 5. Canonical streams

Supported streams include:

- `trade`
- `bbo`
- `mark_price`
- `funding`
- `open_interest`

Additional streams may be added later only through explicit contract change.

## 6. Canonical field families

Field families are divided into two classes:

### A. Raw or directly carried fields
Fields that come directly from source events or are direct source-preserving mappings.

### B. Canonical derived fields
Fields that are deterministic, local, non-predictive transformations that remain part of canonical per-stream representation.

Canonical derived fields must be explicitly named as such.
They do not replace raw structure.

## 7. Required canonical field families

### Trade
Required field family:

- `price`
- `qty`
- `side_or_signed_flow_proxy`
- `event_delta`
- `count_or_burst`

### BBO
Required field family:

- `bid_price`
- `ask_price`
- `bid_size`
- `ask_size`
- `spread`
- `mid`
- `imbalance_inputs`

### Mark price
Required field family:

- `mark_price`
- `event_delta`
- `index_price_if_available`

### Funding
Required field family:

- `funding_rate`
- `next_funding_time`
- `funding_update_age`

### Open interest
Required field family:

- `open_interest`
- `oi_delta`
- `oi_update_age`

## 8. Exchange-stream availability

Availability is part of the contract.

Example:
- Binance does not provide `open_interest` in the current default collector contract.
- Bybit and OKX do.

If both an exchange-level stream-availability list and a more specific contract override exist,
the explicit contract override wins.

Contract-unavailable coordinates must remain explicitly represented as unavailable.
They must not be confused with stale or missing data.

## 9. Missing vs stale vs padding vs contract-unavailable

These states are distinct and must never be collapsed.

### Contract-unavailable
The exchange-stream-field coordinate is not expected to exist by design.

Example:
- `binance/open_interest/*`

### Missing
The coordinate is contract-available but no valid value is present at the required point.

### Stale
A value exists, but its age exceeds the freshness bound.

### Padding
Insufficient history exists to populate the requested lookback window.

## 10. Resampling and bucketing

Resampling is allowed only in explicit learning-surface construction.

Rules:
- raw event identity must remain recoverable at the contract level
- bucketing must not silently average away exchange or symbol structure
- bucket semantics must be deterministic and documented

Multi-scale windows are allowed and expected.

## 11. Compact-source discovery

Where the current compact source is used, discovery is state-driven.

State file:
- `compacted/_state.json`

Current layout:
- `exchange=<exchange>/stream=<stream>/symbol=<symbol>/date=<YYYYMMDD>/data.parquet`
- `exchange=<exchange>/stream=<stream>/symbol=<symbol>/date=<YYYYMMDD>/meta.json`
- `exchange=<exchange>/stream=<stream>/symbol=<symbol>/date=<YYYYMMDD>/quality_day.json`

This describes the current compact-source discovery contract and does not relax the canonical event or learning-surface rules.

## 12. Derived channels

Derived channels are allowed, but only as augmentation.

Rules:
- raw channels remain primary
- derived channels do not replace raw fields
- derived channels must be versioned and documented
- derived channels may include:
  - venue spreads
  - relative moves
  - imbalance
  - signed flow proxy
  - OI delta
  - funding delta
- freshness / latency

## 13. Prohibited reductions in core data contract

Forbidden in the core contract:
- exchange averaging
- symbol averaging
- stream single-scalar collapse
- implicit field dropping
- hidden resampling that destroys structure

These may exist only in explicit compatibility layers.

## 14. Learning-surface relation

The data contract defines what can be preserved.
The learning surface defines how that preserved structure is assembled into trainable sequences.

The learning surface may compress representation, but it may not violate:
- axis identity
- contract availability semantics
- causality
