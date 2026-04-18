---
status: canonical
owner: quantlab
last_reviewed: 2026-04-17
read_when:
  - before_non_trivial_code_changes
  - before_data_or_runtime_changes
supersedes: []
superseded_by: []
---

# Market Scope

## Purpose

This document defines the active trading universe and canonical market-data surface for QuantLab.

If code, configs, or experiments assume a different universe without explicit approval, they are out of scope.

---

## Active trading universe

### Exchanges
- Binance
- Bybit
- OKX

### Instrument class
- futures / perpetual-style derivatives

### Active symbols
- BTCUSDT
- ETHUSDT
- BNBUSDT
- SOLUSDT
- XRPUSDT
- LINKUSDT
- ADAUSDT
- AVAXUSDT
- LTCUSDT
- MATICUSDT

Universe expansion is not the default path.

---

## Canonical stream families

The canonical market-data surface contains five stream families:

1. `trade`
2. `bbo`
3. `mark_price`
4. `funding`
5. `open_interest`

This five-family surface is canonical even when venue availability is sparse.

---

## Venue availability matrix

### Binance
- trade
- bbo
- mark_price
- funding
- open_interest: unsupported

### Bybit
- trade
- bbo
- mark_price
- funding
- open_interest

### OKX
- trade
- bbo
- mark_price
- funding
- open_interest

---

## Required semantic distinctions

These distinctions are mandatory throughout the system:

- `unsupported` = this venue does not provide this stream family in active scope
- `missing` = the venue supports it, but the observation is absent for the relevant step/window
- `stale` = the last known value exists but freshness is outside accepted limits
- `padding` = synthetic placeholder used only for shape control where explicitly allowed

These states must never be collapsed into one another without explicit versioned design.

In particular:

- unsupported is not zero,
- missing is not unsupported,
- stale is not missing.

---

## Scale implications

The working data plane is high-volume websocket data with approximately 106 million events per day.

This means the system must assume:

- bursty arrival patterns,
- non-trivial ordering and recovery challenges,
- stateful online feature computation,
- deterministic replay requirements,
- explicit backpressure and observability needs.

Designs that only work for notebook-scale data are not strategic by default.

---

## Engineering implications

The active scope requires:

- event-native ingestion and normalization,
- exchange-aware semantics,
- stateful online feature updates,
- offline replay that matches runtime meaning,
- explicit freshness and stale-state policies,
- live-execution controls that respect venue-specific behavior.

---

## Non-goals

The following are not default goals:

- expanding to more exchanges without a commercial case,
- expanding to spot or options by default,
- hiding venue identity to simplify models,
- reducing the canonical surface for convenience alone,
- treating sparse availability as a bug.

---

## Decision rule

Any proposed change that alters the active universe, venue availability semantics, or canonical stream meaning requires explicit documentation updates and governance review.
