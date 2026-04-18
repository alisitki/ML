# QuantLab ML

QuantLab ML is a multi-exchange ML trading system for futures markets.

Current intended market scope:

- venues: Binance, Bybit, OKX
- symbols: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, LINKUSDT, ADAUSDT, AVAXUSDT, LTCUSDT, MATICUSDT
- canonical stream families: trade, bbo, mark_price, funding, open_interest
- venue-specific asymmetry:
  - Binance: trade, bbo, mark_price, funding
  - Bybit: trade, bbo, mark_price, funding, open_interest
  - OKX: trade, bbo, mark_price, funding, open_interest

The system exists to convert high-volume websocket market events into:

1. canonical market state
2. offline training and evaluation surfaces
3. runtime ML inference
4. live execution intent for a thin executor

This repository is not a broker/exchange connectivity product by itself.
It owns the research, artifact, runtime-inference, and execution-intent side of the stack.
The executor remains thin and is responsible for feasibility checks, capital controls, order submission, and order lifecycle handling.

## Core boundary

```text
websocket events
  -> canonicalization
  -> observation / state surfaces
  -> offline ML training and evaluation
  -> policy artifacts and registry
  -> runtime inference
  -> execution intent
  -> thin live executor
```

## System rules

- Time-ordered evaluation only.
- Random split is forbidden.
- Leakage tolerance is zero.
- Offline and online feature semantics must stay aligned.
- Unsupported stream coordinates are not the same as missing or stale values.
- Runtime uses inference artifacts only.
- The executor must not become the hidden strategy brain.
- Live promotion follows: offline evaluation -> paper/sim -> live candidate.

## Read this first

- `AGENTS.md`
- `docs/DOCS_INDEX.md`
- `docs/PRODUCT_THESIS.md`
- `docs/MARKET_SCOPE.md`
- `docs/ONLINE_RUNTIME_MODEL.md`
- `docs/COMMERCIALIZATION_GATES.md`
- `docs/PROJECT_STATE.md`

## Canonical technical docs

- `docs/QUANTLAB_CONSTITUTION.md`
- `docs/RUNTIME_BOUNDARY.md`
- `docs/CANONICAL_MARKET_DATA_CONTRACT.md`
- `docs/OBSERVATION_SCHEMA.md`
- `docs/ACTION_SPACE.md`
- `docs/REWARD_SPEC.md`
- `docs/REWARD_SPEC_V1.md`
- `docs/SPLIT_POLICY.md`
- `docs/SPLIT_POLICY_V1.md`
- `docs/POLICY_ARTIFACT_SCHEMA.md`
- `docs/REGISTRY_SCHEMA.md`
- `docs/EXECUTION_INTENT_SCHEMA.md`
- `docs/PROMOTION_GATE.md`
- `docs/EVALUATION_RUNBOOK.md`
- `docs/REMOTE_GPU_RUNBOOK.md`
