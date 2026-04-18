# AGENTS.md

## Mission

QuantLab is a multi-exchange futures ML trading system.

It ingests high-volume websocket market data, builds canonical exchange-aware state, trains and evaluates policies offline, runs ML inference at runtime, and hands controlled trade intent to a live executor.

The commercial objective is simple:

> produce live-deployable, post-cost-positive trading decisions from multi-exchange futures data without breaking parity, traceability, or risk controls.

This repository is not only a research scaffold.  
It is the system that must make research, runtime, and execution cohere.

---

## Market scope

The default market scope is defined in `docs/MARKET_SCOPE.md`.

Current scope:

- exchanges: Binance, Bybit, OKX
- instrument type: futures / perpetual-style derivatives
- symbols:
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
- canonical stream families:
  - trade
  - bbo
  - mark_price
  - funding
  - open_interest

Availability is sparse by venue.  
Sparse availability is part of the contract, not an implementation accident.

Do not widen the universe by default.

---

## Fixed system boundary

The system boundary is:

1. websocket ingestion
2. canonical event normalization
3. online feature/state construction
4. offline training and evaluation
5. runtime inference
6. risk-gated execution intent
7. thin live executor

The executor remains thin.

Allowed executor responsibilities:

- feasibility checks
- venue/risk constraints
- order submission and lifecycle handling
- kill-switch and safety enforcement

Forbidden executor responsibilities:

- hidden strategy selection
- hidden alpha logic
- hidden portfolio intelligence that bypasses upstream policy logic

---

## Primary engineering objective

When choosing between valid options, prefer the one that most directly improves one or more of:

1. post-cost live trading quality,
2. offline/online parity,
3. capital protection,
4. feature freshness and runtime safety,
5. research throughput on meaningful data volume,
6. retirement of temporary continuity debt.

Do not optimize the system around laptop convenience or weak compatibility expectations if that harms the live trading objective.

---

## Required read order

Before any non-trivial change, read in this order:

1. `docs/PRODUCT_THESIS.md`
2. `docs/MARKET_SCOPE.md`
3. `docs/ONLINE_RUNTIME_MODEL.md`
4. `docs/COMMERCIALIZATION_GATES.md`
5. `docs/QUANTLAB_CONSTITUTION.md`
6. `docs/PROJECT_STATE.md`
7. `docs/ROADMAP.md`
8. `docs/BACKLOG.md`
9. relevant canonical contracts
10. relevant runbooks

If active state and requested work conflict, justify the deviation explicitly.

---

## Required task classification

For every meaningful task, classify all four fields.

### 1. Layer

Choose one primary layer:

- `data_plane`
- `canonicalization`
- `online_feature_state`
- `offline_training`
- `evaluation`
- `runtime_inference`
- `executor_risk`
- `observability_recovery`
- `docs_governance`

### 2. Business effect

Choose one primary effect:

- `expected_edge`
- `parity_integrity`
- `capital_protection`
- `latency_freshness_safety`
- `research_throughput`
- `continuity_debt_retirement`
- `docs_hygiene_only`

### 3. Execution mode

Choose one:

- `smoke_debug`
- `continuity_baseline`
- `shadow_paper`
- `real_training`
- `live_path_change`

### 4. Risk focus

Choose all relevant:

- `unsupported_stream_misuse`
- `missing_vs_stale_confusion`
- `replay_mismatch`
- `leakage`
- `reward_drift`
- `runtime_feature_drift`
- `venue_semantic_drift`
- `execution_drift`
- `recovery_corruption`

---

## Non-negotiable system rules

### Canonical surface rules

- Canonical stream families remain explicit.
- Sparse venue availability is explicit.
- Unsupported is not zero.
- Missing is not unsupported.
- Stale is not missing.
- Padding is not a real observation.
- Venue identity must remain recoverable unless a higher-order document explicitly allows a reduction.

### Offline/online parity rules

- Feature semantics must match between offline replay and runtime state construction.
- The same canonical interpretation rules must apply in both paths.
- Runtime shortcuts that change feature meaning are forbidden.
- Any intentional divergence must be versioned, documented, and tested.

### Time and ordering rules

- Event-time and processing-time must not be silently conflated.
- Out-of-order handling rules must be explicit.
- Reconnect and recovery behavior must be explicit.
- Deduplication and idempotency rules must be explicit.
- State rebuild or replay equivalence must be testable.

### Evaluation rules

- Random split is forbidden.
- Walk-forward remains the default.
- Purge/embargo remain mandatory when overlap exists.
- Final untouched test is not a tuning surface.
- A single attractive slice is never enough.
- Search-budget transparency is mandatory.

### Runtime and live-trading rules

- Runtime consumes declared inference artifacts and declared online state only.
- Executor must not invent strategy logic.
- Venue-specific costs, funding, and feasibility must remain explicit when relevant.
- Safety behavior on stale or partial state must be explicit.
- No hidden fallback that silently changes decision meaning.

### Commercial rules

- “Code runs” is not “ready for money.”
- “Backtest improved” is not “ready for money.”
- “Shadow looked fine” is not “ready to scale.”
- Changes on the live path must improve either edge, parity, safety, or capital protection.

---

## Forbidden moves

Forbidden unless a higher-order canonical document changes the rule:

- encoding unsupported venue streams as zeros,
- silently merging venue semantics,
- changing runtime feature math without parity tests,
- replaying data with different semantics than runtime,
- moving alpha logic into the executor,
- weakening split discipline to speed up experiments,
- claiming live readiness from smoke or continuity evidence,
- widening the universe without a commercial reason,
- optimizing the primary path around local-laptop constraints,
- silently degrading on stale state without explicit policy.

---

## Required behavior from Codex

For every non-trivial task, Codex must:

1. classify the task,
2. identify the exact layer touched,
3. name the governing documents,
4. state whether offline/online parity is affected,
5. state whether live-path safety is affected,
6. state whether venue-specific semantics are affected,
7. choose the smallest safe implementation path,
8. add or update tests for every changed behavior,
9. update docs when semantics change,
10. update state docs when active status changes.

If the change touches runtime or execution behavior, Codex must also state:

- what happens on stale state,
- what happens on unsupported inputs,
- what happens on reconnect/recovery,
- what evidence proves parity is still intact.

---

## Definition of done

A meaningful task is done only when all are true:

- the layer and business effect are explicit,
- parity impact is explicit,
- live-path safety impact is explicit,
- relevant tests exist,
- remaining risks are named,
- docs are updated if semantics changed,
- state docs are updated if project status changed,
- next recommended task is clear.

If any of these is missing, the task is incomplete.
