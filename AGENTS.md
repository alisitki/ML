# AGENTS.md

## Mission

QuantLab's ultimate goal is an end-to-end multi-exchange futures ML trading system.

That target system:

1. ingests high-volume websocket market data
2. builds exchange-aware canonical state
3. trains and evaluates policies offline
4. runs runtime inference
5. hands controlled trade intent to a thin executor
6. advances through commercialization gates toward live capital deployment

The commercial objective is simple:

> produce live-deployable, post-cost-positive trading decisions from multi-exchange futures data without breaking parity, traceability, or risk controls.

Current implemented scope is stronger on canonical semantics, offline trajectory building, offline training/evaluation, artifact and registry discipline, and runtime/execution contracts than on live operation.

The next build phase is the live-operating half:

- websocket ingestion
- online state / feature service
- replay-vs-live parity tooling
- degraded-input behavior
- selector runtime
- shadow/paper loop
- thin executor integration and live controls

Treat `ultimate goal`, `current implemented scope`, and `next build phase` as three separate truth surfaces: destination, current HEAD reality, and planned next work. They must never be collapsed into a single claim.

This repository is not only a research scaffold, but it is also not yet a fully implemented live trading system. Agents must preserve both truths at the same time.

---

## Phase-awareness rules

- Always distinguish `ultimate goal`, `current implemented scope`, and `next build phase`.
- Treat those three labels as separate authority surfaces for destination, current implemented reality, and planned next work.
- A missing capability is not automatically a defect if it clearly belongs to a later planned phase.
- A document, task summary, or agent output must never describe a later-phase capability as current implemented reality unless code and evidence prove it.
- If target architecture and current repo reality differ, say so explicitly rather than flattening them together.
- Do not downscope the ambition merely because later-phase layers are not built yet.

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

The target end-state system boundary is:

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

The target boundary does not imply that every layer already exists in implemented form today.

---

## Primary engineering objective

When choosing between valid options, prefer the one that most directly improves one or more of:

1. post-cost live trading quality
2. offline/online parity
3. capital protection
4. feature freshness and runtime safety
5. research throughput on meaningful data volume
6. retirement of temporary continuity debt

Do not optimize the system around laptop convenience or weak compatibility expectations if that harms the live trading objective.

---

## Required read order

Before any non-trivial change, read in this order:

1. `README.md`
2. `docs/DOCS_INDEX.md`
3. `docs/PROJECT_STATE.md`
4. `docs/ROADMAP.md`
5. `docs/BACKLOG.md`
6. `docs/MARKET_SCOPE.md`
7. `docs/PRODUCT_THESIS.md`
8. `docs/ONLINE_RUNTIME_MODEL.md`
9. `docs/COMMERCIALIZATION_GATES.md`
10. `docs/QUANTLAB_CONSTITUTION.md`
11. `docs/RUNTIME_BOUNDARY.md`
12. relevant canonical contracts
13. relevant runbooks

Use `.agents/skills/quantlab-governance/SKILL.md` for governance-sensitive work before proposing or applying changes.
This fixed bootstrap order prioritizes current repo truth before aspirational target-state docs. Offline-closure and continuity docs stay conditional under relevant runbooks rather than in the mandatory bootstrap sequence.
If active state and requested work conflict, justify the deviation explicitly.

---

## Authority model

Use these precedence rules explicitly.

Current repo truth surface:

- `docs/PROJECT_STATE.md` defines what current `HEAD` materially is.
- `docs/ROADMAP.md` defines phase order, gating, and what is planned next versus current.
- `docs/BACKLOG.md` defines the active execution queue and current priority ordering.

If they disagree, resolve by specificity and recency within that current-truth surface rather than by reaching first for historical material.

Machine-readable continuity-closeout surface:

- `docs/continuity_closeout/*.yaml` are the machine-readable current closeout records.
- `docs/CONTINUITY_CLOSEOUT_RECORDS.md` defines the record model and interpretation rules.
- Historical continuity docs may explain how a closeout decision was reached, but they do not override the current YAML decision state.

Historical and target-state surfaces:

- `docs/history/` is evidence and chronology only.
- Historical docs never override newer current-head truth when `PROJECT_STATE.md`, `ROADMAP.md`, `BACKLOG.md`, or closeout YAMLs disagree.
- `docs/PRODUCT_THESIS.md`, `docs/ONLINE_RUNTIME_MODEL.md`, and `docs/COMMERCIALIZATION_GATES.md` define destination or later-phase architecture, not current implemented reality.
- `docs/DECISIONS.md`, canonical contract docs, and relevant runbooks are the constraint/governance surface: they do not define current status by themselves, but they constrain permissible execution and must not be violated.

Active-track rule:

- `QL-014` is the active docs-truth hardening item.
- `QL-031` is the single active next offline-hardening batch in this workspace.
- Docs hardening must not be framed as an alternative execution track that overrides `QL-031`.

---

## Required task classification

For every meaningful task, classify all five fields.
This ordering, these exact machine keys, and these cardinality semantics are canonical for repo templates and governance-skill output.

- `task_phase`: choose one primary
- `layer`: choose one primary
- `business_effect`: choose one primary
- `execution_mode`: choose one
- `risk_focus`: choose all relevant

### `task_phase`

Choose one primary:

- `target-state clarification`
- `current-phase hardening`
- `next-phase enablement`
- `optional research`
- `non-priority work`

### `layer`

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

### `business_effect`

Choose one primary:

- `expected_edge`
- `parity_integrity`
- `capital_protection`
- `latency_freshness_safety`
- `research_throughput`
- `continuity_debt_retirement`
- `docs_hygiene_only`

### `execution_mode`

Choose one:

- `smoke_debug`
- `continuity_baseline`
- `shadow_paper`
- `real_training`
- `live_path_change`

### `risk_focus`

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

- "Code runs" is not "ready for money."
- "Backtest improved" is not "ready for money."
- "Shadow looked fine" is not "ready to scale."
- Changes on the live path must improve either edge, parity, safety, or capital protection.

---

## Forbidden moves

Forbidden unless a higher-order canonical document changes the rule:

- encoding unsupported venue streams as zeros
- silently merging venue semantics
- changing runtime feature math without parity tests
- replaying data with different semantics than runtime
- moving alpha logic into the executor
- weakening split discipline to speed up experiments
- claiming live readiness from smoke or continuity evidence
- widening the universe without a commercial reason
- optimizing the primary path around local-laptop constraints
- silently degrading on stale state without explicit policy

---

## Required behavior from Codex

For every non-trivial task, Codex must:

1. classify the task across all five required fields using the canonical machine keys and ordering above
2. identify the exact layer touched
3. name the governing documents
4. state whether the task is target-state clarification, current-phase hardening, or next-phase enablement
5. state whether offline/online parity is affected
6. state whether live-path safety is affected
7. state whether venue-specific semantics are affected
8. state whether any missing capability is a current defect or later-phase work
9. choose the smallest safe implementation path
10. add or update tests for every changed behavior
11. update docs when semantics change
12. update state docs when active status changes

If the change touches runtime or execution behavior, Codex must also state:

- what happens on stale state
- what happens on unsupported inputs
- what happens on reconnect/recovery
- what evidence proves parity is still intact

---

## Definition of done

A meaningful task is done only when all are true:

- ultimate goal, current implemented scope, and next build phase are not blurred together
- `task_phase`, `layer`, `business_effect`, `execution_mode`, and `risk_focus` are explicit
- parity impact is explicit
- live-path safety impact is explicit
- relevant tests exist
- remaining risks are named
- docs are updated if semantics changed
- state docs are updated if project status changed
- next recommended task is clear

If any of these is missing, the task is incomplete.
