---
status: canonical
owner: quantlab
last_reviewed: 2026-04-18
read_when:
  - before_runtime_or_data-plane_changes
supersedes: []
superseded_by: []
---

# Online Runtime Model

## Purpose

This document defines QuantLab's target live-path architecture and the conditions required for offline/online parity.

It also states, explicitly, which runtime-related surfaces exist today and which do not.

This is a target-state architecture document. It does not override `docs/PROJECT_STATE.md`, `docs/OFFLINE_CLOSURE_CRITERIA.md`, or the current offline-hardening focus.

---

## Target runtime architecture

The target live path is:

1. websocket ingestion
2. venue-specific parsing
3. canonical event normalization
4. online feature/state update
5. runtime inference
6. risk and feasibility checks
7. execution intent emission
8. thin executor order handling
9. observability and recovery

Each stage must have explicit semantics before QuantLab can claim live-operating capability.

---

## Current implemented runtime-related surface

Today the repository materially provides runtime-adjacent groundwork, not a full live runtime:

- policy artifacts can be exported into inference artifacts
- runtime and executor boundaries are documented
- execution intent has an explicit schema
- runtime compatibility checks reject incompatible artifacts and observation layouts
- offline evaluation, scoring, export, registry, and continuity-audit flows exist

This is necessary groundwork for the live path, but it is not the same thing as a long-running runtime service that consumes live websocket state and drives an executor.

---

## Missing components before QuantLab has a live-operating runtime

The repository does not yet materially implement all of:

- production websocket ingestion across the active venue scope
- long-running venue parser and canonical event normalizer services
- an online state / feature service with explicit freshness management
- replay-vs-live parity tooling over live-style inputs
- a selector runtime daemon consuming declared online state
- executor integration on a shadow/paper or live control loop
- runtime observability and recovery evidence for the full live path

Until those exist, the runtime architecture remains a target design plus partial groundwork.

Phase 2 remains planned next work, but it should not become the main focus while offline closure still has blocked or ambiguous evidence.

---

## Target parity requirements

When the live path is implemented, offline replay and runtime must agree on:

- canonical event meaning
- supported vs unsupported treatment
- missing and stale handling
- state update rules
- feature computation rules
- normalization semantics where applicable
- action-space interpretation where applicable

QuantLab does not yet have end-to-end live evidence for this. The rule is still mandatory.

---

## Target online state requirements

The future online state layer must define:

- event ordering policy
- out-of-order handling
- deduplication policy
- idempotency policy
- freshness policy
- stale-state policy
- reconnect behavior
- warm-start or cold-start behavior
- replay equivalence expectations

These are target implementation requirements, not claims that the current repo already proves them in operation.

---

## Target safety rule for degraded inputs

When inputs are stale, partial, or unavailable, the future live system must follow an explicit policy.

Allowed patterns may include:

- no-trade / hold
- restricted-action mode
- kill-switch / hard stop
- fallback only if feature semantics remain valid and documented

Forbidden pattern:

- silently producing normal-confidence live actions from degraded state without an explicit design and evidence

---

## Executor boundary

The executor remains thin.

It may perform:

- feasibility checks
- venue/risk validation
- order submission
- lifecycle handling
- emergency safety actions

It may not perform:

- hidden alpha generation
- hidden policy ranking
- hidden strategy selection that bypasses runtime inference

This boundary is fixed even while the live-operating half is still being built.

---

## Recovery and observability target

The future live path must make it possible to reconstruct:

- what inputs were seen
- what canonical state existed
- what features were computed
- what inference was produced
- what safety checks were applied
- what order action was taken

A live action that cannot be reconstructed is a control failure.

The current repo defines this requirement but does not yet provide full live-path evidence for it.

---

## Next implementation steps

The next build phase for the runtime half is:

1. websocket ingestion for the declared market scope
2. online state / feature service with explicit unsupported/missing/stale semantics
3. replay-vs-live parity and recovery tooling
4. degraded-input and stale-state behavior checks
5. selector runtime daemon
6. shadow/paper loop with thin executor integration and reconstructable traces

---

## Default decision rule

If a change improves offline metrics but weakens future live-path parity, freshness, recovery, or reconstruction, reject or redesign the change.
