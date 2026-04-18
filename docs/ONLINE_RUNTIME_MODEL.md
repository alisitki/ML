---
status: canonical
owner: quantlab
last_reviewed: 2026-04-17
read_when:
  - before_runtime_or_data-plane_changes
supersedes: []
superseded_by: []
---

# Online Runtime Model

## Purpose

This document defines the live-path operating model from websocket events to runtime inference and controlled execution.

Its core purpose is to preserve offline/online parity while supporting bounded-latency live trading.

---

## Live-path stages

The live path is:

1. websocket ingestion
2. venue-specific parsing
3. canonical event normalization
4. online feature/state update
5. runtime inference
6. risk and feasibility checks
7. execution intent emission
8. thin executor order handling
9. observability and recovery

Each stage must have explicit semantics.

---

## Offline/online parity rule

The live path and offline replay must agree on:

- canonical event meaning,
- supported vs unsupported treatment,
- missing and stale handling,
- state update rules,
- feature computation rules,
- normalization semantics where applicable,
- action-space interpretation where applicable.

If they differ, the live system is not commercially trustworthy.

---

## Online state requirements

The online state layer must define:

- event ordering policy,
- out-of-order handling,
- deduplication policy,
- idempotency policy,
- freshness policy,
- stale-state policy,
- reconnect behavior,
- warm-start or cold-start behavior,
- replay equivalence expectations.

No runtime feature layer is acceptable if these are implicit only.

---

## Safety rule for degraded inputs

When inputs are stale, partial, or unavailable, the system must follow an explicit policy.

Allowed patterns may include:

- no-trade / hold,
- restricted-action mode,
- kill-switch / hard stop,
- fallback only if feature semantics remain valid and documented.

Forbidden pattern:

- silently producing normal-confidence live actions from degraded state without an explicit design and evidence.

---

## Executor boundary

The executor remains thin.

It may perform:

- feasibility checks,
- venue/risk validation,
- order submission,
- lifecycle handling,
- emergency safety actions.

It may not perform:

- hidden alpha generation,
- hidden policy ranking,
- hidden strategy selection that bypasses runtime inference.

---

## Recovery and observability

The live path must make it possible to reconstruct:

- what inputs were seen,
- what canonical state existed,
- what features were computed,
- what inference was produced,
- what safety checks were applied,
- what order action was taken.

A live action that cannot be reconstructed is a control failure.

---

## Performance rule

The system must optimize for bounded live latency and predictable freshness, not just raw throughput.

Do not introduce heavy live-path computation unless it is justified by clear edge or safety benefits.

Latency and freshness budgets must be explicit, measured, and versioned once benchmarks are available.

---

## Promotion rule for live-path changes

A live-path change is not acceptable merely because unit tests pass.

For meaningful live-path changes, evidence should include as appropriate:

- replay parity checks,
- shadow or paper validation,
- stale-input behavior checks,
- recovery or reconnect checks,
- venue-specific correctness checks,
- artifact and logging traceability checks.

---

## Default decision rule

If a change improves offline metrics but weakens live-path parity, freshness, recovery, or reconstruction, reject or redesign the change.
