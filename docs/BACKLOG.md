# Backlog

## Purpose

This file tracks open work in a phase-aware way.

The backlog is organized into:

- current-phase hardening
- next-phase live-path buildout
- later commercialization work

Detailed history for completed foundation work belongs in `docs/DECISIONS.md`, `docs/PROJECT_STATE.md`, and git history rather than in a long mixed-status ledger here.

Status values:

- `todo`
- `in_progress`
- `blocked`

---

## Current-phase hardening

### QL-016

- phase_bucket: `current-phase hardening`
- title: Close the remaining NumPy / legacy continuity inventory
- status: `blocked`
- why_open:
  - PyTorch is the active core-training path, but final freeze/retirement of NumPy continuity depends on an authoritative registry audit
  - the reduced local QL-021 proof bundle is not a closure-grade audit source because copied registry records still reference remote `/root/runs/...` artifact paths
- next_action:
  - obtain the authoritative active registry root(s) or a relocation-safe audit bundle
  - run `quantlab-ml audit-continuity --registry-root ...`
  - freeze or retire the temporary continuity paths if zero active dependency is confirmed
- done_when:
  - external continuity audit no longer shows NumPy as an active core-training dependency
  - the temporary continuity path no longer reads like an open-ended default

### QL-004

- phase_bucket: `current-phase hardening`
- title: Close the remaining observation-schema continuity debt
- status: `in_progress`
- why_open:
  - the temporary legacy compat window still exists for deterministic legacy `linear-policy-v1` artifacts
  - current-phase hardening should narrow or retire continuity tools once active dependency is disproven
- next_action:
  - use the continuity audit outcome to decide whether the remaining legacy compat window can be frozen or removed
  - keep derived-surface and runtime-contract semantics explicit until the window closes
- done_when:
  - the remaining compat window is either retired or explicitly frozen with no ambiguity about its role
  - observation/runtime contract boundaries remain test-covered

### QL-014

- phase_bucket: `current-phase hardening`
- title: Apply interpretation guardrails across task intake and naming
- status: `todo`
- why_open:
  - current-phase truth can still blur allowed, temporary, optional, and default paths if naming or intake discipline drifts
- next_action:
  - require task intake to use phase classification consistently
  - clean up misleading continuity-oriented naming only through explicit compatibility changes
- done_when:
  - new work is less likely to mistake continuity or later-phase surfaces for the default path
  - task intake consistently records phase-aware classification

---

## Next-phase live-path buildout

### QL-024

- phase_bucket: `next-phase live-path buildout`
- title: Build websocket ingestion and canonical event normalization services
- status: `todo`
- why_now:
  - the repo defines live-path semantics, but it does not yet materially implement live ingestion across the declared market scope
- done_when:
  - websocket ingestion exists for the active exchanges and symbols
  - venue-specific parsing and canonical normalization are explicit and testable
  - unsupported/missing/stale semantics remain intact on the live path

### QL-025

- phase_bucket: `next-phase live-path buildout`
- title: Build the online state / feature service
- status: `todo`
- why_now:
  - runtime inference cannot be credible until live state and feature semantics exist as a declared service rather than as documentation only
- done_when:
  - online state updates are explicit
  - freshness, stale-state, deduplication, and idempotency rules are implemented
  - unsupported vs missing vs stale semantics remain distinct

### QL-026

- phase_bucket: `next-phase live-path buildout`
- title: Implement replay-vs-live parity, degraded-input, and recovery tooling
- status: `todo`
- why_now:
  - the future live path needs evidence that runtime state matches replay semantics and behaves safely under reconnect or degraded input
- done_when:
  - replay-vs-live parity checks exist
  - stale-state and partial-input behavior are explicit
  - reconnect and recovery behavior is testable

### QL-027

- phase_bucket: `next-phase live-path buildout`
- title: Stand up the selector runtime daemon
- status: `todo`
- why_now:
  - exported inference artifacts and execution-intent schemas exist, but there is no long-running runtime that consumes live state and produces decisions
- done_when:
  - selector runtime consumes declared inference artifacts and declared online state only
  - runtime decisions are traceable
  - no strategy logic is pushed into the executor

### QL-028

- phase_bucket: `next-phase live-path buildout`
- title: Integrate the thin executor and live controls
- status: `todo`
- why_now:
  - live operation requires explicit feasibility, venue/risk controls, and kill-switch behavior without widening the executor boundary
- done_when:
  - selector-to-executor handoff is explicit
  - feasibility and safety checks are enforced
  - executor remains thin and reconstructable

### QL-029

- phase_bucket: `next-phase live-path buildout`
- title: Build the shadow/paper operating loop
- status: `todo`
- why_now:
  - QuantLab needs a live-style operating loop before any pilot or commercialization claim is credible
- done_when:
  - shadow/paper operation exists
  - decision traces are reconstructable end to end
  - runtime, controls, and execution evidence can be reviewed together

---

## Later commercialization work

### QL-030

- phase_bucket: `later commercialization work`
- title: Automate commercialization-gate evidence and reporting
- status: `todo`
- why_later:
  - gate definitions exist now, but systematic gate evidence should follow the live-path buildout rather than substitute for it
- done_when:
  - gate evidence is generated from actual system runs
  - readiness reporting separates defined gates from evidenced gates automatically

### QL-100

- phase_bucket: `later commercialization work`
- title: ONNX / TensorRT runtime acceleration exploration
- status: `todo`
- why_later:
  - acceleration is not the next bottleneck until runtime selector maturity exists
- done_when:
  - selector runtime is stable
  - deployment artifact path exists

### QL-101

- phase_bucket: `later commercialization work`
- title: Reward v2 path-aware redesign
- status: `todo`
- why_later:
  - reward_v1 must stay stable while the live-operating half is being built
- done_when:
  - reward_v1 is stable and versioned
  - later commercialization evidence justifies a richer reward redesign

---

## Completed foundation milestones

The strongest completed work to date is the canonical and offline foundation:

- QL-001 through QL-003 aligned split and reward semantics with the declared contracts
- QL-005 through QL-006 established artifact, registry, and promotion discipline
- QL-008 through QL-023 built the current offline training, evaluation, streaming, tensor-cache, and state-reconciliation foundation

That foundation is the reason Phase 2 live-path buildout is now the right next step.
