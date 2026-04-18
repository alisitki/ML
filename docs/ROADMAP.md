# Roadmap

## Purpose

This document defines the major build phases for QuantLab.

It answers:

- what the destination is
- which phase the repo is actually in
- what comes next
- what later phases depend on

Use `docs/PROJECT_STATE.md` for the short current snapshot.

---

## Current position

QuantLab is finishing Phase 1 hardening.

Phase 2 is the next major build phase.
Phases 3 through 5 remain target-state plan, not current capability claims.

---

## Phase 1 - Canonical and offline foundation

Purpose:

- freeze market scope and canonical semantics
- build trajectory, training, and evaluation discipline
- enforce artifact and registry lineage
- freeze runtime and executor contracts before live plumbing

Current status:

- materially implemented
- still needs current-phase hardening on continuity-debt closeout and truth-surface clarity

Remaining exit work:

- close QL-016 continuity audit and retire temporary core-training continuity paths when safe
- close remaining explicit compat windows rather than letting them read like growth areas
- keep README, project state, roadmap, backlog, and agent guidance phase-honest

---

## Phase 2 - Runtime/live parity foundation

Purpose:

- build websocket ingestion across the declared market scope
- build online state and feature services
- prove replay-vs-live parity
- define degraded-input, stale-state, and recovery behavior
- stand up a selector runtime that consumes declared live state

Exit criteria:

- live ingestion and canonical normalization are materially implemented
- online state semantics are explicit
- replay-vs-live parity checks exist
- stale/recovery behavior is testable
- selector runtime exists without moving strategy logic into the executor

This is the next major build phase.

---

## Phase 3 - Shadow/paper loop and executor controls

Purpose:

- connect selector runtime to a thin executor
- prove shadow/paper operation under explicit venue/risk controls
- ensure decision traces are reconstructable end to end

Exit criteria:

- shadow/paper loop exists
- executor integration is thin and explicit
- kill-switch and feasibility controls are wired
- live-style traces can be reconstructed from input to execution intent

---

## Phase 4 - Commercialization readiness

Purpose:

- turn target gates into operational evidence
- package replay, runtime, shadow, and control evidence for review
- make readiness claims machine-checkable rather than narrative only

Exit criteria:

- gate evidence is generated systematically
- shadow/paper evidence is attached to promotion decisions
- readiness reporting is explicit about what is evidenced vs only defined

---

## Phase 5 - Pilot/live operations maturity

Purpose:

- run cautious pilot exposure under reviewed controls
- learn from incidents and real frictions
- decide whether scaling is justified

Exit criteria:

- pilot controls are proven
- post-cost behavior survives live frictions
- operational learning is recorded
- scale-up decisions are evidence-backed

---

## Ordering rule

Phases are sequential by default.

Do not skip ahead unless:

- the dependency is explicitly waived
- the deviation is written into `docs/PROJECT_STATE.md`
- the reason is justified

Later phases do not override the current active focus in `docs/PROJECT_STATE.md`.
