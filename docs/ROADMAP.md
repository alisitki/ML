# Roadmap

## Purpose

This document defines the phase order for QuantLab.

It answers:

- what phase the repo is actually in
- what must be true before the next phase becomes the main focus
- what each later phase is allowed to claim

Use `docs/PROJECT_STATE.md` for the short current snapshot and `docs/BACKLOG.md` for the active execution queue.

---

## Current position

QuantLab is still in Phase 1 hardening.

Phase 2 is the next planned build phase, but it does not become the main workstream automatically. Phase 1 exit criteria must be satisfied, or explicitly frozen with written justification, before live/runtime buildout becomes the dominant focus.

---

## Phase 1 - Canonical and Offline Foundation

Purpose:

- freeze market scope and canonical semantics
- build trajectory, training, evaluation, artifact, and registry discipline
- define runtime and executor contracts without pretending the live stack already exists
- close or explicitly freeze temporary continuity windows

Entry criteria:

- declared market scope exists
- canonical contracts exist or are being stabilized

Exit criteria:

- offline engine is materially operational on current HEAD
- offline closure criteria are explicit
- continuity audit can distinguish clear, blocked, and active-dependency cases safely
- repo-truth docs are aligned on current-vs-target capability
- remaining continuity debt is either retired, frozen, or explicitly scoped

Current status:

- materially implemented
- still open on truthful offline-closure hardening

---

## Phase 2 - Runtime / Live Parity Foundation

Purpose:

- build websocket ingestion across the declared market scope
- build online state and feature services
- prove replay-vs-live parity
- define degraded-input, stale-state, reconnect, and recovery behavior
- stand up a selector runtime that consumes declared live state only

Entry criteria:

- Phase 1 offline-closure blockers are no longer ambiguous
- continuity windows are retired or explicitly frozen with evidence-backed scope
- current repo state is honest about what remains unproven

Exit criteria:

- live ingestion and canonical normalization are materially implemented
- online state semantics are explicit
- replay-vs-live parity checks exist
- stale and recovery behavior is testable
- selector runtime exists without moving strategy logic into the executor

Status rule:

- planned next phase
- not the main focus while Phase 1 exit criteria remain open

---

## Phase 3 - Shadow / Paper + Thin Executor Operating Loop

Purpose:

- connect selector runtime to a thin executor
- prove shadow/paper operation under explicit venue/risk controls
- make decision traces reconstructable from input to execution intent

Entry criteria:

- Phase 2 parity and degraded-state behavior are materially implemented
- selector runtime exists with explicit state inputs and traceability

Exit criteria:

- shadow/paper loop exists
- executor integration is thin and explicit
- kill-switch and feasibility controls are wired
- decision traces can be reconstructed end to end

---

## Phase 4 - Commercialization Readiness

Purpose:

- turn target gates into operational evidence
- package offline, parity, shadow, and control evidence for review
- make readiness claims machine-checkable rather than narrative only

Entry criteria:

- shadow/paper operation exists
- gate evidence inputs come from real retained system runs

Exit criteria:

- gate evidence is generated systematically
- shadow/paper evidence is attached to promotion decisions
- readiness reporting is explicit about what is evidenced vs only defined

---

## Phase 5 - Pilot / Live Operations Maturity

Purpose:

- run cautious pilot exposure under reviewed controls
- learn from incidents and real frictions
- decide whether scaling is justified

Entry criteria:

- commercialization-readiness evidence exists
- pilot controls and runbooks are reviewable

Exit criteria:

- pilot controls are proven
- post-cost behavior survives live frictions
- operational learning is recorded
- scale-up decisions are evidence-backed

---

## Ordering rule

Phases are sequential by default.

Do not read a later phase heading as permission to start now regardless of blockers. If a dependency is waived, the deviation must be written explicitly into `docs/PROJECT_STATE.md` and reflected in `docs/BACKLOG.md`.
