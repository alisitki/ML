# Project State

## Purpose

This is the short source of truth for what current HEAD actually is.

Use it to answer:

- what QuantLab ultimately aims to become
- what the repo materially implements today
- whether the offline side is professionally closed
- what is blocking the repo before live/runtime work becomes the main focus

Detailed historical narrative belongs in `docs/DECISIONS.md` and `docs/history/`.

---

## Ultimate goal

QuantLab aims to become an end-to-end multi-exchange futures ML trading system:

- websocket ingestion
- canonical exchange-aware state
- offline training and evaluation
- runtime inference
- thin-executor handoff
- commercialization gates toward live capital deployment

---

## Current phase

QuantLab is in late Phase 1 hardening.

The repo materially implements the canonical and offline foundation. It does not yet materially implement the live-operating half. Phase 2 remains planned next-phase work, not current implemented reality and not the main focus while offline closure blockers remain open.

---

## Current verdict

QuantLab is `offline operational but not professionally closed`.

Why:

- current HEAD contains a real offline engine with trajectory build, train, evaluate, score, export, and registry flows
- retained QL-021 proof bundles under `outputs/` show controlled remote-GPU hot-path evidence
- offline closure is still incomplete because continuity retirement depends on auditable registry scope, relocation-safe retained artifact handling, and broader evidence packaging discipline

---

## Current implemented strengths

- canonical exchange-aware market-data and observation semantics
- offline trajectory building
- walk-forward training and evaluation discipline
- artifact export and registry discipline
- runtime-facing contracts and thin-executor boundary definitions
- retained controlled-proof surfaces plus governance and runbook discipline

---

## Current missing layers

- production websocket ingestion across the active venue scope
- online state / feature service
- replay-vs-live parity tooling over live-style inputs
- selector runtime daemon
- thin executor integration and live control loop
- shadow/paper operating loop
- system-generated commercialization evidence above the offline gate

These missing layers are planned later-phase work, not current defects by default.

---

## Current focus

- keep repo-truth docs aligned with current HEAD rather than planned target-state architecture
- harden `quantlab-ml audit-continuity` so zero-record and unreadable-retained-artifact cases become explicit blocked results
- define explicit offline-closure criteria and continuity-audit procedure
- leave evidence-dependent items visible instead of writing optimistic closure language

---

## Blocked before live-path focus

- authoritative registry root discovery for current active continuity truth
- closure-grade continuity decision on NumPy and legacy compat windows
- relocation-safe retained evidence pack discipline for copied remote runs
- broader multi-window and champion/challenger offline evidence that goes beyond a single controlled proof bundle

Until those are explicit, Phase 2 is still planned next work but not the main execution focus.

---

## Not started / not main focus yet

The following remain visible but are not the current main focus:

- websocket ingestion services
- online feature/state service
- replay-vs-live parity harnesses over live-style inputs
- selector runtime
- thin executor operating loop
- shadow/paper operation
- live-path observability and recovery evidence

---

## Current interpretation notes

- The retained QL-021 bundles in `outputs/` are real controlled-proof evidence for the offline hot path. They do not, by themselves, prove authoritative continuity closure or live readiness.
- Runtime and executor contracts exist as governance and artifact surfaces. They are not the same thing as a live selector daemon plus executor loop.
- Commercialization gates are defined, but no gate above the offline side is currently operationally evidenced in this repository.
