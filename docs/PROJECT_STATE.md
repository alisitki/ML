# Project State

## Purpose

This is the short, current truth file for QuantLab.

Use it to answer:

- what the project ultimately aims to become
- what the repository materially implements today
- what is missing today
- what is being hardened now
- what should be built next

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

QuantLab is in late Phase 1 hardening: the canonical and offline foundation is materially implemented and is the strongest part of the repository.

The repo is not yet a full live-operating trading system. Phase 2 live-path buildout is the next major phase, not current implemented reality.

---

## Current implemented strengths

- canonical exchange-aware market-data and observation semantics
- offline trajectory building
- walk-forward training and evaluation discipline
- artifact export and registry discipline
- runtime-facing contracts and thin-executor boundary definitions
- governance, runbook, and repo-memory discipline

---

## Current missing layers

- production websocket ingestion across the active venue scope
- online state / feature service
- replay-vs-live parity tooling over live-style inputs
- selector runtime daemon
- thin executor integration and live control loop
- shadow/paper operating loop
- system-generated commercialization evidence

---

## Active focus

- close QL-016 by auditing authoritative registries and retiring or freezing temporary NumPy continuity paths when safe
- keep current-phase truth surfaces honest so later live-path phases are not described as already implemented
- keep promotion and live-capital claims out of scope until live-path evidence exists

---

## Next phase after active focus

Phase 2 - runtime/live parity foundation:

- websocket ingestion for the declared market scope
- online state / feature service with explicit unsupported/missing/stale semantics
- replay-vs-live parity and recovery tooling
- degraded-input and stale-state behavior
- selector runtime
- shadow/paper loop
- thin executor integration and live controls

---

## Current interpretation notes

- QL-021 and related remote-GPU work prove the offline hot path on controlled proof runs. They do not prove live-path operation.
- Runtime and executor contracts exist as governance and artifact surfaces. They are not the same thing as a live selector daemon plus executor loop.
- Commercialization gates are defined, but they are not yet fully evidenced beyond the offline side of the system.
