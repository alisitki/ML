---
status: canonical
owner: quantlab
last_reviewed: 2026-04-18
read_when:
  - before_non_trivial_code_changes
  - before_reprioritization
supersedes: []
superseded_by: []
---

# Product Thesis

## Business destination

QuantLab's destination is an end-to-end multi-exchange futures ML trading system that turns high-volume websocket market data into post-cost-positive, controllable, and traceable live trading decisions.

The target system:

1. ingests websocket market data
2. normalizes exchange-aware canonical events and state
3. trains and evaluates policies offline
4. runs runtime inference
5. hands controlled intent to a thin executor
6. advances through commercialization gates toward live capital deployment

The primary commercial model remains:

> internal proprietary live trading driven by exchange-aware ML policies

---

## Current repo scope

The repository materially implements the foundation that makes that destination credible:

- canonical market-data, observation, reward, split, artifact, registry, and execution-intent contracts
- offline trajectory building
- offline training and evaluation
- artifact export and registry discipline
- runtime and executor boundary definitions
- governance, runbooks, and repo-memory discipline

The repository does not yet materially implement the full live-operating half:

- production websocket ingestion services
- long-running online state / feature services
- replay-vs-live parity proof on live feeds
- a selector runtime daemon
- thin executor integration
- a shadow/paper operating loop
- system-generated commercialization evidence

QuantLab should therefore be described as a phase-aware trading-system foundation with real offline substance, not as a completed live trading stack and not as a throwaway offline scaffold.

---

## Why the current phase matters commercially

The current phase is commercially valid because later live deployment is only trustworthy if the upstream system is disciplined first.

Current work matters when it strengthens:

1. canonical semantics and venue-aware truth
2. leakage-safe offline evaluation
3. artifact lineage and reproducibility
4. runtime boundary clarity
5. future offline/online parity

Missing later-phase capabilities are not defects by default. They become defects when the repo claims them as current implemented reality or when current-phase work blocks the next phase.

---

## Product truth

In this system, model quality alone is not enough.

Real commercial trust requires:

- trustworthy canonical data semantics
- correct feature meaning
- valid evaluation
- explicit runtime boundaries
- safe execution handoff
- reconstructable evidence

If any of these fail, live deployment confidence is fake.

---

## Non-negotiable trade-offs

Never trade away:

- leakage discipline
- venue identity where it matters economically
- parity between offline and live meaning
- explicit degraded-state handling
- thin executor boundary
- traceability of decisions
- promotion discipline

---

## Decision rule

If a proposed task does not plausibly improve edge, parity, safety, capital protection, research throughput, or current-phase clarity, it is probably not core work now.
