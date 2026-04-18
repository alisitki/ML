---
status: canonical
owner: quantlab
last_reviewed: 2026-04-18
read_when:
  - before_promotion_or_live_path_changes
supersedes: []
superseded_by: []
---

# Commercialization Gates

## Purpose

This document defines the readiness gates between research progress and scaled live trading.

The gates are target-state definitions. A gate may be conceptually defined before QuantLab can operationally evidence it.

---

## Target gates vs. current evidence

Current repo truth must always separate:

- the target meaning of each gate
- the current operational evidence for that gate

| Gate | Target meaning | Current status | Missing evidence |
| --- | --- | --- | --- |
| Gate 0 - Data-plane integrity | Trustworthy declared market-data plane with explicit venue semantics, ordering, recovery, and observability | Partially evidenced on offline contract and dataset surfaces; not yet evidenced as a live websocket plane | production websocket ingestion, live ordering/recovery validation, live-path observability |
| Gate 1 - Offline validity | Walk-forward-safe offline evaluation, artifact lineage, search-budget visibility, and reproducible comparison discipline | Strongest current evidence; offline training/evaluation and registry discipline materially exist, but gate reporting is not yet a complete commercialization surface | broader repeatable evidence packs and system-generated gate reporting |
| Gate 2 - Offline/online parity validity | Runtime feature/state behavior matches replay semantics under normal and degraded inputs | Target only | online state service, replay-vs-live parity harness, degraded-input checks, recovery checks |
| Gate 3 - Shadow / paper validity | Shadow or paper loop is reconstructable, safe, and economically interpretable | Target only | selector runtime, shadow/paper loop, executor feasibility integration, live-style decision traces |
| Gate 4 - Small-capital pilot | Controls and evidence are good enough for cautious live pilot review | Target only | kill-switch proof, monitoring and alerting, pilot evidence, failure-handling runbooks |
| Gate 5 - Scale-up readiness | Pilot behavior, runtime stability, and post-cost edge justify meaningful scale consideration | Target only | sustained live evidence, incident learning, operational maturity, scale review |

No gate above Gate 1 is currently operationally evidenced in this repository.

Offline closure still matters before later gates become the main focus. The offline closure criteria in `docs/OFFLINE_CLOSURE_CRITERIA.md` must not remain materially `PARTIAL` or `FAIL` in critical areas while Gate 2+ work is described as if it can proceed unconditionally.

---

## Target gate meanings

### Gate 0 - Data-plane integrity

Required before serious model claims:

- venue parsing works
- canonicalization is stable
- unsupported/missing/stale semantics are explicit
- ordering and recovery rules are explicit
- observability is present

Passing Gate 0 means the market-data surface is trustworthy enough to build on.

### Gate 1 - Offline validity

Required before runtime or capital claims:

- walk-forward evaluation is valid
- leakage controls are intact
- reward semantics are declared
- artifact lineage is present
- search-budget visibility exists
- results are reproducible enough to compare

Passing Gate 1 means the candidate deserves serious comparison.

### Gate 2 - Offline/online parity validity

Required before live-path confidence:

- runtime feature semantics match replay semantics
- state update rules are aligned
- degraded-input behavior is explicit
- recovery and reconnect behavior are validated
- venue-specific semantics remain intact

Passing Gate 2 means live-path behavior is interpretable relative to offline evidence.

### Gate 3 - Shadow / paper validity

Required before capital pilot:

- shadow or paper evidence exists
- decision traces are reconstructable
- stale and degraded-state policies behave as intended
- execution feasibility and venue behavior are acceptable

Passing Gate 3 means the system deserves controlled real-money exposure review.

### Gate 4 - Small-capital pilot

Required before broader deployment:

- risk caps are explicit
- kill-switch behavior is proven
- monitoring and alerts are sufficient
- pilot evidence is reviewed against offline and shadow expectations
- failure-handling procedures are documented

Passing Gate 4 means the system may justify cautious scaling consideration.

### Gate 5 - Scale-up readiness

Required before meaningful production scale:

- pilot behavior remains within acceptable bounds
- operational incidents are understood
- runtime stability is demonstrated
- strategy edge survives real costs and live frictions
- scaling does not violate risk or infrastructure limits

Passing Gate 5 means the system is operationally ready to scale under reviewed controls.

---

## Mandatory interpretation rule

Never collapse these gates into one another.

In particular:

- backtest success is not parity success
- parity success is not shadow success
- shadow success is not pilot success
- pilot success is not scale readiness
- defined later gates do not waive unresolved offline closure blockers

Defined is not evidenced.

---

## Documentation rule

Any milestone claim must name the highest gate actually evidenced.

Claims without gate language are ambiguous and should not drive capital decisions.
