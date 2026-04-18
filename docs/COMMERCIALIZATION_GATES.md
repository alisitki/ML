---
status: canonical
owner: quantlab
last_reviewed: 2026-04-17
read_when:
  - before_promotion_or_live_path_changes
supersedes: []
superseded_by: []
---

# Commercialization Gates

## Purpose

This document defines the readiness gates between research progress and scaled live trading.

It exists to prevent the common failure where a promising model is mistaken for a commercially ready system.

---

## Gate 0 — Data-plane integrity

Required before serious model claims:

- venue parsing works,
- canonicalization is stable,
- unsupported/missing/stale semantics are explicit,
- ordering and recovery rules are explicit,
- observability is present.

Passing Gate 0 means:
- the market-data surface is trustworthy enough to build on.

It does not mean:
- the model has edge.

---

## Gate 1 — Offline validity

Required before runtime or capital claims:

- walk-forward evaluation is valid,
- leakage controls are intact,
- reward semantics are declared,
- artifact lineage is present,
- search-budget visibility exists,
- results are reproducible enough to compare.

Passing Gate 1 means:
- the candidate deserves serious comparison.

It does not mean:
- the candidate is ready for live trading.

---

## Gate 2 — Offline/online parity validity

Required before live-path confidence:

- runtime feature semantics match replay semantics,
- state update rules are aligned,
- degraded-input behavior is explicit,
- recovery and reconnect behavior are validated,
- venue-specific semantics remain intact.

Passing Gate 2 means:
- live-path behavior is interpretable relative to offline evidence.

It does not mean:
- the candidate is ready for production scale.

---

## Gate 3 — Shadow / paper validity

Required before capital pilot:

- shadow or paper evidence exists,
- decision traces are reconstructable,
- stale and degraded-state policies behave as intended,
- execution feasibility and venue behavior are acceptable.

Passing Gate 3 means:
- the system deserves controlled real-money exposure review.

It does not mean:
- scaling is justified.

---

## Gate 4 — Small-capital pilot

Required before broader deployment:

- risk caps are explicit,
- kill-switch behavior is proven,
- monitoring and alerts are sufficient,
- pilot evidence is reviewed against offline and shadow expectations,
- failure-handling procedures are documented.

Passing Gate 4 means:
- the system may justify cautious scaling consideration.

---

## Gate 5 — Scale-up readiness

Required before meaningful production scale:

- pilot behavior remains within acceptable bounds,
- operational incidents are understood,
- runtime stability is demonstrated,
- strategy edge survives real costs and live frictions,
- scaling does not violate risk or infrastructure limits.

Passing Gate 5 means:
- the system is operationally ready to scale under reviewed controls.

---

## Mandatory interpretation rule

Never collapse these gates into one another.

In particular:

- backtest success is not parity success,
- parity success is not shadow success,
- shadow success is not pilot success,
- pilot success is not scale readiness.

---

## Documentation rule

Any milestone claim must name the highest gate actually passed.

Claims without gate language are considered ambiguous and should not drive capital decisions.
