---
status: canonical
owner: quantlab
last_reviewed: 2026-04-17
read_when:
  - before_non_trivial_code_changes
  - before_reprioritization
supersedes: []
superseded_by: []
---

# Product Thesis

## Purpose

QuantLab exists to turn high-volume multi-exchange futures market data into live-deployable ML trading decisions.

The business objective is not to maximize experiment count.  
The business objective is to produce post-cost-positive, controllable, and traceable live trading behavior.

---

## What QuantLab is

QuantLab is:

- a multi-exchange futures market-data system,
- an offline training and evaluation system,
- an online inference and decision system,
- a controlled live-execution handoff system.

It is a full research-to-live pipeline, not just a notebook or model zoo.

---

## What QuantLab is not

QuantLab is not:

- a generic backtester,
- a retail signal service,
- a purely offline research sandbox,
- a live executor with hidden strategy logic,
- a compatibility museum for every past path.

---

## Primary commercial model

The primary commercial model is:

> internal proprietary live trading driven by exchange-aware ML policies

This means the highest priorities are:

- economically meaningful edge,
- offline/online parity,
- runtime safety,
- capital protection,
- scalable research throughput.

---

## Value creation rule

A change creates real value only if it improves at least one of:

1. live post-cost decision quality,
2. parity between offline evidence and runtime behavior,
3. safety under stale, partial, or recovery conditions,
4. capital protection and control quality,
5. research throughput on meaningful data scale.

Changes that mainly improve convenience without helping these are lower priority.

---

## Product truth

In this system, model quality alone is not enough.

Real edge requires:

- trustworthy canonical data,
- correct feature semantics,
- valid evaluation,
- live-path parity,
- safe execution,
- reconstructable operational traces.

If any of these fail, commercial trust fails.

---

## Non-negotiable trade-offs

Never trade away:

- leakage discipline,
- venue identity where it matters economically,
- parity between offline and live meaning,
- explicit degraded-state handling,
- thin executor boundary,
- traceability of live decisions,
- promotion discipline.

---

## Decision rule

If a proposed task does not plausibly improve edge, parity, safety, capital protection, or throughput, it is probably not core work now.
