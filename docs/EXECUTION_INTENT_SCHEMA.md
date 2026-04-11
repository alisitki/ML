# Execution Intent Schema

## Purpose

This document defines the canonical output contract from runtime selector / inference to executor.

This is the hard seam between:
- policy selection logic
- execution logic

The executor should consume execution intent, not raw policy artifacts.

## 1. Core principle

Execution intent is the narrowed, final, executor-facing decision.

It must be:
- explicit
- unambiguous
- traceable
- feasible-checkable

## 2. Mandatory fields

Every execution intent must include:

- `intent_id`
- `policy_id`
- `artifact_id`
- `decision_timestamp`
- `target_asset`
- `venue`
- `action`
- `notional_or_size`
- `leverage`
- `ttl_seconds`
- `confidence_or_score`
- `selector_trace_id`

## 3. Allowed action set

At minimum:
- `abstain`
- `enter_long`
- `enter_short`
- `hold`
- `exit`

## 4. Feasibility relation

Execution intent is not a guarantee of executability.

Executor must still check:
- balance
- min order size
- venue status
- leverage bounds
- operational constraints

But executor must not reinterpret strategy logic.

## 5. No-trade intent

`abstain` is a first-class execution intent.

No-trade is explicit, not implicit absence of instruction.

## 6. TTL semantics

Every executable non-abstain intent must include `ttl_seconds`.

If TTL expires before execution, executor may reject or discard the intent according to execution policy.

## 7. Traceability rule

Every execution intent must be traceable back to:
- registered policy
- inference artifact
- runtime selector decision context

## 8. Prohibited ambiguity

Forbidden:
- venue-less executable intent
- asset-less executable intent
- size-less directional intent
- artifact-untraceable intent
- executor-side inference of missing strategic fields
