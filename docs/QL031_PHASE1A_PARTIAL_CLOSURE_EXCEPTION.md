---
status: canonical
owner: quantlab
last_reviewed: 2026-04-20
read_when:
  - before_phase1a_parallel_v2_execution
supersedes: []
superseded_by: []
---

# QL-031 Phase 1A Partial-Closure Exception

## Purpose

This memo defines the narrow exception that allows `QL-031 Phase 1A` to start before broader offline closure reaches `PASS`.

It does not reopen the retained-bundle integrity track.
It does not authorize `Phase 1B`, `Phase 2`, event-native work, promotion, or live/runtime interpretation.

## Decision

- `YES`: allow a partial-closure exception for `Parallel V2 Phase 1A` only.
- Scope stays limited to semantics/objective redesign on the current flat observation surface.
- `linear-policy-v1` remains the intact baseline and rollback anchor.

## Exact boundary

Allowed:

- `V2` action vocabulary with explicit `abstain` / `hold` / `exit` semantics
- flat-only `policy_state` consumption
- joint action-mask semantics
- `H=4` state-aware bootstrap oracle with a 4-row local horizon (`t..t+3`)
- auxiliary value head for the oracle return
- current observation surface retained
- current line of work remains `Parallel V2`

Not allowed:

- structured encoder work
- `Phase 1B`
- `Phase 2`
- event-native redesign
- promotion-rule changes
- same-root untouched-final discipline changes
- executor/live enablement for `linear-policy-v2`

## Residual unknowns accepted

- broader multi-window / multi-slice empirical closure remains `PARTIAL`
- the same-root blocker remains open because the retained `2026-01-25` surface produced no promotable champion
- same-root comparison-report linkage remains blocked by `no_current_champion`
- live replay/runtime parity is still out of scope for this exception

## Evidence interpretation

The slim retained blocker bundle at `outputs/ql031-same-root-proof-20260419` is a blocker record only.

It is not a runnable training/evaluation payload source for `Phase 1A`.

`Phase 1A` evidence must come only from:

- a `fresh full same-root run`, or
- a `payload-complete same-root run`

## V1 / V2 compatibility boundary

- `PolicyRuntimeBridge.decide()` remains backward-compatible as `decide(artifact, observation, policy_state=None, ...)`
- `linear-policy-v1` continues to accept `policy_state=None`
- shared reward / evaluation / runtime semantics are gated by `action_space_version`
- config-level V2 base actions stay generic: `abstain`, `hold`, `exit`, `enter_long`, `enter_short`
- internal V2 training/runtime vocabulary is the 9-logit venue-expanded joint action vocabulary

## Stop / rollback conditions

Stop if any of the following becomes necessary:

- changing the current observation surface
- changing `tensor_cache_v1`
- changing split discipline or promotion rules
- letting V2 semantics leak into V1 behavior
- interpreting `linear-policy-v2` as executor/live deployable
- treating the slim retained blocker bundle as a runnable `Phase 1A` payload

## Phase 1B preconditions

Do not plan `Phase 1B` until all of the following exist:

- at least one `fresh full same-root run` or `payload-complete same-root run`
- training summary, search budget, validation summary, and untouched-final evaluation report
- V1 vs Phase 1A same-surface ablation
- oracle diagnostics: label coverage, masked-out rows, joint-action histogram, hold/exit usage, flip and venue-switch diagnostics
- explicit evidence on whether the remaining blocker is encoder expressivity or still semantics/objective economics
