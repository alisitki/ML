# Project State

## Purpose

This is the live state file for QuantLab.

Use this file to answer:
- where we are now
- what is currently active
- what the next task is
- what is blocked
- what was just completed

This file must stay short and current.

---

## Current snapshot

- current_phase: `Phase 2 registry and promotion discipline`
- current_focus: `QL-005 artifact metadata and execution intent path are complete; enforce registry schema and promotion-gate behavior next`
- current_blocker: `None`
- declared_next_task: `Implement QL-006 to enforce registry schema, score history, search-budget fields, and champion/challenger constraints`
- not_now:
  - `TensorRT optimization`
  - `live deployment plumbing`
  - `reward_v2 path-aware redesign`
  - `advanced selector heuristics`

## Active work item

```yaml
id: QL-006
title: Registry schema and promotion-gate enforcement
status: in_progress
owner: codex
depends_on:
  - QL-005
done_when:
  - unscored champion impossible
  - registry fields match REGISTRY_SCHEMA
  - promotion prerequisites are checkable
```

## Current blocker details

None.

## Recently completed

- constitutional layer written
- canonical contract docs written
- runtime/executor boundary written
- operational repo-memory layer added
- QL-001 gap audit completed
- confirmed split_v1 and reward_v1 mismatches recorded in BACKLOG
- no confirmed observation blocker was found from current audit evidence
- QL-002 split_v1 alignment completed
- canonical train/validation/final untouched test segmentation now exists with persisted split artifacts and walk-forward folds
- QL-007 minimum turnover-event dependency was resolved inside the reward path without a broader action-space epic
- QL-003 reward_v1 parity alignment completed
- effective selected-venue semantics are now explicit in reward context during reward application and evaluation
- QL-005 policy artifact metadata and execution intent path alignment completed
- policy artifacts now carry canonical runtime metadata and compatibility tags, and runtime selector output can be materialized as explicit execution intent

## Immediate next actions

1. Implement QL-006 to enforce registry schema and promotion-gate behavior.
2. Wire search-budget and lineage completeness through the registry path.
3. Reassess QL-004 only if a concrete observation-schema gap emerges beyond the completed audit.

## Update rule

After every meaningful task:
- update `current_focus` if it changed
- update `declared_next_task`
- update `Recently completed`
- update the active work item or replace it
