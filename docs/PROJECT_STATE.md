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

- current_phase: `Phase 0 -> Phase 1 handoff`
- current_focus: `Freeze governance/spec memory, then align code to observation/reward/split v1 contracts`
- current_blocker: `None`
- declared_next_task: `Run a gap audit between current implementation and OBSERVATION_SCHEMA + REWARD_SPEC_V1 + SPLIT_POLICY_V1, then open concrete backlog items`
- not_now:
  - `TensorRT optimization`
  - `live deployment plumbing`
  - `reward_v2 path-aware redesign`
  - `advanced selector heuristics`

## Active work item

```yaml
id: QL-001
title: Gap audit for v1 implementation alignment
status: in_progress
owner: codex
depends_on:
  - governance docs frozen
done_when:
  - current code is compared against OBSERVATION_SCHEMA
  - current code is compared against REWARD_SPEC_V1
  - current code is compared against SPLIT_POLICY_V1
  - concrete mismatches are written into BACKLOG.md
  - PROJECT_STATE.md is updated with the next recommended task
```

## Current blocker details

None.

## Recently completed

- constitutional layer written
- canonical contract docs written
- runtime/executor boundary written
- operational repo-memory layer added

## Immediate next actions

1. Run v1 implementation gap audit.
2. Convert findings into concrete backlog items.
3. Start with the highest-priority blocker from the audit.

## Update rule

After every meaningful task:
- update `current_focus` if it changed
- update `declared_next_task`
- update `Recently completed`
- update the active work item or replace it
