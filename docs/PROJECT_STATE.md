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

- current_phase: `Phase 5 real policy-learning implementation complete`
- current_focus: `Verification gate is now fully green; QL-009 phase-5 candidate search expansion is done`
- current_blocker: `none`
- declared_next_task: `Reassess QL-004 (Observation Schema Enforcement) or begin QL-100 (ONNX/TensorRT runtime acceleration)`
- not_now:
  - `live deployment plumbing`
  - `reward_v2 path-aware redesign`
  - `advanced selector heuristics`

## Active work item

```yaml
id: none
title: No active work item — QL-009 complete, verification gate green
status: done
```

## Current blocker details

None. All four verification gates are now clean:
- `ruff check .` → All checks passed!
- `mypy src` → Success: no issues found in 41 source files
- `pytest -q` → 80 passed
- `git diff --check` → clean

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
- QL-006 registry schema and promotion-gate enforcement completed
- registry now records auditable promotion decisions and prevents automatic score-only champion promotion
- Phase 4 paper/sim evidence operationalization completed
- paper/sim evidence is now attached as a first-class registry-linked record, and promotion decisions can link evaluation, comparison, paper/sim, and deployment evidence without ad-hoc placeholders
- QL-008 real training loop implementation completed
- the active training path now fits learned policy weights, records search-budget metadata, and selects checkpoints on validation without touching the final untouched test
- QL-009 candidate search expansion and repo-wide verification gate cleanup completed
- explicit candidate_search configs now produce multiple learned candidates; repo-wide ruff/mypy/pytest/diff-check verification gate is now fully clean

## Immediate next actions

1. Reassess QL-004 (Observation Schema Enforcement) for concrete gaps beyond the completed audit.
2. If no QL-004 gap emerges, begin QL-100 (ONNX/TensorRT inference acceleration exploration).

## Update rule

After every meaningful task:
- update `current_focus` if it changed
- update `declared_next_task`
- update `Recently completed`
- update the active work item or replace it
