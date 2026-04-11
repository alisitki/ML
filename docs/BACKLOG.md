# Backlog

## Purpose

This file tracks near-term and medium-term work items.

Status values:
- `todo`
- `in_progress`
- `blocked`
- `done`

---

## Active items

### QL-001
- title: Gap audit for v1 implementation alignment
- status: in_progress
- depends_on: governance/spec freeze
- scope: observation schema, reward_v1, split_v1
- done_when:
  - doc/code mismatches are listed
  - blockers are separated from non-blockers
  - next task is declared in PROJECT_STATE

### QL-002
- title: Implement split_v1_walkforward canonical builder behavior
- status: todo
- depends_on: QL-001
- scope: fold generation, purge width, embargo width, split artifacts
- done_when:
  - fold boundaries are persisted
  - purge/embargo widths are persisted
  - split version is stored
  - overlap case tests pass

### QL-003
- title: Enforce reward_v1 math in code
- status: todo
- depends_on: QL-001
- scope: venue-aware reward, no `venue=None` fallback, funding freshness, infeasible-action penalty
- done_when:
  - implementation matches REWARD_SPEC_V1
  - parity tests pass
  - no hidden averaging remains

### QL-004
- title: Enforce canonical observation schema in code
- status: todo
- depends_on: QL-001
- scope: axes, masks, derived surface, scale preset, causality
- done_when:
  - code matches OBSERVATION_SCHEMA
  - schema-sensitive tests pass
  - compatibility layer boundaries are explicit

### QL-005
- title: Freeze policy artifact and execution intent path
- status: todo
- depends_on: QL-001
- scope: artifact metadata, compatibility tags, execution intent contract
- done_when:
  - selector output matches EXECUTION_INTENT_SCHEMA
  - executor boundary stays thin
  - artifact compatibility enforcement exists

### QL-006
- title: Registry schema and promotion-gate enforcement
- status: todo
- depends_on: QL-005
- scope: score history, lineage, search-budget fields, champion/challenger constraints
- done_when:
  - unscored champion impossible
  - registry fields match REGISTRY_SCHEMA
  - promotion prerequisites are checkable

## Parked items

### QL-100
- title: ONNX / TensorRT runtime acceleration exploration
- status: todo
- depends_on: runtime selector maturity
- scope: inference acceleration only
- done_when:
  - runtime selector is stable
  - deployment artifact path exists

### QL-101
- title: Reward v2 path-aware redesign
- status: todo
- depends_on: reward_v1 stabilization
- scope: richer risk and carry treatment
- done_when:
  - reward_v1 is stable and versioned
