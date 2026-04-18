---
status: canonical
owner: quantlab
last_reviewed: 2026-04-18
read_when:
  - before_non_trivial_code_changes
supersedes: []
superseded_by: []
---

# Docs Index

## Classes

### Canonical
Defines enduring rules and interfaces. If code conflicts with these, code is wrong.

### Operational
Defines current state and near-term priority.

### Runbook
Defines executable procedures.

### Template
Defines required structure for agent tasks and reports.

### Historical
Records audits, remediations, and past investigations. Historical docs do not override canonical or operational docs.

## Reading order

1. `docs/PRODUCT_THESIS.md`
2. `docs/MARKET_SCOPE.md`
3. `docs/ONLINE_RUNTIME_MODEL.md`
4. `docs/COMMERCIALIZATION_GATES.md`
5. `docs/QUANTLAB_CONSTITUTION.md`
6. `docs/RUNTIME_BOUNDARY.md`
7. `docs/PROJECT_STATE.md`
8. `docs/ROADMAP.md`
9. `docs/BACKLOG.md`
10. `docs/DECISIONS.md`
11. relevant canonical contracts
12. relevant runbooks

## Canonical docs

- `docs/PRODUCT_THESIS.md`
- `docs/MARKET_SCOPE.md`
- `docs/ONLINE_RUNTIME_MODEL.md`
- `docs/COMMERCIALIZATION_GATES.md`
- `docs/QUANTLAB_CONSTITUTION.md`
- `docs/RUNTIME_BOUNDARY.md`
- `docs/CANONICAL_MARKET_DATA_CONTRACT.md`
- `docs/OBSERVATION_SCHEMA.md`
- `docs/ACTION_SPACE.md`
- `docs/REWARD_SPEC.md`
- `docs/REWARD_SPEC_V1.md`
- `docs/SPLIT_POLICY.md`
- `docs/SPLIT_POLICY_V1.md`
- `docs/POLICY_ARTIFACT_SCHEMA.md`
- `docs/REGISTRY_SCHEMA.md`
- `docs/EXECUTION_INTENT_SCHEMA.md`
- `docs/PROMOTION_GATE.md`

## Runbooks

- `docs/EVALUATION_RUNBOOK.md`
- `docs/REMOTE_GPU_RUNBOOK.md`

## Operational docs

- `docs/PROJECT_STATE.md`
- `docs/ROADMAP.md`
- `docs/BACKLOG.md`
- `docs/DECISIONS.md`

## Templates

- `docs/TASK_TEMPLATE.md`
- `docs/REPORT_TEMPLATE.md`

## Historical docs

Move audits and remediation notes under:

```text
docs/history/2026Q2/
```

Recommended historical files:

- `PROD_HIDDEN_BLOCKER_AUDIT.md`
- `REMEDIATION_BATCH_1.md`
- `REMEDIATION_BATCH_2.md`
