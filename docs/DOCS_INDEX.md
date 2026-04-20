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

## Start here

- Repository entry surface: `README.md`
- Agent operating rules: `AGENTS.md`
- Current repo truth: `docs/PROJECT_STATE.md`
- Phase order and gates: `docs/ROADMAP.md`
- Active queue: `docs/BACKLOG.md`
- Long-term intent: `docs/PRODUCT_THESIS.md`
- Governance-sensitive agent path: `.agents/skills/quantlab-governance/SKILL.md`

---

## Authority map

- current implemented repo reality -> `docs/PROJECT_STATE.md`
- current phase order and gates -> `docs/ROADMAP.md`
- active execution queue / outstanding work -> `docs/BACKLOG.md`
- current machine-readable continuity closeout state -> `docs/continuity_closeout/*.yaml`
- offline closure definition -> `docs/OFFLINE_CLOSURE_CRITERIA.md`
- continuity audit method -> `docs/CONTINUITY_AUDIT_RUNBOOK.md`
- continuity authority discovery method -> `docs/CONTINUITY_AUTHORITY_DISCOVERY_RUNBOOK.md`
- continuity closeout record format -> `docs/CONTINUITY_CLOSEOUT_RECORDS.md`
- target business destination -> `docs/PRODUCT_THESIS.md`
- target live-path architecture -> `docs/ONLINE_RUNTIME_MODEL.md`
- target commercialization gates -> `docs/COMMERCIALIZATION_GATES.md`
- executor / runtime boundary -> `docs/RUNTIME_BOUNDARY.md`

Current implemented reality is governed by `docs/PROJECT_STATE.md`, `docs/ROADMAP.md`, and `docs/BACKLOG.md`.
Current continuity retirement state is governed by `docs/continuity_closeout/*.yaml` interpreted through `docs/CONTINUITY_CLOSEOUT_RECORDS.md`.
`docs/PRODUCT_THESIS.md`, `docs/ONLINE_RUNTIME_MODEL.md`, and `docs/COMMERCIALIZATION_GATES.md` describe destination or later-phase architecture and do not override current implemented reality without explicit supersession.
Historical docs under `docs/history/` preserve chronology and evidence only; they do not override newer current-head truth or the closeout YAML surface.
If two docs seem to disagree, the more specific authority above wins unless a newer canonical doc explicitly supersedes it.

Current active-track note:

- `QL-014` is the active docs-truth item.
- `QL-031` is the single active next offline-hardening batch.
- Docs hardening should clarify that track, not compete with it.

---

## Reading order

1. `README.md`
2. `AGENTS.md`
3. `docs/PROJECT_STATE.md`
4. `docs/ROADMAP.md`
5. `docs/BACKLOG.md`
6. `docs/MARKET_SCOPE.md`
7. `docs/PRODUCT_THESIS.md`
8. `docs/ONLINE_RUNTIME_MODEL.md`
9. `docs/COMMERCIALIZATION_GATES.md`
10. `docs/QUANTLAB_CONSTITUTION.md`
11. `docs/RUNTIME_BOUNDARY.md`
12. relevant canonical contracts
13. relevant runbooks

---

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

---

## Canonical version map

- reward -> `reward_v1`
- split -> `split_v1_walkforward`
- observation schema -> `observation_schema_v1`
- action space -> `action_space_v1`
- policy artifact schema -> `policy_artifact_v2`
- legacy policy artifact schema -> `policy_artifact_v1`
- strict runtime contract -> `runtime_contract_v1`
- execution intent schema -> `execution_intent_v1`
- registry schema -> no separate version id declared on current `HEAD`

---

## Operational docs

- `docs/PROJECT_STATE.md`
- `docs/ROADMAP.md`
- `docs/BACKLOG.md`
- `docs/QL031_MODEL_REDESIGN_PLAN.md`
- `docs/OFFLINE_CLOSURE_CRITERIA.md`
- `docs/CONTINUITY_AUDIT_RUNBOOK.md`
- `docs/CONTINUITY_AUTHORITY_DISCOVERY_RUNBOOK.md`
- `docs/CONTINUITY_CLOSEOUT_RECORDS.md`
- `docs/DECISIONS.md`

---

## Runbooks

- `docs/EVALUATION_RUNBOOK.md`
- `docs/REMOTE_GPU_RUNBOOK.md`
- `docs/CONTINUITY_AUDIT_RUNBOOK.md`
- `docs/CONTINUITY_AUTHORITY_DISCOVERY_RUNBOOK.md`

---

## Historical material

Historical docs do not override canonical or operational docs.

Primary location:

```text
docs/history/2026Q2/
```
