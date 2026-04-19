# Task Template

## 1. Task classification

Use these exact machine keys in this order:
- `task_phase`: choose one primary
- `layer`: choose one primary
- `business_effect`: choose one primary
- `execution_mode`: choose one
- `risk_focus`: choose all relevant

### `task_phase`
Choose one primary:
- `target-state clarification`
- `current-phase hardening`
- `next-phase enablement`
- `optional research`
- `non-priority work`

### `layer`
Choose one primary:
- `data_plane`
- `canonicalization`
- `online_feature_state`
- `offline_training`
- `evaluation`
- `runtime_inference`
- `executor_risk`
- `observability_recovery`
- `docs_governance`

### `business_effect`
Choose one primary:
- `expected_edge`
- `parity_integrity`
- `capital_protection`
- `latency_freshness_safety`
- `research_throughput`
- `continuity_debt_retirement`
- `docs_hygiene_only`

### `execution_mode`
Choose one:
- `smoke_debug`
- `continuity_baseline`
- `shadow_paper`
- `real_training`
- `live_path_change`

### `risk_focus`
Choose all relevant:
- `unsupported_stream_misuse`
- `missing_vs_stale_confusion`
- `replay_mismatch`
- `leakage`
- `reward_drift`
- `runtime_feature_drift`
- `venue_semantic_drift`
- `execution_drift`
- `recovery_corruption`

## 2. Task restatement

Restate the task in one paragraph.

## 3. Why now

State:
- what failure mode it reduces,
- what business value it improves,
- why it should happen now.

## 4. Governing docs

List the relevant:
- strategy docs
- constitution / runtime boundary
- contracts
- runbooks
- state / backlog / decisions

## 5. Current-truth and track interpretation

State explicitly:
- which current-truth docs govern this task now
- whether the task is aligned with current repo truth or only target-state intent
- whether any missing capability is a current defect or later-phase work
- whether the task affects `QL-014`, `QL-031`, or neither
- if `QL-014` / `QL-031` are relevant, confirm this task does not invent a competing active workstream

## 6. Impact statements

State explicitly:
- offline/online parity impact
- live-path safety impact
- venue-semantics impact
- whether promotion or live-readiness interpretation changes

## 7. Main risks

State selected `risk_focus` values first, then add brief task-specific notes if needed.
Canonical `risk_focus` values:
- `unsupported_stream_misuse`
- `missing_vs_stale_confusion`
- `replay_mismatch`
- `leakage`
- `reward_drift`
- `runtime_feature_drift`
- `venue_semantic_drift`
- `execution_drift`
- `recovery_corruption`

## 8. Smallest valid plan

Give the smallest plan that solves the task without widening scope.

## 9. Files likely touched

List the likely files.

## 10. Definition of done

State:
- tests required
- docs required
- state updates required
- evidence scope
- remaining unverified assumptions

## 11. Explicit non-goals

State what this task must not turn into.

## 12. Next recommended task

State the next task that should follow and why it comes next without overriding the current active track declared by `docs/BACKLOG.md`.
