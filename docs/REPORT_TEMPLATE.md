# Report Template

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

## 2. Summary of change

Describe what changed.

## 3. Why it changed

Explain the primary business or safety effect:
- expected edge
- parity integrity
- capital protection
- latency/freshness safety
- research throughput
- docs hygiene only

## 4. Rule or contract served

State the governing docs served by the change.

## 5. Current-truth and track interpretation

State explicitly:
- which current-truth docs governed the work
- whether any missing capability remains a current defect or later-phase work
- whether this work touched `QL-014`, `QL-031`, or neither
- if relevant, confirm the change did not create a competing active workstream

## 6. Files changed

List files changed.

## 7. Tests and verification

List:
- commands run
- important results
- what was not tested
- whether evidence is `smoke_debug`, `continuity_baseline`, `shadow_paper`, `real_training`, or `live_path_change`

## 8. Impact statements

State explicitly:
- offline/online parity impact
- live-path safety impact
- venue-semantics impact
- whether promotion or live-readiness interpretation changed

## 9. Remaining risks

State:
- blockers
- non-blockers
- unverified assumptions
- follow-up risks

## 10. Docs and state updates

State whether the following were updated:
- `PROJECT_STATE.md`
- `BACKLOG.md`
- `DECISIONS.md`
- canonical docs
- runbooks

## 11. Commercial / live-trading interpretation

State plainly:
- whether this improves edge discovery,
- whether this improves offline/online parity,
- whether this reduces live-trading risk,
- whether promotion or live readiness changed.

## 12. Explicit non-goals

State what this change did not attempt to solve.

## 13. Next recommended task

Propose the next task and why it should come next.
