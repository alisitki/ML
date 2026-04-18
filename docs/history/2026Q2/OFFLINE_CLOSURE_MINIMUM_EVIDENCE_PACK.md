# Offline Closure Minimum Evidence Pack

**Date:** 2026-04-18  
**Purpose:** Record the minimum repo-tracked evidence package produced during the Phase 1 closeout sprint without overstating authoritative continuity truth or broader offline closure.

## 1. Sprint closeout sentence

`authoritative scope still blocked; closeout remains pending with sharper blockers`

That sentence is intentional. This pack documents real progress, but it does not claim that QuantLab is offline professionally closed or ready to treat the inspected retained bundle as authoritative in any form.

## 2. Continuity scope discovery

### Repo-tracked facts

- The default configured registry root remains `outputs/registry` via `configs/registry/default.yaml`.
- `outputs/registry` is not present on current HEAD in this workspace.
- Current HEAD therefore cannot prove the active continuity closeout scope from repo-tracked state alone.

### Inspected retained bundle

- Inspected root: `outputs/ql021-acceptance-proof-20260417-no-trpro7995wx/registry`
- Evidence class: external retained evidence
- Why it is usable for inspected-scope truth:
  - active registry record exists
  - local `registry/artifacts/` fallback is readable
  - `artifact_load_failures=[]`
  - `audit_scope_verdict=clear_in_inspected_scope`

### Continuity audit result on the inspected retained bundle

| Field | Value |
| --- | --- |
| `inspected_evidence_kind` | `external_retained_evidence` |
| `authority_status` | `unconfirmed` |
| `closeout_decision_allowed` | `false` |
| `closeout_blockers` | `["authoritative_scope_not_confirmed"]` |
| `active_record_count` | `1` |
| `readable_active_artifact_count` | `1` |
| `active_status_counts` | `{"challenger": 1}` |
| `active_training_backend_counts` | `{"pytorch": 1}` |
| `active_legacy_compat_artifact_count` | `0` |
| `registry_local_fallback_policy_ids` | `["policy-68c931ef490f"]` |
| `ready_to_close_numpy_continuity_window` | `true` |
| `ready_to_retire_legacy_compat_window` | `true` |

Interpretation:

- The inspected retained scope is clean for the audited continuity windows.
- The inspected retained scope is still not authoritative evidence.
- Without a separately confirmed external active registry root, this retained bundle must remain `external_retained_evidence` and must not be relabeled as `authoritative_evidence`.
- NumPy and legacy compat windows therefore remain repo-tracked pre-decision records.

## 3. Current-head same-surface comparison surface

Two current-head retained runs are available for a narrow same-surface comparison:

| Run | Policy ID | Evaluation Surface | Total Net Return | Composite Rank | Promotion State |
| --- | --- | --- | ---: | ---: | --- |
| `ql021-acceptance-proof-20260417-no-trpro7995wx` | `policy-68c931ef490f` | `controlled-remote-example-20260125:split_v1_walkforward:reward_v1` | `-0.9497811681221033` | `0.7990989468379202` | `not_started` |
| `ql021-controlled-remote-rerun-20260417-build-fresh` | `policy-cdc01e8eb191` | `controlled-remote-example-20260125:split_v1_walkforward:reward_v1` | `-1.138819734534162` | `0.7986698090486026` | `not_started` |

Observed deltas on the same surface:

- total net return delta: `+0.18903856641205863` in favor of `policy-68c931ef490f`
- composite rank delta: `+0.0004291377893175241` in favor of `policy-68c931ef490f`

Shared current-head facts across both runs:

- `training_snapshot_id=s3-controlled-remote-v1:controlled-remote-example-20260125`
- `reward_version=reward_v1`
- `training_backend=pytorch`
- `training_device=cuda`
- `tensor_cache_used=true`
- `jsonl_fallback_used=false`
- `selection_fold_count=3`

Limitations:

- This is a same-surface retained-run comparison, not a registry-backed champion/challenger comparison report.
- `comparison_report_id` remains missing on the retained registry surface that exists.
- `paper_sim_evidence_id` remains missing.
- This pack does not provide multi-window or multi-slice evidence.

## 4. Missing evidence after this sprint

- Fresh authoritative evidence for the active continuity closeout scope, produced by a future controlled rerun or a concrete external root already present when closure is revisited
- Authoritative rerun of `audit-continuity` on that fresh authoritative scope
- Multi-window or multi-slice offline evaluation pack
- Registry-backed champion/challenger comparison report surface
- Paper/sim linkage for any surface used to argue promotion readiness

## 5. Resulting repo-tracked decisions

- `docs/continuity_closeout/*.yaml` remain `pending_authoritative_evidence`.
- Their `latest_audit_scope_verdict` now reflects `clear_in_inspected_scope` rather than `blocked`.
- The blocker is no longer “inspection failed”; the blocker is “fresh authoritative evidence does not yet exist in this workspace baseline.”
- The historical local authority-discovery loop is closed for this workspace unless a concrete external path is already present when closure is revisited.
- `QL-031` broader offline evidence expansion is the current active repo batch.
- See `docs/history/2026Q2/CONTINUITY_AUTHORITY_DECISION.md` for the authority posture that keeps the retained bundle non-authoritative.
