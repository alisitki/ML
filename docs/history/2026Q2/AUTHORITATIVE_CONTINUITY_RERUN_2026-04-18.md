# Authoritative Continuity Rerun 2026-04-18

**Status:** complete  
**Task:** `QL-016` / `QL-004` fresh authoritative controlled rerun

## External run root

- run root: `/workspace/runs/ql016-ql004-authoritative-20260418`
- registry root: `/workspace/runs/ql016-ql004-authoritative-20260418/registry`
- host class: single-GPU external Vast instance
- GPU: `NVIDIA GeForce RTX 5090`

## Controlled run scope

- data config: `configs/data/controlled-remote-day.yaml`
- training config: `configs/training/production.yaml`
- reward config: `configs/reward/default.yaml`
- evaluation config: `configs/evaluation/default.yaml`
- snapshot surface: `controlled-remote-example-20260125`

## Stage exits

- `build=0`
- `train=0`
- `evaluate=0`
- `score=0`
- `export=0`

## Controlled run outputs

- `inspect_s3.json` showed `matched_partition_count=133`
- `policy_id=policy-fd389f520ad3`
- `training_backend=pytorch`
- `training_device=cuda`
- `tensor_cache_used=true`
- `jsonl_fallback_used=false`
- `evaluation` completed on `final_untouched_test`
- `score` appended a challenger record on the external active registry scope
- `inference_artifact.json` was exported successfully

## Authority discovery result

- discovery summary: `/workspace/runs/ql016-ql004-authoritative-20260418/continuity_authority_discovery.json`
- `eligible_external_candidate_count=1`
- `authority_confirmation_candidate_path=/workspace/runs/ql016-ql004-authoritative-20260418/registry`
- decision: `authority_confirmation_step_allowed`

## Authoritative audit result

- authoritative audit output: `/workspace/runs/ql016-ql004-authoritative-20260418/continuity_audit_authoritative.json`
- `inspected_evidence_kind=authoritative_evidence`
- `authority_status=confirmed`
- `closeout_decision_allowed=true`
- `audit_scope_verdict=clear_in_inspected_scope`
- `closeout_blockers=[]`
- `active_training_backend_counts={"pytorch": 1}`
- `active_numpy_training_backend_count=0`
- `active_legacy_compat_artifact_count=0`
- `ready_to_close_numpy_continuity_window=true`
- `ready_to_retire_legacy_compat_window=true`

## Resulting closeout decisions

- `numpy_training_backend -> RETIRE`
- `legacy_linear_policy_v1_compat -> RETIRE`

## Repo consequence

- `docs/continuity_closeout/*.yaml` move from `pending_authoritative_evidence` to `decided`
- `QL-016` and `QL-004` leave the active backlog
- `QL-031` remains the single active next offline batch

## Sprint closeout sentence

`authoritative scope confirmed; NumPy and legacy compat continuity windows retired`
