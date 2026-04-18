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
- operational record: `docs/history/2026Q2/AUTHORITATIVE_CONTINUITY_RERUN_OPERATIONS_2026-04-18.md`

## Retained minimum evidence bundle

- retained-local bundle path: `outputs/ql016-ql004-authoritative-minimum-20260418`
- retained-local manifest path: `outputs/ql016-ql004-authoritative-minimum-20260418/bundle_manifest.json`
- retained-local checksum path: `outputs/ql016-ql004-authoritative-minimum-20260418/SHA256SUMS`
- retained copy kind: `repo-local retained minimum evidence bundle`
- retained bundle disk usage: `192M`
- retained bundle unique bytes: `190676858`
- source authority remains the external rerun root and its active registry scope
- this retained copy is not relabeled as `authoritative_evidence`

Included retained surfaces:

- `continuity_audit_authoritative.json`
- `continuity_authority_discovery.json`
- `inspect_s3.json`
- `policy.json`
- `evaluation.json`
- `score.json`
- `inference_artifact.json`
- `trajectories/manifest.json`
- `trajectories/tensor_cache_v1/tensor_cache_manifest.json`
- `registry/index.json`
- `registry/records/policy-fd389f520ad3.json`
- `registry/evaluations/policy-fd389f520ad3.json`
- `registry/scores/policy-fd389f520ad3.json`
- `registry/artifacts/policy-fd389f520ad3.json` as a local hardlink to `policy.json`
- `build.log`, `train.log`, `evaluate.log`, `score.log`, `export.log`
- `build.exit`, `train.exit`, `evaluate.exit`, `score.exit`, `export.exit`
- exact retained copies of `configs/data/controlled-remote-day.yaml`, `configs/training/production.yaml`, `configs/reward/default.yaml`, and `configs/evaluation/default.yaml`

Intentionally excluded heavy surfaces:

- raw market data
- `trajectories/development.jsonl` (`21G`)
- `trajectories/train.jsonl` (`17G`)
- `trajectories/validation.jsonl` (`4.1G`)
- `trajectories/final_untouched_test.jsonl` (`4.1G`)
- full `trajectories/tensor_cache_v1/` payload (`67G`)
- temporary transfer files
- duplicate artifact bytes beyond the `policy.json` hardlink alias

## Proof boundary

This retained minimum bundle proves:

- the exact authoritative rerun root that produced the `QL-016` and `QL-004` closeout decisions
- the authority discovery result that allowed authority confirmation
- the authoritative continuity audit result that confirmed `authority_status=confirmed`
- the exact stage exits, configs, registry record, evaluation report, score report, and exported inference artifact used in that decision path
- the training summary that shows `training_backend=pytorch`, `training_device=cuda`, `tensor_cache_used=true`, and `jsonl_fallback_used=false`

This retained minimum bundle does not prove:

- raw market data fidelity
- full trajectory payload semantics
- full tensor cache contents
- broader multi-window or multi-slice offline closure
- live runtime readiness, websocket correctness, or execution safety
- that the retained copy itself should be treated as a new authoritative registry root

## Sprint closeout sentence

`authoritative scope confirmed; NumPy and legacy compat continuity windows retired`
