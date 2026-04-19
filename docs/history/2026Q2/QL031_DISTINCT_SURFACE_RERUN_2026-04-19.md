# QL-031 Distinct Surface Rerun 2026-04-19

**Status:** complete  
**Task:** `QL-031` minimal distinct-surface controlled rerun fallback

## Task classification

- `task_phase=current-phase hardening`
- `layer=offline_training`
- `business_effect=continuity_debt_retirement`
- `execution_mode=real_training`
- `risk_focus=leakage,reward_drift,venue_semantic_drift`

## External run root

- run root: `/workspace/runs/ql031-controlled-remote-rerun-20260419-20260126`
- registry root: `/workspace/runs/ql031-controlled-remote-rerun-20260419-20260126/registry`
- host class: single-GPU external Vast instance
- GPU: `NVIDIA GeForce RTX 4090`

## Controlled run scope

- data config: `configs/data/ql031-controlled-remote-day-20260126.yaml`
- training config: `configs/training/production.yaml`
- reward config: `configs/reward/default.yaml`
- evaluation config: `configs/evaluation/default.yaml`
- snapshot surface: `controlled-remote-example-20260126`

## Stage exits

- `build=0`
- `train=0`
- `evaluate=0`
- `score=0`
- `export=0`

## Controlled run outputs

- `inspect_s3.json` showed `matched_partition_count=133`
- `policy_id=policy-bc4e759de1b2`
- `training_backend=pytorch`
- `training_device=cuda`
- `tensor_cache_used=true`
- `jsonl_fallback_used=false`
- `selection_fold_count=3`
- `evaluation` completed on `final_untouched_test`
- `score` appended a challenger record on the retained external registry scope
- `inference_artifact.json` was exported successfully

## Retained local bundle

- retained-local bundle path: `outputs/ql031-controlled-remote-rerun-20260419-20260126`
- retained-local manifest path: `outputs/ql031-controlled-remote-rerun-20260419-20260126/bundle_manifest.json`
- retained-local checksum path: `outputs/ql031-controlled-remote-rerun-20260419-20260126/SHA256SUMS`
- retained copy kind: `repo-local retained distinct-surface evidence bundle`
- retained bundle evidence class: `external-retained-evidence`
- retained bundle authority: `unconfirmed`
- retained bundle disk usage: `192M`
- retained bundle unique bytes: `190739975`
- source authority remains the external rerun root and its retained registry scope
- this retained copy is not relabeled as `authoritative_evidence`

Included retained surfaces:

- `inspect_s3.json`
- `policy.json`
- `evaluation.json`
- `score.json`
- `inference_artifact.json`
- `trajectories/manifest.json`
- `trajectories/tensor_cache_v1/tensor_cache_manifest.json`
- `registry/index.json`
- `registry/records/policy-bc4e759de1b2.json`
- `registry/evaluations/policy-bc4e759de1b2.json`
- `registry/scores/policy-bc4e759de1b2.json`
- `registry/artifacts/policy-bc4e759de1b2.json` as a local hardlink to `policy.json`
- `build.log`, `train.log`, `evaluate.log`, `score.log`, `export.log`
- `build.exit`, `train.exit`, `evaluate.exit`, `score.exit`, `export.exit`
- exact retained copies of `configs/data/ql031-controlled-remote-day-20260126.yaml`, `configs/training/production.yaml`, `configs/reward/default.yaml`, and `configs/evaluation/default.yaml`

Intentionally excluded heavy surfaces:

- raw market data
- `trajectories/development.jsonl` (`21G`)
- `trajectories/train.jsonl` (`17G`)
- `trajectories/validation.jsonl` (`4.1G`)
- `trajectories/final_untouched_test.jsonl` (`4.1G`)
- full `trajectories/tensor_cache_v1/` payload (`67G`)
- temporary transfer files
- duplicate artifact bytes beyond the `policy.json` hardlink alias

## QL-031 integration result

- local batch output root: `outputs/analysis/ql031-with-rerun-20260419`
- batch status: `distinct_retained_surface_found`
- discovered distinct surface:
  - `evaluation_surface_id=controlled-remote-example-20260126:split_v1_walkforward:reward_v1`
  - `slice_id=controlled-remote-example-20260126`
  - `train_window=2026-01-26T00:00:00+00:00 -> 2026-01-26T15:59:00+00:00`

## Offline evidence-pack consequence

- offline evidence pack path: `outputs/analysis/ql031-with-rerun-20260419/offline_evidence_pack.md`
- source count: `3`
- explicit evidence classes remain:
  - `outputs/ql016-ql004-authoritative-minimum-20260418/registry -> external_retained_evidence / confirmed`
  - `outputs/ql021-acceptance-proof-20260417-no-trpro7995wx/registry -> external_retained_evidence / unconfirmed`
  - `outputs/ql031-controlled-remote-rerun-20260419-20260126/registry -> external_retained_evidence / unconfirmed`
- broader evidence is no longer same-surface-only because the pack now includes one distinct retained surface on `2026-01-26`
- comparison-report linkage is still absent on this retained surface
- paper/sim linkage is still absent on this retained surface
- broader offline closure therefore remains partial and does not move to `PASS`

## Repo consequence

- `docs/PROJECT_STATE.md` and `docs/BACKLOG.md` now reflect that the retained proof set includes one distinct-surface external retained rerun
- authoritative continuity closure remains unchanged and still depends on the already-recorded authoritative rerun scope
- this rerun broadens offline evidence only; it does not claim promotion readiness, live-path readiness, or runtime parity changes

## Proof boundary

This retained distinct-surface bundle proves:

- the exact external rerun root that produced a new retained surface on `2026-01-26`
- the exact stage exits, configs, registry record, evaluation report, score report, and exported inference artifact used in that rerun
- the QL-031 batch result that confirms a distinct retained surface is now discoverable from repo-local retained evidence
- the offline evidence-pack expansion from two retained sources to three retained sources without relabeling any retained root as authoritative evidence
- the training summary that shows `training_backend=pytorch`, `training_device=cuda`, `tensor_cache_used=true`, and `jsonl_fallback_used=false`

This retained distinct-surface bundle does not prove:

- authoritative continuity closure
- champion/challenger comparison-report linkage
- paper/sim linkage
- offline closure `PASS`
- live runtime readiness, websocket correctness, or execution safety
- that the retained copy itself should be treated as a new authoritative registry root

## Sprint closeout sentence

`distinct retained surface added; broader offline evidence improved, authoritative continuity closure unchanged`
