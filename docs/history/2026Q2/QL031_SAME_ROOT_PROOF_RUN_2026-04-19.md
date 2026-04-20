# QL-031 Same-Root Proof Run 2026-04-19

> Historical record only.
> This file preserves the retained same-root `QL-031` proof run evidence and blocker boundary. It does not override current repo truth by itself.
> Current truth lives in `docs/PROJECT_STATE.md`, `docs/BACKLOG.md`, `docs/OFFLINE_CLOSURE_CRITERIA.md`, and the closeout records under `docs/continuity_closeout/`.

**Status:** complete with blocker  
**Task:** `QL-031` same-root retained proof chain attempt

## Task classification

- `task_phase=current-phase hardening`
- `layer=evaluation`
- `business_effect=continuity_debt_retirement`
- `execution_mode=real_training`
- `risk_focus=leakage,reward_drift,venue_semantic_drift`

## External run root

- run root: `/workspace/runs/ql031-same-root-proof-20260419`
- registry root: `/workspace/runs/ql031-same-root-proof-20260419/registry`
- retained-local bundle path: `outputs/ql031-same-root-proof-20260419`
- retained bundle evidence class: `external-retained-evidence`
- retained bundle authority: `unconfirmed`
- host class: single-GPU external Vast instance
- GPU: `NVIDIA GeForce RTX 5090`
- vCPU: `48`
- disk: `500 GB`
- storage note: advertised `disk_bw 23552 MB/s`

## Controlled run scope

- data config: `configs/data/controlled-remote-day.yaml`
- training config: `configs/training/production-ql031-search.yaml`
- reward config: `configs/reward/default.yaml`
- evaluation config: `configs/evaluation/default.yaml`
- snapshot surface: `controlled-remote-example-20260125`

## Stage exits

- `build=0`
- `train=0`
- `evaluate=0`
- `score=0`
- `export=0`
- `post-train=2`

## Same-root run result

- `inspect_s3.json` showed `matched_partition_count=133`
- the candidate search plus final refit materialized four scored same-surface finalists in one registry root:
  - `policy-74d1bd2b2653`
  - `policy-a3c86de1a7bd`
  - `policy-4fa973ba4eac`
  - `policy-a533065715bf`
- one paper/sim evidence record was linked for the selected finalist: `paper-sim-63fd4f92f1b7`
- one promotion decision was recorded: `promotion-d0613238bccc`
- that promotion decision was `reject`
- exact failure reason: `economics.post_cost_positive`
- no current champion was created
- `compare-policies` then failed with `blocking_reasons=no_current_champion`

## Final-test economics

Final untouched test net returns for every scored finalist remained negative:

- `policy-74d1bd2b2653 -> -1.1548682071472196`
- `policy-a3c86de1a7bd -> -1.6103038742845924`
- `policy-4fa973ba4eac -> -1.0176848428327405`
- `policy-a533065715bf -> -1.404111052687772`

This run therefore did not contain a promotable same-root champion under the current promotion gate.

## Retained local bundle

- retained-local manifest path: `outputs/ql031-same-root-proof-20260419/bundle_manifest.json`
- retained-local checksum path: `outputs/ql031-same-root-proof-20260419/SHA256SUMS`
- retained bundle disk usage: `818M`
- retained bundle unique bytes: `762643873`
- the retained bundle keeps the exact failure-state artifacts, including:
  - `policy.json`
  - `policy_search.json`
  - `policy_candidates/*`
  - `evaluation.json`
  - `score.json`
  - `inference_artifact.json`
  - `champion-paper-sim.md`
  - `champion-promotion-evidence.yaml`
  - `champion-promotion-decision.json`
  - `challenger-evaluation.json`
  - `challenger-score.json`
  - `post-train.log`
  - `post-train.exit`
  - `registry/promotions/*`
  - `registry/paper_sim/*`
  - `registry/scores/*`
  - `registry/evaluations/*`

## QL-031 integration result

- local batch output root: `outputs/analysis/ql031-same-root-proof-20260419`
- batch status: `distinct_retained_surface_found`
- retained distinct surface remains:
  - `evaluation_surface_id=controlled-remote-example-20260126:split_v1_walkforward:reward_v1`
  - `slice_id=controlled-remote-example-20260126`
  - `train_window=2026-01-26T00:00:00+00:00 -> 2026-01-26T15:59:00+00:00`
- same-root blocker inventory remains blocked by:
  - `no_current_champion`
  - no registry-backed comparison reports

## Proof boundary

This retained same-root bundle proves:

- the exact same-root retained run existed and completed through training on `2026-01-25`
- the exact finalist registry state, paper/sim record, and rejected promotion decision can be audited locally
- the retained blocker is not generic missing linkage; it is specifically the absence of any post-cost-positive promotable finalist on the governing final test
- the `2026-01-26` retained rerun still exists as a distinct second surface

This retained same-root bundle does not prove:

- a valid same-root champion/challenger comparison report
- a promotable same-root champion
- offline closure `PASS`
- authoritative continuity closure
- live runtime readiness or execution safety

## Sprint closeout sentence

`same-root retained proof run completed; blocker sharpened to no promotable champion because every finalist remained post-cost negative`
