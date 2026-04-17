# Project State

## Purpose

This is the live state file for QuantLab.

Use this file to answer:
- where we are now
- what is currently active
- what the next task is
- what is blocked
- what was just completed

This file must stay short and current.

---

## Current snapshot

- current_phase: `Phase 5 — QL-022 truth reconciliation landed, QL-023 registry accounting hardening completed under scoped verification, and QL-021 acceptance is now closed. D-018 and the runbook now treat average GPU utilization as diagnostic telemetry rather than a hard acceptance gate for controlled proof runs.`
- current_focus: `Resume QL-016 continuity closeout now that QL-021 operational and acceptance axes are done; keep promotion out of scope and preserve the clean remote proof bundle as the retained operational evidence.`
- current_blocker: `QL-016 still needs external \`audit-continuity\` evidence against active registries before the NumPy reference path and legacy continuity windows can move from tracking to freeze/retire action.`
- declared_next_task: `QL-016 — run \`quantlab-ml audit-continuity --registry-root ...\` against the active registries, then decide whether NumPy / legacy continuity paths can freeze or retire without breaking active inventory.`
- not_now:
  - `live deployment plumbing`
  - `cloud provisioning automation`
  - `reward_v2 path-aware redesign`
  - `advanced selector heuristics`

## Active work item

```yaml
id: ql-016-continuity-closeout
title: Close the remaining NumPy / legacy continuity inventory after QL-021 acceptance closure
status: in_progress
path_classification: temporary compatibility maintenance
completed_prerequisite:
  ql_021_operational_state: done
  ql_021_acceptance_state: done
  ql_021_promotion_state: not_started
current_blocker:
  - external `audit-continuity` evidence is not yet captured against active registries
next_action:
  - run `quantlab-ml audit-continuity --registry-root ...` and decide freeze/retire action for NumPy and legacy continuity surfaces
```

## Current state details

**QL-021 acceptance closure (2026-04-17):**
- `build/train/evaluate/score/export` all exited `0`; the hot path is real, not hypothetical.
- `train.log` shows `training_backend=pytorch`, `training_device=cuda`, `tensor_cache_used=true`, `jsonl_fallback_used=false`; `evaluate.log` shows `compiled_policy_mode=tensor_cache_linear_policy_batch`.
- The clean remote proof run meets the timing gates: train epochs `28.8-34.9 sec`, validation `7.5-18.8 sec`, final evaluate `7.7 sec`.
- D-018 and `docs/REMOTE_GPU_RUNBOOK.md` now treat average GPU utilization as diagnostic telemetry only for QL-021-style controlled proof runs. The hard acceptance truth is direct hot-path evidence: `training_device=cuda`, `tensor_cache_used=true`, `jsonl_fallback_used=false`, required chain exit codes `0`, explicit train/evaluate execution evidence, and satisfied timing gates.
- Exact active-window GPU averages remain recorded (`6.07%` across the full search window, `8.98%` across canonical refit full, `11.58%` across canonical refit epoch-only windows, `0.0%` across final evaluate), but they no longer block QL-021 acceptance closure.
- Promotion is not started: the retained candidate is still a challenger and `final_untouched_test` total net return is negative (`-0.9497811681221033`).

## Recently completed

- constitutional layer written
- canonical contract docs written
- runtime/executor boundary written
- operational repo-memory layer added
- QL-001 gap audit completed
- confirmed split_v1 and reward_v1 mismatches recorded in BACKLOG
- no confirmed observation blocker was found from current audit evidence
- QL-002 split_v1 alignment completed
- canonical train/validation/final untouched test segmentation now exists with persisted split artifacts and walk-forward folds
- QL-007 minimum turnover-event dependency was resolved inside the reward path without a broader action-space epic
- QL-003 reward_v1 parity alignment completed
- effective selected-venue semantics are now explicit in reward context during reward application and evaluation
- QL-005 policy artifact metadata and execution intent path alignment completed
- policy artifacts now carry canonical runtime metadata and compatibility tags, and runtime selector output can be materialized as explicit execution intent
- QL-006 registry schema and promotion-gate enforcement completed
- registry now records auditable promotion decisions and prevents automatic score-only champion promotion
- Phase 4 paper/sim evidence operationalization completed
- paper/sim evidence is now attached as a first-class registry-linked record, and promotion decisions can link evaluation, comparison, paper/sim, and deployment evidence without ad-hoc placeholders
- QL-008 real training loop implementation completed
- the active training path now fits learned policy weights, records search-budget metadata, and selects checkpoints on validation without touching the final untouched test
- QL-009 candidate search expansion and repo-wide verification gate cleanup completed
- explicit candidate_search configs now produce multiple learned candidates; repo-wide ruff/mypy/pytest/diff-check verification gate is now fully clean
- Remediation Batch 1 verification completed; audit findings are now classified in `docs/REMEDIATION_BATCH_1.md`
- secret hygiene guardrails now include `.env.example` plus stricter `.gitignore` coverage; current repo inspection found no `.env` path tracked in git history, but local secret rotation remains an external action if those credentials are live
- NumPy-vs-PyTorch trainer drift is now recorded explicitly in `docs/DECISIONS.md` without changing the constitutional PyTorch target
- lightweight structured logging now covers trajectory building, training search, and registry mutations
- the active shared feature extractor now includes derived-surface channels, closing the confirmed Batch 1 observation-path gap
- `MomentumBaselineTrainer` is no longer a silent alias; it remains import-compatible through a deprecated shim
- Remediation Batch 2 completed; new artifacts now embed a strict runtime contract and runtime rejects scale-spec, raw-shape, derived-contract, derived-channel, feature-dimension, and deprecated-adapter mismatches before inference
- a temporary legacy compat window now exists only for deterministic legacy `linear-policy-v1` artifacts; legacy acceptance is explicit, logged, and deprecated rather than silent
- deprecated `momentum-baseline-v1` artifacts are now quarantined from the normal runtime path and rejected with an explicit retrain/re-export message
- Batch 2 verification and patch notes are recorded in `docs/REMEDIATION_BATCH_2.md`
- interpretation precedence is now explicit in `AGENTS.md`; allowed, optional, and temporary paths are no longer left to implicit reading
- `docs/TASK_TEMPLATE.md` now requires path classification before implementation so temporary or optional paths are harder to grow accidentally
- D-011 now behaves as an active guardrail instead of a passive note: NumPy is continuity-only and not a strategic growth path
- derived surface is now documented as augmentation only; support for derived channels no longer implies a core/default requirement
- legacy compat is now documented as a retirement-tracked continuity tool, not a development magnet
- QL-010 completed: walk-forward fold consumption is now mandatory for candidate selection, purge is applied before fold training, and candidate specs are still canonically refit on train with validation-only epoch selection
- development-region trajectories are now persisted explicitly so fold validation can consume boundary-crossing development steps without weakening canonical split boundaries
- QL-011 completed: `configs/training/production.yaml` now carries the canonical `1m×8`, `5m×8`, `15m×8`, `60m×12` observation preset
- `README.md` and config-state text now say explicitly that `configs/training/default.yaml` is continuity/smoke only and is not the production profile
- QL-012/013 tracking completed: `quantlab-ml audit-continuity` now reports active training backend counts, active legacy compat dependence, and whether the temporary continuity windows are ready to close
- D-014 accepted: meaningful real training is now explicitly PyTorch-first, GPU-first when available, and remote-GPU-first rather than local-laptop-first
- constitution, runtime boundary, roadmap, and agent guidance now say that local CPU/laptop runs are smoke/debug/tiny-baseline/short-validation tools only
- `docs/TASK_TEMPLATE.md` and `readme.md` now separate local smoke/debug mode, continuity baseline mode, and real training mode so laptop convenience is harder to mistake for the strategic path
- QL-016 implementation is now active: `LinearPolicyTrainer` defaults to the PyTorch core backend while preserving `linear-policy-v1` payload/export semantics and the strict runtime contract surface
- NumPy is now a narrow reference/parity backend only; the deprecated `MomentumBaselineTrainer` shim routes explicitly through that continuity path instead of defining the core direction
- PyTorch-vs-NumPy parity coverage now exercises default, `search-small`, and `production` training profiles plus runtime decision parity without relaxing fold, purge, or final untouched test discipline
- QL-017 completed: the trainer now records the effective training device, CUDA availability, and device name in structured logs and `training_summary`, and the repo now has an official provider-agnostic remote GPU workflow centered on a controlled first run instead of laptop-scale continuity examples
- QL-018 completed: `configs/data/controlled-remote-day.yaml` now provides the official bounded full-day remote snapshot example, and `docs/REMOTE_GPU_RUNBOOK.md` defines bootstrap, preflight, command flow, expected outputs, acceptance criteria, and first-failure triage for the first controlled Vast run
- README now points explicitly to the controlled remote-day config and the official remote GPU runbook so the first real run no longer depends on inferred workflow knowledge
- QL-019 completed: streaming JSONL trajectory build/directory flow implemented; `TrajectoryBuilder.build_to_directory()` and `TrajectoryDirectoryStore` now persist metadata plus per-split JSONL records without assembling a production-size `TrajectoryBundle`
- `TrajectoryDirectoryStore` writes manifest.json + per-split JSONL files; explicit line-size guard (warn 512 MB / fail 2 GB) makes oversized records immediately visible
- `TrajectoryBundle` in-memory path is now explicitly marked FIXTURE / TEST COMPAT ONLY at module, class, and method level with a grep-able `_FIXTURE_TEST_COMPAT_ONLY` sentinel
- QL-020 completed: the active prod trainer/evaluate path now uses fold-aware streaming batch training, train-only two-pass streaming normalization, and true streaming validation/evaluation; prod no longer assembles giant dense train/validation matrices and the matrix-first path lives only in explicit compat/test helpers
- prod streaming logs and `training_summary` now expose `effective_batch_size`, `estimated_batch_bytes`, `batches_per_epoch`, `batch_target_bytes`, and `proxy_validation_used=false` so remote OOM/throughput triage is not blind
- CLI `build-trajectories` now calls streaming path by default; `train` and `evaluate` auto-detect directory vs legacy JSON file, and directory `evaluate` now streams `final_untouched_test` instead of materializing the split list
- 54 targeted training/evaluation/CLI/reward tests now pass for the streaming-batch refactor; `.venv/bin/ruff check .` and `.venv/bin/mypy src` are clean
- QL-021 local implementation completed: `build-trajectories` now writes canonical JSONL plus `tensor_cache_v1/` shard sidecars for `development`, `train`, `validation`, and `final_untouched_test`
- prod directory train now uses tensor-cache shards for train-only feature stats, vectorized batch assembly, and train-time validation; JSONL fallback is explicit temporary compatibility only
- prod directory evaluate now uses the same tensor-cache family and a shared batched linear-policy evaluator, so final evaluate no longer reparses policy payloads or rebuilds features step-by-step
- 44 targeted tensor-cache trajectory/evaluation/CLI tests passed locally, covering cache output, prod fast-path guardrails, and explicit JSONL fallback behavior
- QL-022 completed: repo-memory now distinguishes QL-021 `operational_state`, `acceptance_state`, and `promotion_state` instead of collapsing them into one status, and the 2026-04-17 retained rerun is recorded as operationally done
- proof-gated reconciliation found no reopen trigger for QL-010 / QL-011 / QL-012 / QL-013 from current repo truth; they remain append-only history rather than being reopened by stale summary text alone
- `outputs/ql021-controlled-remote-rerun-20260417-build-fresh/acceptance_evidence.json` now acts only as a derived retained-evidence index over existing logs, manifests, artifacts, and GPU CSV samples
- QL-021 retained-evidence reconciliation completed: the retained 2026-04-17 logs and GPU CSVs are sufficient to prove operational success and timing gates, and the clean remote proof run later confirmed those same hot-path signals on real remote GPU infrastructure
- QL-023 completed under scoped verification: manifest-based registry registration now derives dataset-surface train coverage from retained split facts, falls back to retained tensor-cache evidence only when that recovery surface is actually available, and otherwise fails loudly instead of warning and emitting zero train coverage
- clean remote QL-021 proof capture completed on a `500 GB` / `RTX 4090` / `Threadripper PRO 7995WX` host: `build/train/evaluate/score/export` again exited `0`, the tensor-cache hot path stayed active, and timing gates remained within the runbook limits
- `outputs/ql021-acceptance-proof-20260417-no-trpro7995wx/acceptance_evidence.json` now acts only as an optional derived retained-evidence index over the new proof bundle, and the reduced local bundle intentionally keeps manifests, registry JSONs, logs, and GPU CSVs while omitting the full `112G` trajectory payload
- D-018 accepted: QL-021-style controlled proof runs now use direct hot-path evidence as the hard acceptance signal, while average GPU utilization remains diagnostic telemetry only
- QL-021 acceptance completed: `operational_state=done`, `acceptance_state=done`, and `promotion_state=not_started`

## Immediate next actions

1. **QL-016 continuity closeout** — run `quantlab-ml audit-continuity --registry-root ...` against the active registries now that QL-021 acceptance is closed.
2. Freeze or retire the NumPy reference path and any still-unused legacy continuity surfaces once the external audit confirms zero active dependency.
3. Preserve the reduced local QL-021 proof bundle and derived acceptance index as retained operational evidence; promotion remains out of scope.
4. Keep QL-014 later and keep QL-100 / QL-101 parked.

## Update rule

After every meaningful task:
- update `current_focus` if it changed
- update `declared_next_task`
- update `Recently completed`
- update the active work item or replace it
