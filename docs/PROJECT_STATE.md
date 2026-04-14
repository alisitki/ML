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

- current_phase: `Phase 5 — second controlled remote GPU profiling run completed (2026-04-14, RTX 3090). build-trajectories exit 0. Training profiling confirmed bottleneck: Python GIL single-threaded feature extraction. Diagnosis complete, QL-021 is the next implementation task.`
- current_focus: `Implement QL-021: write pre-computed float32 tensor files at build-trajectories time; train reads binary via torch.load() — eliminates Python GIL bottleneck, enables GPU utilization.`
- current_blocker: `none blocking QL-021 implementation — diagnosis done, path is clear`
- declared_next_task: `QL-021 — implement pre-computed tensor output in build-trajectories (write per-split float32 .pt tensor files alongside JSONL), update _train_streaming_epoch to detect and load binary tensors, re-run controlled remote rerun to verify GPU utilization > 0% and epoch < 5 min`
- not_now:
  - `live deployment plumbing`
  - `cloud provisioning automation`
  - `reward_v2 path-aware redesign`
  - `advanced selector heuristics`

## Active work item

```yaml
id: ql-021-precomputed-tensor-format
title: Implement pre-computed float32 tensor output in build-trajectories; update train to read binary tensors
status: ready_to_implement
evidence:
  profiling_run_date: 2026-04-14
  instance: RTX 3090 / 96 GB RAM / 32 vCPU
  fold1_epoch_wall_time: 40 min 10 sec
  batch_wall_time: 8.03 sec/batch
  step_wall_time: 0.335 sec/example
  gpu_avg_utilization: 0.45%
  gpu_max_utilization: 70% (single spike at model init)
  vmstat_cpu: us=2-3% (1/32 cores), id=96-97%
  wchan: 0 (pure userspace, not IO-blocked)
  io_wait: 0%
  block_read_during_train: ~0 (JSONL in page cache)
  bottleneck_root: Python GIL single-threaded list assembly in observation_feature_vector()
  mechanism: feature_dim=680K × 6 arrays/scale × 4 scales → Python .tolist() + np.asarray() per step
  fix: pre-compute at build time → torch.load() at train time
  expected_speedup: 160-500x per epoch
```

## Current blocker details

**Root cause confirmed by measurement (2026-04-14 profiling run, RTX 3090):**
- `build-trajectories` exit 0: `total_records=220, total_steps=26360, fold_count=3` ✅
- `train` started: `training_device=cuda`, `device_name=NVIDIA GeForce RTX 3090` ✅
- `feature_dim=680,413`, `effective_batch_size=24`, `batches_per_epoch=300`
- Fold 1 epoch: **40 min 10 sec** = **8.03 sec/batch** = **0.335 sec/example**
- GPU: **0.45% average** (2020 nvidia-smi samples), 70% only at model init spike
- vmstat: `us=2-3%, id=96-97%` — 1 Python thread on 1 of 32 CPUs
- wchan=0 — pure userspace CPU, not blocked in kernel
- bi≈0, wa=0% — JSONL files in 73 GB page cache, IO not the cause
- Bottleneck: `observation_feature_vector()` inside `_train_streaming_epoch`, Python GIL list ops

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

## Immediate next actions

1. **QL-021** — implement pre-computed tensor output in `build-trajectories`: write `train_tensors.pt`, `validation_tensors.pt`, `test_tensors.pt` (float32, shape=[N, feature_dim]) alongside JSONL during `build_to_directory`.
2. Update `StreamingBatchLoader` (or batch plan) to detect and use binary tensor files, falling back to JSONL-assembly only when tensor files are absent.
3. Add a test that verifies the tensor output path produces arrays with correct shape and dtype without Pydantic deserialization overhead.
4. Re-run controlled remote GPU rerun after fix: confirm GPU utilization > 0%, wall-time per epoch < 5 min, 8 epochs complete end-to-end.
5. Run `quantlab-ml audit-continuity` as a parallel operational follow-up after the run succeeds.
6. Freeze or retire the NumPy reference path once external audit confirms zero active dependency.

## Update rule

After every meaningful task:
- update `current_focus` if it changed
- update `declared_next_task`
- update `Recently completed`
- update the active work item or replace it
