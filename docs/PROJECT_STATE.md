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

- current_phase: `Phase 5 core training path now consumes walk-forward folds, ships an explicit canonical production observation preset, and runs on a PyTorch-backed trainer under the existing runtime contract; external continuity audit and legacy-compat retirement are still open`
- current_focus: `The PyTorch core trainer is now active without widening the runtime payload surface, and the remaining work is external continuity audit plus the provider-agnostic remote GPU workflow/runbook for meaningful real training`
- current_blocker: `none`
- declared_next_task: `Run the continuity audit against active registries to confirm NumPy/legacy dependency counts, then define the provider-agnostic remote GPU real-training workflow and official runbook without widening runtime or continuity scope`
- not_now:
  - `live deployment plumbing`
  - `cloud provisioning automation`
  - `reward_v2 path-aware redesign`
  - `advanced selector heuristics`

## Active work item

```yaml
id: pytorch-core-training-switch-over
title: Activate the PyTorch core trainer while keeping NumPy reference-only until external continuity audit closes the window
status: in_progress
```

## Current blocker details

None. The latest full gate record for the PyTorch core-training migration batch is:
- `.venv/bin/ruff check .` → All checks passed!
- `.venv/bin/mypy src` → Success: no issues found in 43 source files
- `.venv/bin/pytest` → 103 passed
- `git diff --check` → clean

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

## Immediate next actions

1. Run `quantlab-ml audit-continuity --registry-root ...` against active registries so NumPy freeze/deprecation and legacy-compat retirement are blocked by real inventory evidence rather than local inference.
2. Define the provider-agnostic remote GPU workflow for meaningful training/search runs without widening into orchestration, secret, or scheduler work.
3. Publish the official real-training runbook so production-profile training stops reading like laptop-scale continuity work.
4. Freeze or retire the NumPy reference path once the external continuity audit confirms zero active dependency.
5. Retire the temporary legacy compat window once the continuity audit reports zero active legacy dependencies.

## Update rule

After every meaningful task:
- update `current_focus` if it changed
- update `declared_next_task`
- update `Recently completed`
- update the active work item or replace it
