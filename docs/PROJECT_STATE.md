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

- current_phase: `Phase 5 real policy-learning implementation complete`
- current_focus: `Batch 2 remediation complete: new artifacts now carry a strict runtime contract, runtime rejects observation/feature mismatches early, and only a temporary explicit legacy compat window remains for deterministic legacy linear artifacts`
- current_blocker: `none`
- declared_next_task: `Reassess walk-forward fold consumption before QL-100, then close the remaining QL-004 gaps around the production observation preset and eventual legacy compat-window removal`
- not_now:
  - `live deployment plumbing`
  - `reward_v2 path-aware redesign`
  - `advanced selector heuristics`

## Active work item

```yaml
id: remediation-batch-2
title: Tighten runtime/artifact compatibility so observation, derived-surface, and feature-layout mismatches fail early instead of looking runnable
status: done
```

## Current blocker details

None. Verification gates for the remediation batch are clean:
- `.venv/bin/ruff check .` → All checks passed!
- `.venv/bin/mypy src` → Success: no issues found in 42 source files
- `.venv/bin/pytest` → 96 passed
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

## Immediate next actions

1. Decide whether walk-forward fold consumption should become a dedicated training backlog item before QL-100.
2. Close the remaining QL-004 gaps: add a production observation preset/config and define the removal plan for the temporary legacy compat window.

## Update rule

After every meaningful task:
- update `current_focus` if it changed
- update `declared_next_task`
- update `Recently completed`
- update the active work item or replace it
