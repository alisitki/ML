# Backlog

## Purpose

This file tracks near-term and medium-term work items.

Status values:
- `todo`
- `in_progress`
- `blocked`
- `done`

---

## Active items

### QL-001
- title: Gap audit for v1 implementation alignment
- status: done
- depends_on: governance/spec freeze
- scope: observation schema, reward_v1, split_v1
- done_when:
  - doc/code mismatches are listed
  - blockers are separated from non-blockers
  - next task is declared in PROJECT_STATE
- audit_findings:
  - observation: no confirmed blocker from audit evidence; current fixture-surface implementation preserves the core observation axes, masks, and causality semantics covered by existing tests
  - reward_v1: confirmed blocker; evaluation collapses explicit decision venue/size/leverage semantics, reward application still permits venue=None fallback, selected_venue is not propagated, funding freshness/formula semantics drift from reward_v1, and turnover-event semantics are not implemented
  - split_v1: confirmed blocker; implementation is limited to static train/eval ranges and lacks validation/final untouched test, deterministic walk-forward folds, purge/embargo handling, and split artifacts/version persistence

### QL-002
- title: Implement split_v1_walkforward canonical builder behavior
- status: done
- depends_on: QL-001
- scope: train/validation/final untouched test segmentation, deterministic walk-forward fold generation, purge width, embargo width, split artifacts, split version persistence
- blocker_reason:
  - overlap leakage risk remains without canonical walk-forward plus purge/embargo discipline
  - official evaluation remains invalid and backtest-overfitting risk remains high without validation/final untouched test segmentation
  - promotion and out-of-sample discipline cannot be enforced without persisted split artifacts and versioned boundaries
- completion_notes:
  - canonical train/validation/final_untouched_test splits now exist in the trajectory bundle
  - split artifact now persists split_v1_walkforward version, fold generation config, purge/embargo widths, fold boundaries, and final untouched test boundary
  - default evaluation path now targets validation while final untouched test remains off the default development loop
- done_when:
  - deterministic walk-forward folds exist inside the development region
  - validation and final untouched test segments exist
  - fold boundaries are persisted
  - purge/embargo widths equal the maximum information reach
  - final untouched test boundary is persisted
  - split version is stored
  - overlap case tests pass

### QL-003
- title: Enforce reward_v1 math in code
- status: done
- depends_on: QL-001, QL-007
- scope: explicit decision venue propagation, no `venue=None` fallback, horizon-end pricing, funding freshness threshold semantics, reward_v1 funding formula, turnover-event semantics, infeasible-action penalty parity
- blocker_reason:
  - reward_v1 parity remains broken while official code drifts from the frozen venue, horizon, funding, and turnover semantics
  - official evaluation validity is weakened if evaluated reward semantics differ from declared reward_v1
  - promotion and champion comparison semantics drift if candidates are scored under non-canonical reward behavior
- completion_notes:
  - evaluation now preserves explicit decision venue, size_band_key, and leverage_band_key semantics instead of flattening to action-only fallback
  - directional reward evaluation now requires explicit venue selection and no longer uses `venue=None` best-venue fallback
  - reward math now uses horizon-end pricing, funding freshness thresholding, v1 funding sign semantics, and turnover-event penalties driven by exposure change
  - effective selected venue is now propagated into `RewardContext` during reward application and evaluation, closing the remaining REWARD_SPEC_V1 context gap
- done_when:
  - implementation matches REWARD_SPEC_V1
  - evaluation applies the requested venue semantics instead of implicit best-venue fallback
  - effective selected venue semantics are explicit in reward context
  - horizon-end price semantics are used for reward evaluation
  - funding freshness threshold behavior matches reward_v1
  - turnover penalty is driven by exposure-state change semantics
  - parity tests pass
  - no hidden averaging remains

### QL-004
- title: Enforce canonical observation schema in code
- status: in_progress
- depends_on: QL-001
- scope: axes, masks, derived surface, scale preset, causality, runtime compatibility enforcement
- confirmed_gaps:
  - Batch 1 verification confirmed that `TrajectoryBuilder` produced `derived_surface` channels while the active feature extractor ignored them; Batch 1 remediation wires those channels into the shared training/runtime feature vector with tests.
  - The temporary legacy compat window is still open for deterministic legacy `linear-policy-v1` artifacts; this is explicit and logged, but it remains a retirement item rather than a final-state design.
  - Derived-surface support exists as augmentation only; it does not elevate derived channels to the core/default observation path.
- completion_notes:
  - Batch 2 added a structured strict runtime contract to new policy artifacts, including scale specs, raw surface shapes, derived contract/version metadata, derived channel templates/signature, and expected feature dimension.
  - Runtime now rejects scale-spec mismatches, raw-shape mismatches, derived contract drift, derived channel identity/order drift, feature-dimension mismatches, and deprecated `momentum-baseline-v1` artifacts before inference.
  - Legacy artifacts are no longer silently accepted; only deterministic legacy `linear-policy-v1` artifacts can enter through a temporary explicit compat window, and acceptance is logged as deprecated.
  - The canonical production observation preset now exists as `configs/training/production.yaml`, while `configs/training/default.yaml` remains an explicitly non-production continuity/smoke profile.
  - Development-region trajectories are now persisted explicitly so walk-forward fold consumption can use boundary-crossing development steps without weakening canonical split semantics.
- remaining_follow_ups:
  - remove the temporary legacy compat window after the continuity audit reports zero active legacy dependencies
- done_when:
  - code matches OBSERVATION_SCHEMA
  - schema-sensitive tests pass
  - compatibility layer boundaries are explicit

### QL-005
- title: Freeze policy artifact and execution intent path
- status: done
- depends_on: QL-001
- scope: artifact metadata, compatibility tags, execution intent contract
- completion_notes:
  - policy artifacts now carry canonical top-level identity, runtime metadata, compatibility tags, lineage hooks, and reward/evaluation surface linkage
  - runtime bridge now enforces artifact compatibility against observation schema requirements instead of guessing compatibility
  - runtime selector output can now be materialized as an explicit execution intent with traceable policy/artifact ids, venue, size, leverage, ttl, and selector trace id
- done_when:
  - selector output matches EXECUTION_INTENT_SCHEMA
  - executor boundary stays thin
  - artifact compatibility enforcement exists

### QL-007
- title: Minimal policy-state dependency for reward_v1 turnover-event semantics
- status: done
- depends_on: QL-001
- scope: minimum exposure-state representation and action-to-state mapping needed so reward_v1 can distinguish exposure changes from non-changes for turnover-event semantics
- completion_notes:
  - no separate epic was required; reward_v1 turnover-event semantics now derive from existing action metadata plus evaluation-side policy-state transitions
  - venue/size/leverage ownership remained inside QL-003 because those semantics are part of the core reward application path rather than a separate dependency track
- done_when:
  - reward code can determine whether a decision changes exposure state
  - turnover-event semantics can be implemented without guessing from action labels alone
  - the dependency stays limited to turnover-event support rather than expanding into a broader action-space epic

### QL-006
- title: Registry schema and promotion-gate enforcement
- status: done
- depends_on: QL-005
- scope: score history, lineage, search-budget fields, champion/challenger constraints
- completion_notes:
  - registry records now carry canonical policy identity, search-budget summary, split evidence, runtime compatibility tags, evaluation linkage, and promotion-decision lineage
  - scored candidates remain challengers until explicit promotion; champion status is no longer assigned automatically from raw score ranking
  - promotion gate now records auditable decisions and enforces split discipline, leakage checks, search-budget presence, champion comparison, reproducibility, artifact completeness, and runtime boundary evidence
  - paper/sim evidence is now recorded as a first-class registry-linked evidence record, and promotion consumes typed paper/sim evidence ids instead of ad-hoc report placeholders
- done_when:
  - unscored champion impossible
  - registry fields match REGISTRY_SCHEMA
  - promotion prerequisites are checkable

### QL-008
- title: Replace baseline policy-learning path with a real training loop
- status: done
- depends_on: phase-4 paper/sim operationalization
- scope: learned policy fitting on train only, validation-based model selection, search-budget summary recording, policy artifact / inference artifact separation
- completion_notes:
  - the active training path now fits learned linear policy weights from train split data instead of using the old momentum-threshold heuristic
  - learned normalization is fit on train only, best checkpoint selection uses validation only, and final untouched test remains outside the training loop
  - training artifacts still register cleanly, export to inference artifacts, and remain compatible with evaluation, registry, and promotion-gate flows
- done_when:
  - real training loop exists
  - produced policies carry search-budget transparency metadata
  - validation-based selection does not consume final untouched test
  - OOS evidence path remains intact

### QL-009
- title: Expand the real training path into explicit search-budgeted candidate generation
- status: done
- depends_on: QL-008
- scope: opt-in `candidate_search` config, multi-candidate trainer result surface, backward-compatible selected-artifact CLI flow, registry-visible run linkage without registry schema drift
- completion_notes:
  - explicit `candidate_search` configs now produce multiple learned candidates under a shared `training_run_id` and global search-budget summary
  - validation remains the only development-time selection surface and final untouched test remains outside the loop
  - `quantlab-ml train` keeps the selected artifact at `--output`, emits a search manifest and sidecar candidate artifacts only for multi-candidate runs, and can register all candidates without widening the registry schema
  - repo-wide verification gate is now fully green: `ruff check .`, `mypy src`, `pytest -q` (80 passed), `git diff --check` all clean
  - typing fixes: unused imports, None guards, Literal cast, return type annotations, branch-local type narrowing — no semantic changes
  - `_coerce_event_time()` typing surface preserved; 4 targeted input-surface tests added
- done_when:
  - more than one learned candidate can be produced under explicit search-budget accounting
  - validation selection remains the only development-time selection surface
  - produced candidates remain registrable and promotion-gate compatible
  - full verification gate is green

### QL-010
- title: Decide and implement walk-forward fold consumption in the core training loop
- status: done
- depends_on: QL-004, QL-008
- scope: decide whether fold-aware development iteration is mandatory now; if yes, implement it explicitly rather than leaving folds as metadata-only
- completion_notes:
  - walk-forward fold consumption is now mandatory for candidate selection instead of remaining metadata-only
  - fold selection uses persisted development-region trajectories so validation windows can cross the canonical train/validation boundary without dropping usable steps
  - purge width is applied before fold training and normalization is fit on fold-train only, reducing overlap leakage risk explicitly
  - selected candidate specs are refit on the canonical train split and still choose epochs on canonical validation only; final untouched test remains outside the loop
- done_when:
  - the repo has an explicit decision on fold consumption
  - if required, trainer behavior uses the decided fold discipline
  - the remaining backtest-overfitting risk is documented or reduced explicitly

### QL-011
- title: Add the canonical production observation preset and clarify smoke-vs-production training profiles
- status: done
- depends_on: QL-004
- scope: add the explicit `1m×8`, `5m×8`, `15m×8`, `60m×12` production preset/config and stop relying on a smoke-oriented training profile to stand in for future readiness
- notes:
  - `configs/training/default.yaml` remains continuity-oriented for now; any rename to `smoke.yaml` or `fixture-train.yaml` should happen only in an explicit CLI/test compatibility batch
- completion_notes:
  - `configs/training/production.yaml` now carries the canonical production observation preset as a first-class profile
  - `README.md` and project state text now say explicitly that `configs/training/default.yaml` is continuity/smoke only and is not the production profile
- done_when:
  - production observation preset is a first-class config
  - smoke/fixture profile and production profile are clearly separated

### QL-012
- title: Track D-011 exit criteria and the PyTorch core-training migration milestone
- status: done
- depends_on: D-011 guardrails
- scope: make the PyTorch core-training migration and NumPy-path exit criteria visible, tracked, and auditable
- completion_notes:
  - registry continuity audit now reports active training backend counts and makes the NumPy continuity window machine-visible instead of doc-only
  - project state and decisions now point at `quantlab-ml audit-continuity` for explicit D-011 exit tracking
- done_when:
  - D-011 exit criteria are represented in state/backlog/task flow
  - NumPy continuity support no longer reads like an open-ended default

### QL-013
- title: Retire or freeze legacy compat paths instead of expanding them
- status: done
- depends_on: D-012 guardrails
- scope: track actual artifact inventory dependency for legacy compat layers and retire/freeze any compat path that no longer preserves active continuity
- completion_notes:
  - registry continuity audit now reports active legacy compat dependence and deprecated momentum artifacts explicitly
  - legacy compat retirement is now keyed to observed active inventory rather than open-ended cautionary text alone
- done_when:
  - compat layers have explicit retirement/freeze tracking
  - no compat path survives as an implied growth area without an active dependency

### QL-014
- title: Adopt interpretation guardrails across task intake and naming clarity
- status: todo
- depends_on: AGENTS interpretation precedence, TASK_TEMPLATE path classification
- scope: ensure new tasks actually use path classification and evaluate whether misleading continuity-oriented names should be cleaned up in a dedicated batch
- done_when:
  - task intake consistently records path classification
  - allowed/temporary/optional paths are less likely to be mistaken for defaults
  - any future config rename is handled as an explicit compatibility change, not an incidental cleanup

### QL-015
- title: Clarify the local-vs-remote execution policy for training modes
- status: done
- depends_on: D-014
- scope: make the repo say explicitly that meaningful real training is PyTorch-first and remote GPU-first when available, while local CPU/laptop runs remain smoke/debug/continuity only
- completion_notes:
  - constitution, decisions, runtime boundary, roadmap, project state, README, and AGENTS now separate real training from local continuity modes
  - task intake now asks whether a task is real training, continuity baseline, or smoke/debug and whether it is accidentally optimizing around laptop constraints
- done_when:
  - local smoke/debug, continuity baseline, and real training modes are clearly distinguished
  - provider-agnostic remote GPU intent is explicit for meaningful training

### QL-016
- title: Migrate the core training path from temporary NumPy continuity to PyTorch
- status: in_progress
- depends_on: D-011, D-014, QL-012
- scope: replace the temporary NumPy continuity backend in the core training path while preserving fold consumption, train-only normalization, validation-only selection, and search-budget accounting
- completion_notes:
  - `LinearPolicyTrainer` now defaults to the PyTorch core backend while preserving the existing `linear-policy-v1` runtime adapter and `LinearPolicyParameters` payload surface
  - NumPy remains only as a narrow reference/parity backend and the deprecated `MomentumBaselineTrainer` shim now routes explicitly through that continuity path
  - parity coverage now exercises default, `search-small`, and `production` training profiles against the NumPy reference path without weakening split, purge, or final untouched test discipline
  - PyTorch training metadata now records the effective training device, CUDA availability, and device name so a remote GPU run can prove that it actually executed on CUDA rather than merely importing PyTorch
- open_items:
  - run `quantlab-ml audit-continuity --registry-root ...` against active registries before treating the NumPy freeze/deprecation path as closed
  - freeze or retire the NumPy reference path once the external audit confirms zero active dependency
- done_when:
  - PyTorch is the only active core-training backend
  - external continuity audit no longer shows NumPy as an active core-training dependency
  - parity gaps are recorded explicitly if any remain

### QL-017
- title: Define the provider-agnostic remote GPU workflow for real training
- status: done
- depends_on: D-014, QL-016 core switch-over
- scope: define how meaningful training/search runs target remote rented GPU compute, including execution handoff, expected artifacts/logs, and the local-vs-remote boundary, without adding orchestration, secrets, or provider lock-in
- completion_notes:
  - the trainer now exposes effective device resolution in logs and `training_summary`, making the first controlled GPU run auditable instead of inferring GPU use from environment alone
  - `docs/REMOTE_GPU_RUNBOOK.md` now defines the official provider-agnostic remote GPU workflow using Vast only as the concrete example path
  - the first controlled run is explicitly separated from the external `audit-continuity` operational follow-up, so the repo does not treat continuity inventory audit as a blocker for first GPU execution
- done_when:
  - repo has an official remote-GPU real-training workflow
  - Vast.ai or equivalent providers appear only as examples, not hard dependencies
  - local continuity runs are clearly excluded from the real-training default

### QL-018
- title: Publish the official real-training snapshot and runbook
- status: done
- depends_on: QL-017
- scope: define the production-scale observation surface, search budget posture, expected compute class, and reporting bundle for official real-training runs
- completion_notes:
  - `configs/data/controlled-remote-day.yaml` now provides the official bounded full-day example snapshot for the first controlled remote run
  - `docs/REMOTE_GPU_RUNBOOK.md` now defines bootstrap, preflight, command flow, output bundle, acceptance criteria, and failure triage for the first controlled run
  - README now points directly to both the controlled snapshot config and the official remote GPU runbook so the repo no longer relies on implicit workflow knowledge
- done_when:
  - real-training runbook exists
  - production-profile runs no longer read like laptop-sized local work
  - artifact/report expectations are explicit before promotion-facing training begins

### QL-019
- title: Stream production trajectory build output instead of assembling a full bundle
- status: done
- depends_on: QL-018
- scope: persist the canonical build output as manifest plus per-split JSONL so production-size builds do not require an in-memory `TrajectoryBundle`
- completion_notes:
  - `TrajectoryBuilder.build_to_directory()` now writes `manifest.json` plus `development/train/validation/final_untouched_test` JSONL files through `TrajectoryDirectoryStore`
  - line-size guardrails now warn at 512 MB and fail at 2 GB so oversized trajectory records surface immediately
  - `TrajectoryBundle` and `TrajectoryStore` are now explicitly marked FIXTURE / TEST COMPAT ONLY instead of reading like the default path
- done_when:
  - production build writes a streaming trajectory directory
  - production build no longer depends on full-bundle assembly
  - compat boundaries are explicit in code

### QL-020
- title: Replace the active prod matrix-first trainer path with fold-aware streaming batch training
- status: done
- depends_on: QL-019, D-013, D-014
- scope: remove giant dense train/validation matrix assembly from the active prod trainer/evaluate path while preserving fold selection, purge discipline, train-only normalization, runtime payload compatibility, and final untouched test separation
- blocker_reason:
  - streaming JSONL alone did not fix the real blocker because the active prod trainer still rebuilt full dense train/validation matrices and degraded validation to a proxy score
  - overlap leakage and backtest-overfitting risk remain underdocumented if prod validation differs from the real evaluation path
- completion_notes:
  - prod `train_search_from_directory()` now uses train-only two-pass streaming stats, deterministic bounded batches, and true streaming validation/evaluation rather than proxy validation
  - matrix-first preparation now lives only in explicit `training/compat_matrix_first.py` helpers with a grep-able `_FIXTURE_TEST_COMPAT_ONLY` sentinel; guard tests verify prod directory train does not call them
  - prod logs and `training_summary` now expose `effective_batch_size`, `estimated_batch_bytes`, `batches_per_epoch`, `batch_target_bytes`, `training_data_flow`, `validation_data_flow`, and `proxy_validation_used=false`
  - directory `evaluate` now streams `final_untouched_test` through `EvaluationEngine.evaluate_records()` instead of materializing the split list first
- done_when:
  - prod trainer never assembles a giant dense train or validation matrix
  - prod validation uses the real streaming evaluation path
  - train-only normalization remains explicit and test-covered
  - compat/test matrix-first helpers cannot silently re-enter the prod path

### QL-021
- title: Implement shard-based raw tensor cache sidecars and shared batched prod evaluation
- status: in_progress
- depends_on: QL-020
- path_classification: core direction
- scope: write `tensor_cache_v1` shard sidecars for `development`, `train`, `validation`, and `final_untouched_test` during `build_to_directory`; keep canonical JSONL unchanged; move prod `train`, train-time `validation`, and prod directory `evaluate` onto the same cache family with explicit compat-only JSONL fallback
- motivation:
  - first controlled remote GPU run (2026-04-14) confirmed: `build-trajectories` OOM-free ✅, `training_device=cuda` ✅
  - bottleneck identified: each batch (batch_size=24 × feature_dim=680,413) assembles Pydantic `TrajectoryStep` -> numpy -> torch in Python GIL, ~130 MB/batch, 300 batches/epoch — estimated 84 min/epoch, 14+ hours for 8 epochs
  - GPU utilization was 0% (CPU-bound data pipeline, not GPU-bound)
  - bottleneck is NOT OOM, NOT instance size, NOT GPU memory — it is batch assembly cost
  - train-only acceleration is insufficient; validation/final evaluate would reintroduce the same Python-side blocker unless they share the cache family and batched inference path
- completion_notes:
  - `TrajectoryBuilder.build_to_directory()` now writes canonical JSONL plus `tensor_cache_v1/` shard manifests and bounded shard files for every prod split
  - prod `LinearPolicyTrainer.train_search_from_directory()` now uses raw tensor shards for feature stats, batch assembly, and train-time validation; cache-missing directories fail closed unless `allow_jsonl_fallback` is passed explicitly
  - `EvaluationEngine.evaluate_directory()` now uses the same tensor-cache family and batched linear-policy inference; directory evaluate no longer depends on per-step `PolicyRuntimeBridge.decide()`
  - targeted local verification now covers cache output parity, prod fast-path guardrails, train/evaluate fail-closed behavior, and explicit temporary-compat fallback behavior
- status_axes:
  - operational_state: done — the retained 2026-04-17 controlled rerun completed `build/train/evaluate/score/export` with exit `0`
  - acceptance_state: pending — the retained 2026-04-17 GPU samples remain below the runbook `avg_gpu_utilization >= 20` gate even when bounded to exact active windows, so targeted remote resampling is still required
  - promotion_state: not_started — the retained candidate remains a challenger and final untouched test net return is negative
- acceptance_criteria:
  - `build-trajectories` writes canonical JSONL plus `tensor_cache_v1` shard sidecars, replay sidecars, and a cache manifest for all 4 prod splits during `build_to_directory`
  - prod `train`, train-time `validation`, and prod directory `evaluate` use the tensor-cache family by default with `tensor_cache_used=true` and `jsonl_fallback_used=false`
  - no change to JSONL output (remains for compatibility / inspection)
  - controlled remote rerun shows `epoch_wall_sec < 300`, per-epoch `validation_wall_sec < 60`, `evaluate_wall_sec < 180`, and no `137`/OOM exits
  - average GPU utilization materially increases above the baseline and reaches at least `20%`
- done_when:
  - shard-based tensor cache sidecars are written by build and consumed by train/validation/evaluate
  - the controlled remote rerun proves the throughput and GPU-utilization gates above
  - 8 epochs complete end-to-end in practical wall-time with no silent JSONL fallback
  - local verification remains clean (`ruff`, `mypy`, targeted `pytest`)
- open_items:
  - retained evidence inspection is complete: the 2026-04-17 logs and GPU CSVs prove the hot path and timing gates, but exact active-window GPU averages remain below the runbook threshold
  - capture only the missing remote GPU-utilization proof on the same controlled config, preferring train/evaluate-only resampling if the retained trajectory directory is still available, and then refresh the derived retained-evidence index
  - keep promotion out of scope until a later economic/paper-sim batch

### QL-022
- title: Reconcile source-of-truth state with retained QL-021 evidence
- status: done
- depends_on: QL-021 retained bundle
- path_classification: core direction
- scope: align `PROJECT_STATE`, `BACKLOG`, and retained 2026-04-17 QL-021 evidence so repo-memory, retained artifacts, and current core-path narration describe the same truth
- completion_notes:
  - repo-memory now records QL-021 on 3 independent axes: `operational_state`, `acceptance_state`, and `promotion_state`
  - `outputs/ql021-controlled-remote-rerun-20260417-build-fresh/acceptance_evidence.json` now exists only as a derived retained-evidence index over existing logs, manifests, artifacts, and GPU CSV samples; it is not a second source of truth
  - proof-gated reconciliation checked QL-010 / QL-011 / QL-012 / QL-013 and found no reopen trigger from current repo truth, so stale summary drift alone no longer reopens them by default
  - backend/core-path narration and the role of `configs/training/production.yaml` are now aligned with retained run evidence and existing D-011 / D-014 guardrails
- done_when:
  - repo-memory and retained evidence tell the same current story for QL-021
  - QL-021 is not collapsed into one ambiguous status
  - no unrelated historical QL is reopened without explicit proof
- non_goals:
  - no architecture reset
  - no reward redesign
  - no runtime-boundary expansion
  - no search-budget widening
  - no backend flip as part of reconciliation

### QL-023
- title: Fix manifest-based registry coverage accounting on the manifest registration path
- status: done
- depends_on: QL-022
- path_classification: core direction
- scope: derive dataset-surface train coverage from retained split facts when registering directory-path training results, and fail loudly when legacy manifests cannot recover retained train coverage, closing the `train_samples=0` auditability gap without widening registry semantics
- completion_notes:
  - streaming manifests now persist split-scoped record and step counts so retained build output carries the train coverage facts needed by registry accounting
  - manifest registration now derives `train_sample_count` from retained split facts and falls back to the retained tensor-cache manifest only when `trajectory_directory` is supplied and the retained cache evidence is present
  - legacy manifests without split counts now fail loudly instead of warning and emitting zero train coverage
  - regression coverage now protects successful fallback, hard-failure missing-fallback paths, non-zero train coverage on non-empty trajectory directories, and dataset-surface provenance for `train_sample_count`
- done_when:
  - non-empty prod train directories do not register with zero train coverage
  - legacy manifests without reliable fallback inputs fail loudly instead of silently coercing zero train coverage
  - `CoverageStats.field_origins` remain unchanged: dataset-surface-derived fields stay dataset-surface-derived and policy-evidence-derived fields stay policy-evidence-derived
  - lineage and coverage metadata remain auditable without inventing a new registry provenance system


### QL-100
- title: ONNX / TensorRT runtime acceleration exploration
- status: todo
- depends_on: runtime selector maturity
- scope: inference acceleration only
- done_when:
  - runtime selector is stable
  - deployment artifact path exists

### QL-101
- title: Reward v2 path-aware redesign
- status: todo
- depends_on: reward_v1 stabilization
- scope: richer risk and carry treatment
- done_when:
  - reward_v1 is stable and versioned
