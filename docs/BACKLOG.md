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
  - The canonical production preset (`1m×8`, `5m×8`, `15m×8`, `60m×12`) still does not exist as an explicit training config; current shipped configs remain fixture/smoke oriented.
  - The temporary legacy compat window is still open for deterministic legacy `linear-policy-v1` artifacts; this is explicit and logged, but it remains a retirement item rather than a final-state design.
- completion_notes:
  - Batch 2 added a structured strict runtime contract to new policy artifacts, including scale specs, raw surface shapes, derived contract/version metadata, derived channel templates/signature, and expected feature dimension.
  - Runtime now rejects scale-spec mismatches, raw-shape mismatches, derived contract drift, derived channel identity/order drift, feature-dimension mismatches, and deprecated `momentum-baseline-v1` artifacts before inference.
  - Legacy artifacts are no longer silently accepted; only deterministic legacy `linear-policy-v1` artifacts can enter through a temporary explicit compat window, and acceptance is logged as deprecated.
- remaining_follow_ups:
  - add the canonical production observation preset/config (`1m×8`, `5m×8`, `15m×8`, `60m×12`)
  - remove the temporary legacy compat window after legacy artifact stock is refreshed
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

## Parked items

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
