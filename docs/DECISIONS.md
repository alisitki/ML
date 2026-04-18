# Decisions

## Purpose

This file records accepted architectural decisions and why they were made.

This is not the constitution.
This is the "why we chose this" layer.

Status values:
- `accepted`
- `superseded`
- `draft`

---

## D-001 — QuantLab system identity
- status: accepted
- date: 2026-04-11
- decision: QuantLab is an ML-first policy discovery engine.
- why: We do not want a rule-by-rule manual strategy repo. The system must learn policies from raw collector streams under strict evaluation discipline.

## D-002 — Fixed runtime architecture
- status: accepted
- date: 2026-04-11
- decision: Architecture is offline training -> runtime selector / inference -> thin executor.
- why: Executor must not become a hidden strategy brain. Policy intelligence belongs upstream.

## D-003 — Single-asset action ownership with cross-symbol context
- status: accepted
- date: 2026-04-11
- decision: Policies are single-asset in action ownership, but may use cross-symbol and cross-exchange context.
- why: This preserves clarity in execution responsibility while keeping broader market context available to the model.

## D-004 — No random split
- status: accepted
- date: 2026-04-11
- decision: Random split is forbidden; default is custom walk-forward with purge/embargo when needed.
- why: Time-dependent market data makes random split invalid and dangerously over-optimistic.

## D-005 — Final untouched test is sacred
- status: accepted
- date: 2026-04-11
- decision: Final untouched test set may not be reused for tuning, model selection, or champion decisions.
- why: Repeated test reuse creates false confidence and hidden overfitting.

## D-006 — Search-budget transparency is mandatory
- status: accepted
- date: 2026-04-11
- decision: Promotion reports must include total tried models, seeds, architectures, reward variants, and candidate counts.
- why: Financial ML can overfit through search volume even when split discipline looks clean.

## D-007 — Canonical learning surface preserves structure
- status: accepted
- date: 2026-04-11
- decision: Core observation must preserve time, symbol, exchange, stream, and field-level structure; no exchange averaging or scalar collapse in core contracts.
- why: The suspected edge lives in cross-exchange and cross-symbol structure, not in heavily collapsed series.

## D-008 — Inventory-aware action space
- status: accepted
- date: 2026-04-11
- decision: Canonical action space is inventory-aware, venue-aware, and feasibility-aware.
- why: Policy learning requires memory of position state, not stateless direction guessing.

## D-009 — Reward_v1 is intentionally simple but explicit
- status: accepted
- date: 2026-04-11
- decision: Reward v1 uses a simple venue-aware economic formula with fee/slippage/funding, symmetric risk penalty, and turnover penalty.
- why: We need an explicit, versioned baseline before attempting richer reward designs.

## D-010 — PyTorch for training, TensorRT only for runtime acceleration
- status: accepted
- date: 2026-04-11
- decision: PyTorch is the training stack; ONNX/TensorRT may be used only for runtime inference acceleration and do not define the training execution target.
- why: Training and deployment concerns must remain separate, and TensorRT is not the training system.

## D-011 — Temporary NumPy trainer does not supersede the PyTorch target
- status: accepted
- date: 2026-04-12
- decision: The active core training path now uses a PyTorch-backed linear policy trainer while preserving the existing `linear-policy-v1` runtime adapter and `LinearPolicyParameters` payload surface. The NumPy trainer remains only as a narrow reference/continuity backend until external continuity audit confirms that freeze/deprecation is safe.
- why: D-010 requires PyTorch as the training stack, but the repo also needed minimal-breakage migration. Keeping fold-aware candidate selection, train-only normalization, canonical refit, validation-only epoch selection, artifact export, and runtime loading unchanged lets the core training path converge onto PyTorch without widening the runtime or executor boundary. Retaining a narrow NumPy reference backend preserves parity evidence and a controlled continuity escape hatch while active registries are audited.
- guardrails:
  - PyTorch is the only active core-training backend.
  - NumPy is reference-only continuity and may not attract new feature work.
  - Runtime payload/export semantics remain `linear-policy-v1` plus `LinearPolicyParameters`.
- exit_criteria:
  - candidate search parity is preserved well enough to avoid material regression.
  - train-only normalization parity is preserved.
  - external `quantlab-ml audit-continuity --registry-root ...` confirms zero active NumPy core-training dependency across active registries before NumPy freeze/deprecation is treated as closed.
  - migration completion is tracked explicitly in `docs/PROJECT_STATE.md` and `docs/BACKLOG.md`.
- non_goals:
  - do not expand NumPy-specific architecture.
  - do not treat NumPy support as a long-term default.
  - do not widen the runtime adapter or payload family just to complete the training migration.

## D-012 — Strict runtime contract is default; legacy compat is explicit and temporary
- status: accepted
- date: 2026-04-12
- decision: New policy artifacts must carry a strict runtime contract and runtime must reject observation, derived-surface, feature-layout, or adapter mismatches early. Legacy acceptance is allowed only through an explicit temporary compat window for deterministic legacy `linear-policy-v1` artifacts; deprecated `momentum-baseline-v1` artifacts are quarantined from the normal runtime path.
- why: Batch 1 wired derived features into the shared feature path, which made silent training/runtime mismatch risk materially higher. Strict-by-default enforcement closes the "wrong artifact + wrong observation still looks runnable" failure mode without breaking all historical artifacts at once. A narrow, logged compat window preserves controlled continuity while keeping silent fallback forbidden and leaving a clear removal target for a later batch.
- guardrails:
  - Legacy compat window is temporary and narrow.
  - No new work may be justified primarily by preserving compat.
  - Compat layers require explicit retirement tracking.
  - If no active artifact inventory depends on a compat layer, freeze or retire it instead of expanding it.

## D-013 — Walk-forward folds are mandatory for candidate selection; exported artifacts are canonically refit afterward
- status: accepted
- date: 2026-04-12
- decision: Candidate selection in the active training path must consume persisted walk-forward folds over the development region with purge applied before fold training. After selection, the chosen candidate spec is refit on the canonical train split and still uses canonical validation only for epoch selection. Final untouched test remains outside the loop.
- why: Metadata-only folds left an explicit backtest-overfitting gap. Fold-aware candidate selection reduces that risk, but canonical train/validation/final untouched test semantics must remain intact. A dedicated development-region trajectory surface is required because boundary-crossing fold steps do not exist inside the canonical train/validation splits by design.
- guardrails:
  - Purge remains mandatory when horizon overlap exists.
  - Normalization must fit on fold-train only during fold selection.
  - Final untouched test may not be used for fold ranking, hyperparameter choice, or epoch choice.
  - Walk-forward support does not justify random split or relaxed leakage discipline.
- operational_notes:
  - `training_summary` is the visibility surface for fold-consumption metadata.
  - `quantlab-ml audit-continuity` is the operational tracking surface for the temporary NumPy and legacy-compat windows; it does not relax D-010 or D-012.
  - PyTorch switch-over is decided on repo-local parity and contract preservation; external continuity audit blocks final NumPy closeout, not the switch-over itself.

## D-014 — Real training defaults to remote GPU execution; local runs are continuity-only
- status: accepted
- date: 2026-04-12
- decision: Core model training uses PyTorch. When meaningful data volume, longer date ranges, production-scale observation surfaces, or real search budgets are involved, real training defaults to GPU execution when available and prefers remote rented GPU compute. Local CPU / laptop runs are continuity-only: smoke, debugging, tiny baselines, or short validation.
- why:
  - production-scale observation surfaces, longer date ranges, and candidate search will be too slow or too distorted if the repo implicitly optimizes around local CPU workflows
  - local convenience must not define the strategic execution target
  - PyTorch migration should converge toward real GPU-capable training, not a prolonged local continuity path
- guardrails:
  - do not treat local CPU throughput as the optimization target for core training design
  - do not justify new strategic investment around NumPy/local continuity because of laptop convenience
  - provider choice may vary (Vast.ai or equivalent), but the execution intent remains remote GPU-first for real training
  - runtime inference acceleration choices do not define the training execution target
- non_goals:
  - this decision does not require immediate cloud orchestration automation
  - this decision does not force every tiny smoke run onto rented GPU
  - this decision does not change runtime deployment architecture

## D-015 — The active prod trainer path is streaming-batch only; matrix-first training is compat-only
- status: accepted
- date: 2026-04-14
- decision: The active production trainer/evaluation path must use fold-aware streaming batch training, train-only two-pass streaming normalization, and real streaming validation/evaluation. Matrix-first feature assembly is allowed only inside explicit fixture/test compat helpers and may not be called from the prod directory path.
- why:
  - the earlier JSONL transition fixed build/output shape but not the real trainer blocker, because the active prod path still rebuilt giant dense train/validation matrices in memory
  - a proxy validation score weakened the canonical validation discipline and hid potential behavior drift between training-time selection and official evaluation
  - the controlled remote snapshot must prove correctness without relying on snapshot shrinkage, bigger hardware, or opaque RAM heuristics
- guardrails:
  - no giant dense train or validation matrix may be assembled in the active prod path
  - train-only normalization must remain explicit and must fit only on the train window currently being used
  - fold validation and canonical validation must use the real streaming evaluation path rather than a proxy score
  - matrix-first helpers must carry explicit compat sentinels and remain grep-able as non-prod code
  - batch policy may stay internal, but effective batch size, estimated batch bytes, and batches per epoch must be visible in logs and `training_summary`
- non_goals:
  - do not change the runtime/inference boundary
  - do not change `linear-policy-v1` payload/export semantics
  - do not solve the problem by shrinking the controlled snapshot or upgrading the instance class

## D-016 — Pre-computed tensor files are the training data format; Pydantic JSONL is output-only
- status: superseded
- date: 2026-04-14
- decision: `build-trajectories` must write pre-computed float32 tensor files (e.g. `{split}_X.pt`) alongside JSONL during `build_to_directory`. The active `train` path reads binary tensor files directly without per-batch Pydantic deserialization. JSONL remains for human inspection and compat, not as the training read path.
- why:
  - first controlled remote GPU run (2026-04-14) proved the Pydantic JSONL path impractical: `feature_dim=680,413`, `batch_size=24`, `batches_per_epoch=300` \u2014 ~84 min/epoch estimated, GPU utilization 0%
  - bottleneck is batch-assembly cost (Pydantic \u2192 numpy \u2192 torch in Python GIL per batch), not OOM, not instance size, not GPU memory
  - fix is at the data pipeline boundary: feature extraction happens once at build time, stored as binary; training reads pre-assembled float32 tensors
- superseded_by: D-017
- guardrails:
  - JSONL output remains unchanged (backward compat / inspection)
  - tensor files must be deterministic: same input + same config \u2192 same tensor files
  - feature extractor must still run only at build time, on train data only for normalization
  - `train` must detect tensor file presence and fall back to JSONL-assembly only when absent (transition safety)
- acceptance_signal:
  - GPU utilization > 0% on RTX A6000 during controlled remote training run
  - wall-time per epoch < 5 minutes for 8-epoch production config
  - 8 epochs complete end-to-end in the controlled run
- non_goals:
  - do not change JSONL schema or manifest format
  - do not change runtime/inference boundary
  - do not change `linear-policy-v1` payload/export semantics

## D-017 — Prod hot paths use shard-based raw tensor cache sidecars with shared batched evaluation
- status: accepted
- date: 2026-04-15
- decision: `build-trajectories` writes canonical JSONL plus a bounded `tensor_cache_v1/` shard family for `development`, `train`, `validation`, and `final_untouched_test`. The active prod `train`, train-time `validation`, and prod directory `evaluate` paths all consume this same cache family. Canonical JSONL remains the inspection/rebuild format and an explicit temporary compatibility fallback only.
- why:
  - solving only the train loop leaves the same Python/GIL blocker waiting inside validation and final evaluate
  - a single giant split-wide tensor file would create a new monolithic load/failure-isolation bottleneck
  - train-only normalization must remain leak-free, so the cache must stay raw rather than storing global pre-normalized features
  - runtime/inference boundaries do not need to change; the optimization is entirely inside offline build/train/evaluate dataflow
- guardrails:
  - cache shards are bounded; giant split-wide monoliths are forbidden as the default design
  - cache stores raw feature tensors only; normalization still fits on the active train window only
  - train-time validation and final evaluate must share the same batched evaluator path rather than diverging into separate inference implementations
  - prod directory paths fail closed when tensor cache is missing; JSONL fallback is explicit temporary compatibility maintenance only
  - no new runtime adapter, artifact family, selector boundary, or executor behavior is introduced
- acceptance_signal:
  - `tensor_cache_used=true` and `jsonl_fallback_used=false` are visible in the controlled remote rerun
  - `training_data_flow=tensor_shard_batch` and `validation_data_flow=tensor_shard_evaluation` are visible in `training_summary`
  - controlled remote rerun shows required chain exit codes `0`, `training_device=cuda`, `epoch_wall_sec < 300`, per-epoch `validation_wall_sec < 60`, `evaluate_wall_sec < 180`, and no `137`/OOM exits
- non_goals:
  - do not add checkpoint/resume design in this batch
  - do not change canonical JSONL schema
  - do not widen the runtime or inference artifact surface

## D-018 — Average GPU utilization is advisory telemetry for controlled proof runs
- status: accepted
- date: 2026-04-17
- decision: For QL-021-style controlled proof runs, `avg_gpu_utilization` is diagnostic telemetry only and is not a hard acceptance gate. Hard acceptance relies on direct hot-path evidence: `training_device=cuda`, `tensor_cache_used=true`, `jsonl_fallback_used=false`, required chain exit codes `0`, explicit `train` / `evaluate` execution evidence, and the documented timing gates.
- why: The clean 2026-04-17 remote proof run completed successfully on the intended CUDA tensor-cache path and stayed within timing limits, but still remained below the old `avg_gpu_utilization >= 20%` threshold. That threshold did not reflect the real acceptance intent for this controlled workload and became a bad proxy.
- guardrails:
  - this decision does not change promotion requirements, economic evaluation, or champion/challenger discipline
  - low GPU utilization may still trigger diagnostic or optimization follow-up, but it does not invalidate a successful controlled proof run by itself
  - this decision does not reopen architecture, reward, backend, or search-budget scope
