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
- decision: The active core Phase 5 trainer now uses a PyTorch-backed linear policy trainer while preserving the existing `linear-policy-v1` runtime adapter and `LinearPolicyParameters` payload surface. The NumPy trainer remains only as a narrow reference/continuity backend until external continuity audit confirms that freeze/deprecation is safe.
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
