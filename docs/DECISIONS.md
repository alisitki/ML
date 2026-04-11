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
- decision: PyTorch is the training stack; ONNX/TensorRT may be used only for runtime inference acceleration.
- why: Training and deployment concerns must remain separate, and TensorRT is not the training system.
