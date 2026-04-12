# AGENTS.md

## Repository purpose

This repository is the QuantLab ML policy-discovery repository.

Its purpose is to:
- build learning surfaces from raw multi-exchange collector streams,
- define reward and evaluation surfaces,
- train offline policy-learning systems,
- evaluate policies on time-ordered out-of-sample surfaces,
- produce and store policy artifacts,
- support a separate runtime selector / inference layer.

This repository is **not** the live executor.

The live system is intentionally split into:
1. offline training
2. runtime selector / inference
3. thin executor

The executor must remain thin and only handle:
- feasibility checks,
- capital allocation,
- order execution.

## Canonical sources

This file is the operational entrypoint, not the only source of truth.

Canonical documents:
- `docs/QUANTLAB_CONSTITUTION.md`
- `docs/PROMOTION_GATE.md`
- `docs/CANONICAL_MARKET_DATA_CONTRACT.md`
- `docs/OBSERVATION_SCHEMA.md`
- `docs/ACTION_SPACE.md`
- `docs/REWARD_SPEC.md`
- `docs/REWARD_SPEC_V1.md`
- `docs/POLICY_ARTIFACT_SCHEMA.md`
- `docs/RUNTIME_BOUNDARY.md`
- `docs/EXECUTION_INTENT_SCHEMA.md`
- `docs/SPLIT_POLICY.md`
- `docs/SPLIT_POLICY_V1.md`
- `docs/REGISTRY_SCHEMA.md`
- `docs/ROADMAP.md`
- `docs/PROJECT_STATE.md`
- `docs/BACKLOG.md`
- `docs/DECISIONS.md`
- `docs/EVALUATION_RUNBOOK.md`
- `docs/TASK_TEMPLATE.md`
- `docs/REPORT_TEMPLATE.md`

If this file conflicts with those documents, those documents win.

## Interpretation precedence

When repo text leaves room for interpretation, use this precedence order:

1. Constitution
2. Canonical contracts
3. Decisions
4. Project state / backlog
5. Temporary compatibility / implementation convenience

Interpretation guardrails:
- Allowed does not mean default.
- Temporary does not mean strategic direction.
- Optional experiment paths must not be expanded as if they were the core architecture.
- Compatibility paths are tolerated only to preserve continuity, not to attract new development.
- When a temporary path conflicts with the core direction, the core direction wins.

## Project-state reading order

Before proposing or changing code, read in this order:

1. `docs/PROJECT_STATE.md`
2. `docs/ROADMAP.md`
3. `docs/BACKLOG.md`
4. `docs/DECISIONS.md`
5. relevant canonical contract docs

If `docs/PROJECT_STATE.md` declares an active next task, do not invent a different priority unless you explicitly justify the deviation.

Before implementing any meaningful task, explicitly classify the path as one of:
- core direction
- optional experiment
- temporary compatibility maintenance
- forbidden-as-default area

After any meaningful task:
- update `docs/PROJECT_STATE.md`
- update `docs/BACKLOG.md` if task state changed
- update `docs/DECISIONS.md` if a real architectural decision was made
- update canonical docs if semantics changed

## Fixed system identity

QuantLab is an ML-first policy discovery engine.

The model learns from raw collector streams and may produce many single-asset policies.
Cross-symbol context is allowed and encouraged, but action ownership always belongs to one target-asset policy.

Policies may include:
- entry
- exit
- hold horizon
- size
- leverage / margin
- venue choice
- no-trade behavior

Human-readable rules are not required.

## Non-negotiable rules

- Never use random split.
- Default split policy is custom walk-forward.
- If label horizons or information sets overlap, purge + embargo is mandatory.
- Leakage tolerance is zero.
- Any learned transform must fit on train only.
- Final untouched test set must never be reused for tuning, model selection, or champion decisions.
- Search budget transparency is mandatory.
- A single attractive backtest is never enough for promotion.
- Champion / challenger regime is mandatory.
- Registry is mandatory.
- Runtime uses inference artifacts only.
- Live learning is forbidden in the initial operating model.

## Learning-surface rules

Do not destroy the edge through simplification.

The core learning surface must preserve:
- time
- symbol
- exchange
- stream
- field-level raw information
- contract-availability vs missing vs stale vs padding semantics

Do not reintroduce in core contracts:
- exchange averaging
- symbol averaging
- stream single-scalar collapse
- hidden reductions that destroy cross-exchange or cross-symbol structure

Compatibility reductions may exist only in explicit compatibility modules.

## Reward rules

The governing objective is:

net profit
- risk penalty
- unnecessary trading penalty

Fees, funding, slippage, and realistic execution assumptions are mandatory.

Do not drift toward prediction-only scoring.

## Runtime / deployment rules

- PyTorch is the default training stack.
- ONNX and TensorRT are allowed only for runtime inference acceleration.
- Do not move training logic into runtime.
- Do not move hidden strategy logic into executor.
- Training artifact and deployment artifact are separate.

## Required behavior from Codex

When making changes:
- preserve constitutional rules,
- preserve contract semantics,
- call out leakage risk explicitly,
- call out backtest-overfitting risk explicitly,
- update tests for every behavior change,
- update docs when semantics change,
- do not declare success based only on fixture smoke tests,
- do not silently widen scope,
- do not ignore the declared active next task without justification.

## Commands

Use these commands by default unless the task explicitly says otherwise.

### Full test suite

```bash
pytest
```

### Targeted tests

```bash
pytest -q tests/
```

### Lint

```bash
ruff check .
```

### Type check

```bash
mypy src
```

### Notes on commands

- Do not assume fixture-only success means production readiness.
- Do not assume a single backtest means economic validity.
- Do not assume a passing smoke test means reward or split correctness.
- Do not assume compatibility adapters define the core architecture.
- Do not assume an allowed path is a default path.
- Do not assume a temporary continuity path defines long-term direction.

## Required output for meaningful changes

For any non-trivial task, report:
- what changed
- why it changed
- which constitutional or contract rule it serves
- what tests were added or updated
- what remains unverified
- which risks still remain
- what the next recommended task is
