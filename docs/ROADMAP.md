# Roadmap

## Purpose

This document defines the high-level phase order for QuantLab.

It answers:
- where the project is going
- which phase comes before which
- what each phase is trying to prove

This is not the live state file.
Use `docs/PROJECT_STATE.md` for current status.

## Phase 0 — Governance and V1 spec freeze

Goal:
- freeze constitutional rules
- freeze canonical contracts
- freeze v1 reward and split rules
- install repo memory and operating flow

Exit criteria:
- governance docs committed
- operational state docs committed
- AGENTS reading order committed

## Phase 1 — Surface and reward alignment

Goal:
- align code with canonical observation schema
- align code with reward_v1
- align code with split_v1
- remove semantic drift between docs and implementation

Exit criteria:
- observation construction passes schema audit
- reward implementation matches `reward_v1`
- split builder matches `split_v1_walkforward`
- tests cover leakage-sensitive semantics

## Phase 2 — Registry and artifact discipline

Goal:
- enforce policy artifact schema
- enforce registry schema
- enforce champion/challenger gates
- ensure search-budget recording is wired

Exit criteria:
- artifacts versioned and linked
- unscored champion impossible
- promotion gate inputs persisted
- registry lineage works

## Phase 3 — Runtime selector / executor boundary

Goal:
- freeze runtime selector boundaries
- freeze execution intent contract
- ensure executor remains thin
- prevent policy intelligence from leaking into executor

Exit criteria:
- runtime selector consumes inference artifacts
- executor consumes execution intent only
- traceability from runtime decision to artifact exists

## Phase 4 — Paper/sim operating loop

Goal:
- make official evaluation and paper/sim path operational
- make promotion flow reproducible
- ensure artifact completeness for candidate promotion

Exit criteria:
- evaluation runbook is executable
- paper/sim evidence can be attached to candidates
- promotion check can be run consistently

## Phase 5 — Real policy-learning implementation

Goal:
- replace dummy/baseline logic with real policy-learning systems
- converge the core training path onto PyTorch
- make meaningful real training target remote GPU compute when available rather than local laptop continuity assumptions
- preserve all governance and contract constraints
- begin genuine policy discovery

Exit criteria:
- real training loop exists
- meaningful-size training and search runs are explicitly treated as GPU-first when available
- policies are produced under search-budget discipline
- OOS evidence path is intact

## Phase 6 — Deployment readiness

Goal:
- prepare runtime inference for low-latency or production-like execution
- optionally add ONNX/TensorRT acceleration
- preserve selector/executor boundary

Runtime inference acceleration remains separate from Phase 5 training compute decisions.

Exit criteria:
- deployment artifacts exist
- runtime selector can consume them
- executor path remains thin
- no live learning introduced

## Ordering rule

Phases are sequential by default.

Do not skip ahead unless:
- the dependency is explicitly waived,
- the deviation is written into `docs/PROJECT_STATE.md`,
- and the reason is justified.

Later phases and allowed future options do not override the current active next task in `docs/PROJECT_STATE.md`.
A phase label does not mean all carried-over governance gaps are closed; unresolved gaps stay active until state/backlog retire them explicitly.
