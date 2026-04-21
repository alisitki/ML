---
status: canonical
owner: quantlab
last_reviewed: 2026-04-20
read_when:
  - before_ql031_redesign_implementation
supersedes: []
superseded_by: []
---

# QL-031 Model Redesign Plan

## Purpose

This document is the canonical implementation plan for the `QL-031` serious-model redesign track.

It does not change current repo truth by itself.
Current implemented reality is still governed by `docs/PROJECT_STATE.md`, `docs/ROADMAP.md`, and `docs/BACKLOG.md`.

This document defines:

- the approved redesign direction,
- the exact phase split,
- the V2 action contract decision,
- the Phase 0 instrumentation scope,
- the research stage-gates for `Phase 1A` and `Phase 1B`,
- the feasibility gate before any event-native expansion.

## Scope and interpretation

- This is `current-phase hardening` work inside `QL-031`.
- It is not a competing workstream to `QL-031`.
- It does not claim that `Phase 2` live/runtime layers already exist.
- It does not change promotion rules.
- `Phase 1A` and `Phase 1B` thresholds below are research stage-gates only.
- `Phase 1A` and `Phase 1B` thresholds are not promotion criteria.
- Promotion remains governed by `docs/PROMOTION_GATE.md`, `docs/EVALUATION_RUNBOOK.md`, and registry-backed same-surface comparison discipline.

## Current ceiling diagnosis

The approved audit conclusion is:

1. the current offline path preserves a rich structured observation surface in `TrajectoryBuilder`,
2. then collapses that surface into a very large flat feature vector,
3. trains an interaction-poor linear model,
4. uses action semantics that are narrower than the canonical inventory-aware surface,
5. optimizes a train objective that is materially misaligned with the governing sequential post-cost objective.

The approved redesign posture is therefore:

- keep `linear-policy-v1` as baseline and rollback anchor,
- build a parallel `V2` path,
- measure semantics/objective gain separately from encoder gain,
- preserve untouched-final discipline, promotion discipline, and runtime/evaluation traceability.

## Approved target design

### Primary recommendation

Build a parallel `V2` path in two controlled stages:

1. `Phase 1A`: semantics/objective redesign on the current observation surface.
2. `Phase 1B`: structured encoder redesign on top of the exact same `Phase 1A` semantics/objective.

This split is mandatory.
Without it, gain from action semantics and objective alignment cannot be separated from gain from the encoder.

### V2 action contract decision

This is an approved `V2` design decision, not current `v1` repo truth.

The `V2` action vocabulary is:

- `abstain`
- `hold`
- `exit`
- `enter_long@binance`
- `enter_long@bybit`
- `enter_long@okx`
- `enter_short@binance`
- `enter_short@bybit`
- `enter_short@okx`

The approved `V2` semantics are:

- `abstain` is valid only when the policy is currently `flat`
- in-position continuation is expressed only by `hold`
- position close is expressed only by `exit`
- implicit continuation through `abstain` is forbidden
- implicit close through `abstain` is forbidden
- implicit flip is forbidden
- implicit in-position venue switch is forbidden

State machine:

- `flat -> abstain => flat`
- `flat -> enter_long@venue => long@venue`
- `flat -> enter_short@venue => short@venue`
- `long@venue -> hold => long@venue`
- `short@venue -> hold => short@venue`
- `long@venue -> exit => flat`
- `short@venue -> exit => flat`

Invalid in `V2` bootstrap path:

- `flat -> hold`
- `flat -> exit`
- `long@venue -> abstain`
- `short@venue -> abstain`
- `long@venue -> enter_long@*`
- `short@venue -> enter_short@*`
- `long@venue -> enter_short@*`
- `short@venue -> enter_long@*`
- any `enter_*@venue` where the venue is contract-unavailable or decision-time infeasible

### Canonical tokenization for `Phase 1B`

Token definition:

- `token = (scale, bucket, symbol, exchange, stream)`

Token counts on the current 3 × 10 × 5 production surface:

- raw stream tokens: `(8 + 8 + 8 + 12) * 10 * 3 * 5 = 5400`
- derived tokens: `13`
- policy-state tokens: `1`
- total tokens: `5414`

Canonical tensor spec:

- `raw_field_tensor: [B, 5400, 7, 6]`
- `field_presence_mask: [B, 5400, 7]`
- `token_id_tensor: [B, 5400, 6]`
- `derived_value_tensor: [B, 13, 1]`
- `derived_id_tensor: [B, 13]`
- `policy_state_tensor: [B, 1, 9]`

Raw feature slot semantics in `raw_field_tensor[..., :, 6]`:

- `value`
- `log1p_age`
- `padding`
- `unavailable`
- `missing`
- `stale`

`field_presence_mask` is required so that stream-local field counts do not collapse into numeric zero.

Default `Phase 1B` model envelope:

- `d_model = 192`
- `n_heads = 8`
- `n_layers = 6`
- `ffn_dim = 768`
- `dtype = bf16`

### Factorized attention plan

The approved `Phase 1B` encoder uses factorized attention, not full global raw-token attention.

Stages:

1. stream-local field encoder
2. temporal attention over `(symbol, exchange, stream, scale)`
3. cross-sectional attention over `(scale, bucket, stream)`
4. stream-mixing attention over `(scale, bucket, symbol, exchange)`
5. stream collapse into entity-time tokens
6. target-conditioned readout over entity-time tokens plus derived tokens plus one policy-state token

Complexity:

- full raw-token global attention score surface per head/layer would be `5414^2 = 29,311,396`
- approved factorized plan target is approximately `240,494` score elements per head/layer/sample
- this is roughly `122x` lower than naive full global attention on the raw token set

Single high-end GPU target envelope:

- train peak memory: `< 16 GB`
- preferred train peak: `12–14 GB`
- inference peak memory: `< 2.5 GB`

### Dead-space interpretation

The redesign plan treats dead space in two categories:

- `structural sparsity`: code-proven padding created by the shared `field_total` packing layout
- `empirical sparsity`: actual always-zero or near-zero occupancy in real tensor-cache shards

Only the first is already proven from code.
The second must be measured from retained tensor shards before architecture work begins.

## Phase plan

## Phase 0 - Instrumentation and truth extraction

Purpose:

- quantify structural sparsity,
- measure empirical sparsity from actual shards,
- expose state-conditioned behavior diagnostics,
- measure the gap between flat greedy labels and state-aware labels,
- produce the evidence needed to decide whether `Phase 1A` is sufficient or `Phase 1B` is mandatory.

Required outputs:

- structural sparsity summary
- empirical sparsity summary
- per-segment feature occupancy summary
- action and venue label histograms
- policy-state histograms
- evaluation behavior diagnostics

## Phase 1A - Semantics/objective redesign on current surface

`Phase 1A` keeps the current flattened observation surface.

Current execution boundary:

- Phase 1A is carried by `linear-policy-v2` on the flat surface
- config-level base actions stay generic: `abstain`, `hold`, `exit`, `enter_long`, `enter_short`
- internal training/runtime vocabulary is the 9-logit venue-expanded joint action vocabulary
- `PolicyRuntimeBridge.decide()` stays backward-compatible for `linear-policy-v1` via `policy_state=None`
- shared reward / evaluation / runtime semantics are gated by `action_space_version`
- Phase 1A evidence must come from a `fresh full same-root run` or a `payload-complete same-root run`
- the slim blocker bundle at `outputs/ql031-same-root-proof-20260419` is not a runnable Phase 1A payload source

It changes only:

- the action contract,
- invalid action masking,
- state-aware label generation,
- bootstrap training objective,
- explicit `policy_state` consumption on the flat path.

Approved bootstrap oracle for `Phase 1A`:

- exact horizon: `H_bootstrap = 4`
- objective: undiscounted 4-step cumulative post-cost net reward
- split boundary: only train/validation
- leakage guard: if the full 4-row local horizon `t..t+3` is not available inside the same split and same trajectory chunk, the bootstrap supervised label is masked out
- unavailable venues and infeasible actions are masked before the oracle solves the local decision problem

Approved `Phase 1A` bootstrap loss:

- masked joint-action cross-entropy against the `H=4` oracle
- auxiliary value regression to the oracle 4-step return

Research stage-gate, not promotion gate:

- same-root untouched-final `total_net_return >= -0.60`
- `trade_rate <= 0.65`
- `gross_directional_pnl > 0.15`
- `fee_slippage_burden <= 0.75`
- `mean_dwell_steps >= 2.0`
- `flip_rate <= 0.20`
- `venue_switch_rate <= 0.08`
- peak GPU memory `< 10 GB`
- wall-clock per candidate `<= 45 min`
- batch-1 p95 inference latency `<= 2 ms`
- artifact size `<= 10 MB`

These thresholds are only evidence that `Phase 1A` is materially different from the current path.
They do not authorize promotion.

## Phase 1B - Structured encoder redesign

`Phase 1B` keeps `Phase 1A` action semantics, invalid mask semantics, policy-state semantics, and bootstrap objective fixed.

It changes only:

- the encoder,
- the feature packing interface,
- the runtime payload format needed to carry the structured model.

Research stage-gate, not promotion gate:

- same-root untouched-final `total_net_return >= -0.25`
- `Phase 1B - Phase 1A total_net_return delta >= +0.20`
- `trade_rate <= 0.45`
- `gross_directional_pnl > 0.35`
- `fee_slippage_burden <= 0.50`
- `mean_dwell_steps >= 4.0`
- `flip_rate <= 0.12`
- `venue_switch_rate <= 0.05`
- peak GPU memory `< 16 GB`
- wall-clock per candidate `<= 90 min`
- batch-1 p95 inference latency `<= 5 ms`
- artifact size `<= 32 MB`

Required secondary retained-surface smoke test:

- at least one secondary retained-surface smoke evaluation must run outside the same-root `2026-01-25` surface
- the smoke test is a retained-surface sanity check only, not a promotion surface
- minimum smoke threshold: `total_net_return >= -0.15` on the secondary retained surface used for the check

These thresholds are only evidence that the structured encoder is buying something beyond the semantics/objective redesign.
They do not authorize promotion.

## Phase 2 - Ceiling-seeking economics-aligned path

`Phase 2` is not automatically in scope for the current hardening pass.

It becomes active only if:

- `Phase 1B` still shows evidence that intra-bucket event order is the remaining hard blocker, and
- raw event data needed for event-native tokenization is actually available in an implementation-safe way.

Approved ceiling objective:

- truncated economics-aligned fine-tuning horizon: `H_ceiling = 16`
- masked policy optimization on the deterministic replay surface
- invalid actions remain masked at every step
- bootstrap cross-entropy is warm-start only; it is not the final training objective

## Phase 2 feasibility gate

Current retained snapshot artifacts are not sufficient for event-native modeling.

Why:

- the current builder emits latest-known snapshot buckets, not bounded event microsequences
- current tensor-cache artifacts carry flat features and replay rows, not event-native token windows

Phase 2 therefore requires one of:

- a new event-token cache built from raw normalized event sources during trajectory build, or
- a trajectory artifact extension that carries bounded intra-bucket event windows

This is `offline expansion` work and maps naturally to `QL-033`, not automatic `QL-031` scope.

## Runtime and export plan

`V1` and `V2` must coexist.

Rules:

- no in-place replacement of `linear-policy-v1`
- `linear-policy-v1` remains baseline and rollback anchor
- `V2` artifacts are explicit and separately versioned
- runtime must dispatch by artifact-declared adapter, not by guesswork

Approved `V2` runtime/export shape:

- keep the outer `PolicyArtifact` shell
- add `runtime_contract_v2`
- add explicit `action_vocabulary`
- add explicit `token_spec`
- add explicit `policy_state_requirements`
- add explicit token-count expectations
- add serialization digests for parity checks

Approved serialization family:

- single-file artifact
- config + token spec + normalization stats + weights + vocab inside one serialized payload

Deterministic parity checks are mandatory:

- observation -> tokenization parity
- serialized -> reloaded artifact parity
- evaluation-path vs runtime-path identical logits or identical decisions on the same input
- `V1` behavior unchanged

## Promotion rule

Nothing in this document weakens promotion discipline.

Promotion still requires all current governing conditions, including:

- same-surface comparison against the current champion when a champion exists
- complete search-budget accounting
- post-cost-positive governing economics
- reproducibility within acceptable tolerance
- paper/sim linkage where required

This redesign plan defines research and implementation sequencing only.
