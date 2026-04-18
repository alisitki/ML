---
status: operational
owner: quantlab
last_reviewed: 2026-04-18
read_when:
  - before_claiming_offline_closure
  - before_reprioritizing_toward_live_runtime_work
supersedes: []
superseded_by: []
---

# Offline Closure Criteria

## Purpose

This document defines what "offline professionally closed" means for QuantLab.

QuantLab is not offline professionally closed just because the training CLI runs or because a single controlled proof bundle exists. Closure requires explicit evidence across the offline system, not optimistic wording.

Keep these distinct:

- `repo-tracked artifact`
- `external retained evidence`
- `authoritative evidence`

---

## Verdict rule

Use these repo-level verdicts:

- `offline professionally closed`: every closure area below is `PASS`
- `offline operational but not professionally closed`: at least one closure area is `PARTIAL`, with no critical `FAIL`
- `offline foundation only`: one or more critical closure areas are still `FAIL`

Any `PARTIAL` or `FAIL` area blocks claims that the offline side is professionally closed.

---

## Closure areas

### 1. Data universe and representation

- `PASS`: declared exchanges, symbols, stream families, sparse availability, and unsupported/missing/stale semantics are explicit, test-covered, and consistently reflected in docs and code
- `PARTIAL`: contracts exist, but current-head truth about active scope or representation still drifts in some surfaces
- `FAIL`: current code or docs silently widen scope, collapse unsupported/missing/stale semantics, or cannot state the active universe cleanly

Required evidence:

- canonical docs aligned with current code
- tests covering declared semantics

### 2. Training engine maturity

- `PASS`: trajectory build, train, evaluate, score, and export flows are materially operational and exercised on current HEAD
- `PARTIAL`: the main offline engine works, but one or more continuity or operational edge cases still fail non-actionably
- `FAIL`: core offline commands are missing, untestable, or misleading about the active path

Required evidence:

- current-head CLI or test coverage for the main offline path
- clear distinction between smoke, continuity, and real-training modes

### 3. Evaluation and validation discipline

- `PASS`: walk-forward discipline, purge/embargo, untouched final test handling, and search-budget visibility are explicit and preserved in current behavior
- `PARTIAL`: the structure exists, but closure-grade comparison evidence is still thin or limited to a narrow proof surface
- `FAIL`: random split, leakage, or ambiguous evaluation semantics are still present on the active path

Required evidence:

- evaluation tests
- retained evaluation artifacts or reports that match declared discipline

### 4. artifact / registry / compatibility truth

- `PASS`: active registries are auditable, compatibility windows are either retired or explicitly frozen, and authoritative evidence exists for the closeout scope
- `PARTIAL`: registry and compatibility discipline exist, but authoritative continuity truth or relocation-safe retained artifacts are still incomplete
- `FAIL`: active continuity windows read like safe retirement by mistake, or registry truth cannot be audited safely

Required evidence:

- successful continuity audit on the authoritative registry scope
- explicit decision on each remaining continuity window

### 5. Reproducibility and operational proof

- `PASS`: repo-tracked artifacts or explicitly referenced external retained evidence are readable, materially complete for their claimed purpose, and tied to current-head semantics
- `PARTIAL`: retained proof exists, but some bundles are reduced, scope-limited, or not sufficient for closure-grade decisions
- `FAIL`: proof claims rely on missing repo-tracked artifacts, broken external retained evidence, or unverifiable narrative only

Required evidence:

- repo-tracked artifacts or explicitly referenced external retained evidence for claimed proof surfaces
- honest limits on what each retained bundle can and cannot prove

### 6. Docs and phase honesty

- `PASS`: README, project state, roadmap, backlog, and runbooks agree on current capability, target-state intent, and blocked items
- `PARTIAL`: most truth surfaces are aligned, but some still imply closure or next-phase focus too early
- `FAIL`: repo docs materially blur target-state architecture with current implemented reality

Required evidence:

- consistent repo-truth docs on current HEAD

---

## Current interpretation rule

This document defines closure criteria. It does not declare that current HEAD already passes them.

Use `docs/PROJECT_STATE.md` for the current verdict and `docs/BACKLOG.md` for the remaining blocked items.
