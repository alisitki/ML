# Backlog

## Purpose

This file is the current execution queue.

Use these labels to read ordering quickly:

- `DOCS_TRUTH`
- `OFFLINE_HARDENING`
- `OFFLINE_EXPANSION`
- `LIVE_ENABLEMENT`

Status values:

- `todo`
- `in_progress`
- `blocked`

---

## DOCS_TRUTH

### QL-014

- label: `DOCS_TRUTH`
- status: `in_progress`
- title: Keep repo-truth surfaces phase-honest
- why_open:
  - current HEAD is stronger on offline substance than on closure honesty
  - state, roadmap, backlog, and runbooks must keep current capability separate from target-state architecture
  - repo-tracked artifact, external retained evidence, and authoritative evidence must not read like the same thing
- done_when:
  - `README.md`, `PROJECT_STATE.md`, `ROADMAP.md`, `BACKLOG.md`, and `DOCS_INDEX.md` agree
  - offline closure criteria, continuity audit method, and closeout record format are explicit
  - live/runtime work remains visible without reading as the current main focus

---

## OFFLINE_HARDENING

### QL-016

- label: `OFFLINE_HARDENING`
- status: `blocked`
- title: Close the remaining NumPy / legacy continuity inventory
- why_open:
  - PyTorch is the active core-training path, but final retirement or freeze of continuity windows depends on authoritative registry truth
  - the authority-aware audit surface exists, but authoritative evidence for the active registry scope is still missing
  - external retained bundles can contain copied registry JSON with non-local artifact paths, which is not retirement-grade evidence by itself
- next_action:
  - identify the authoritative registry root(s) or a relocation-safe retained bundle
  - rerun `quantlab-ml audit-continuity --registry-root ...` with explicit evidence classification when needed
  - update the repo-tracked closeout records only after the authoritative scope is confirmed
- evidence_needed:
  - authoritative registry root confirmation
  - readable retained artifact paths for every active record in scope
- done_when:
  - authoritative evidence is confirmed for the active scope
  - zero active dependency is proven on that scope, or the remaining dependency is explicitly frozen with written scope

### QL-004

- label: `OFFLINE_HARDENING`
- status: `blocked`
- title: Close the remaining observation-schema continuity debt
- why_open:
  - the temporary legacy compat window still exists for deterministic legacy `linear-policy-v1` artifacts
  - whether it can be retired depends on the same authoritative continuity truth as QL-016
  - the repo-tracked closeout record should stay `pending_authoritative_evidence` until that truth exists
- next_action:
  - keep the closeout record pending until QL-016 confirms authoritative evidence
  - only then decide whether the window should be retired, frozen, or kept temporary with explicit scope
- evidence_needed:
  - authoritative continuity audit result
- done_when:
  - the compat-window closeout record moves from `pending_authoritative_evidence` to `decided`

### QL-031

- label: `OFFLINE_HARDENING`
- status: `blocked`
- title: Broaden offline closure evidence beyond the single controlled proof surface
- why_open:
  - current HEAD contains real controlled-proof evidence, but that is not the same thing as professional offline closure
  - broader multi-window and champion/challenger evidence remains incomplete
  - this remains outside the minimum closure-truth batch because current HEAD only has `comparison_report_id` placeholder linkage, not a narrow comparison-report artifact surface
- evidence_needed:
  - multi-window or multi-slice offline evaluation packs
  - current-head champion/challenger comparison surfaces
- done_when:
  - offline closure criteria move from `PARTIAL` to `PASS` on evidence areas that are currently unproven

---

## OFFLINE_EXPANSION

### QL-032

- label: `OFFLINE_EXPANSION`
- status: `blocked`
- title: Check whether checkpoint/resume is a genuinely small follow-on hardening task
- why_open:
  - interruption tolerance matters for larger retained offline runs
  - current batch should not half-build checkpointing without a clean seam
- next_action:
  - only implement if a narrow, testable path is already near completion
  - otherwise leave it explicitly blocked with acceptance criteria
- done_when:
  - checkpoint/resume is either implemented cleanly with tests or intentionally deferred with clear scope

### QL-033

- label: `OFFLINE_EXPANSION`
- status: `blocked`
- title: Expand research/evidence surface without weakening closure discipline
- why_open:
  - search, architecture, and evaluation expansion are valid later offline work
  - they should not be used to paper over unresolved continuity or truth gaps
- evidence_needed:
  - explicit acceptance criteria per expansion item
- done_when:
  - expansion work is scoped as additive offline research rather than fake closure

---

## LIVE_ENABLEMENT

These items stay visible, but they are later-phase work.

### QL-024

- label: `LIVE_ENABLEMENT`
- status: `todo`
- title: Build websocket ingestion and canonical event normalization services

### QL-025

- label: `LIVE_ENABLEMENT`
- status: `todo`
- title: Build the online state / feature service

### QL-026

- label: `LIVE_ENABLEMENT`
- status: `todo`
- title: Implement replay-vs-live parity, degraded-input, and recovery tooling

### QL-027

- label: `LIVE_ENABLEMENT`
- status: `todo`
- title: Stand up the selector runtime daemon

### QL-028

- label: `LIVE_ENABLEMENT`
- status: `todo`
- title: Integrate the thin executor and live controls

### QL-029

- label: `LIVE_ENABLEMENT`
- status: `todo`
- title: Build the shadow/paper operating loop

### QL-030

- label: `LIVE_ENABLEMENT`
- status: `todo`
- title: Automate commercialization-gate evidence and reporting

### QL-100

- label: `LIVE_ENABLEMENT`
- status: `todo`
- title: ONNX / TensorRT runtime acceleration exploration

### QL-101

- label: `LIVE_ENABLEMENT`
- status: `todo`
- title: Reward v2 path-aware redesign

---

## Ordering note

Live enablement is the next planned build direction, but it is not the current main focus while `DOCS_TRUTH` and `OFFLINE_HARDENING` items remain blocked or ambiguous.
