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

`QL-016` and `QL-004` were closed on 2026-04-18 when fresh authoritative evidence was produced by the controlled rerun at `/workspace/runs/ql016-ql004-authoritative-20260418`.

Result:

- the external active registry root is confirmed and unique for the closeout scope
- `audit-continuity` returned `authority_status=confirmed` and `audit_scope_verdict=clear_in_inspected_scope`
- NumPy continuity is retired
- legacy `linear-policy-v1` compat continuity is retired
- see `docs/history/2026Q2/AUTHORITATIVE_CONTINUITY_RERUN_2026-04-18.md`

### QL-031

- label: `OFFLINE_HARDENING`
- status: `in_progress`
- title: Broaden offline closure evidence beyond the single controlled proof surface
- why_open:
  - this is now the single active next batch in this workspace because historical local authority discovery is closed here
  - current HEAD now has a repo-tracked minimum evidence pack that indexes one inspected-scope continuity audit and one same-surface current-head retained-run comparison
  - current HEAD still contains only narrow same-surface proof; broader multi-window or multi-slice evidence remains incomplete
  - `comparison_report_id`, paper/sim linkage, and champion-backed comparison surfaces remain missing on the retained proof surfaces currently available
- next_action:
  - expand beyond the current same-surface proof using currently available retained surfaces only
  - produce broader multi-window or multi-slice offline evidence without claiming authoritative continuity closure
  - improve comparison-report and paper/sim linkage honesty on the retained proof surfaces that already exist
- evidence_needed:
  - multi-window or multi-slice offline evaluation packs
  - current-head champion/challenger comparison surfaces
  - comparison-report and paper/sim linkage for any surface that is used to argue promotion readiness
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

Live enablement is the next planned build direction, but it is not the current main focus while `DOCS_TRUTH` and broader `OFFLINE_HARDENING` evidence items remain open. In this workspace, `QL-031` is the single active next batch now that authoritative continuity closure for `QL-016` and `QL-004` is complete.
