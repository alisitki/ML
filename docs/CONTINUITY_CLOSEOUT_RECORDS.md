# Continuity Closeout Records

## Purpose

This document defines the repo-tracked record format for continuity-window closeout state.

It exists so QuantLab can track closeout status for temporary continuity windows without pretending that a real closeout decision already exists.

## Required distinctions

Use these terms explicitly:

- `repo-tracked artifact`: a versioned file on current `HEAD`
- `external retained evidence`: retained material outside the repo-tracked surface, including ignored paths such as `outputs/`
- `authoritative evidence`: evidence whose active registry scope has been confirmed for closeout decisions

## Record model

Every continuity closeout record must carry:

- `window_id`
- `scope_kind`
- `authority_status`
- `latest_audit_scope_verdict`
- `blocking_reasons`
- `next_required_evidence`
- `last_reviewed`
- `decision_status`
- `decision`

`decision_status` is mandatory:

- `pending_authoritative_evidence`: the record is pre-decision and `decision` must be `null`
- `decided`: the record carries a real `RETIRE`, `FREEZE`, or `KEEP-TEMPORARY-WITH-EXPLICIT-SCOPE` decision

Real closeout decisions require authoritative evidence. A record may exist before that point, but it must remain pre-decision.

## Repo-tracked records

Current repo-tracked continuity records live under:

```text
docs/continuity_closeout/
```

Current records may be either pre-decision or decided, depending on whether authoritative evidence has been attached.

No record may claim `RETIRE` or `FREEZE` without authoritative evidence.
