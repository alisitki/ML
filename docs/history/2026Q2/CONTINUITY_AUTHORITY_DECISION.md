# Continuity Authority Decision

**Date:** 2026-04-18  
**Status:** active

## Decision

For continuity closeout, only a confirmed external active registry root may be treated as `authoritative_evidence`.

The inspected relocation-safe retained bundle at:

`outputs/ql021-acceptance-proof-20260417-no-trpro7995wx/registry`

remains `external_retained_evidence` only.

## Why

- The default configured registry root is `outputs/registry`, and that root is not present on current `HEAD`.
- The inspected retained bundle is readable and `clear_in_inspected_scope`, but inspected-scope truth is not the same thing as authoritative active-scope truth.
- Promoting the retained bundle to `authoritative_evidence` without a separately confirmed external active registry root would create a false positive closeout path.

## Consequences

- `docs/continuity_closeout/*.yaml` must remain `pending_authoritative_evidence` until an external active registry root is confirmed.
- No rerun of `audit-continuity` may use `authoritative-evidence` on the retained bundle alone.
- The retained bundle may still be cited to narrow blockers, prove relocation-safe readability, and document `clear_in_inspected_scope`.

## What Would Change This

- A real external active registry root is discovered and confirmed as the closeout scope.
- That external root can be audited with readable active records and artifacts.
- Only after that confirmation may `audit-continuity` be rerun as `authoritative-evidence`.
