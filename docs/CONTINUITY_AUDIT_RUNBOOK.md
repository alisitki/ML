---
status: operational
owner: quantlab
last_reviewed: 2026-04-18
read_when:
  - before_closing_continuity_windows
  - before_treating_registry_audit_as_retirement_evidence
supersedes: []
superseded_by: []
---

# Continuity Audit Runbook

## Purpose

This runbook defines the procedure for auditing temporary continuity windows such as the NumPy training backend and legacy runtime-compat artifacts.

The audit output is scope-limited. It can tell you what is true in the inspected registry scope. It does not, by itself, prove that the inspected scope is authoritative.

---

## Inputs

Required:

- inspected registry root
- readable registry records for the scope under review

Preferred:

- the authoritative active registry root
- or a relocation-safe retained bundle that includes readable local artifact copies under `registry/artifacts/`

Command:

```bash
quantlab-ml audit-continuity --registry-root <registry-root>
```

---

## Output fields to read first

- `audit_scope_verdict`
- `blocking_reasons`
- `artifact_load_failures`
- `active_training_backend_counts`
- `active_legacy_compat_artifact_count`
- `ready_to_close_numpy_continuity_window`
- `ready_to_retire_legacy_compat_window`

Interpret the booleans only after confirming the scope verdict is not `blocked`.

---

## Scope verdicts

- `blocked`: the audit cannot support a continuity decision on this scope
- `active_dependency_present`: active records were readable and at least one continuity dependency is still present
- `clear_in_inspected_scope`: active records were readable and the inspected scope shows no remaining dependency for the audited windows

`clear_in_inspected_scope` is still not global retirement proof unless the inspected root is confirmed to be authoritative.

---

## Acceptable continuity decisions

Only these decisions are acceptable:

- `RETIRE`
- `FREEZE`
- `KEEP-TEMPORARY-WITH-EXPLICIT-SCOPE`

Decision rules:

- `RETIRE`: only when the authoritative registry scope is clear and the remaining continuity window is no longer needed
- `FREEZE`: when active dependency is clear in authoritative scope, but removal is intentionally deferred and tightly scoped
- `KEEP-TEMPORARY-WITH-EXPLICIT-SCOPE`: when the audit is blocked, incomplete, or still shows active dependency

Anything that looks like "probably safe" without authoritative evidence is `KEEP-TEMPORARY-WITH-EXPLICIT-SCOPE`.

---

## Blocked cases

### Zero active records

If the registry has zero active records, the audit is `blocked`.

Why:

- zero active records does not prove that the inspected root is the full active scope
- zero active records must not look like successful retirement

Required action:

- confirm the authoritative registry root
- rerun the audit on that scope

### Broken or non-relocatable retained artifact paths

If registry records point at unreadable paths, the audit is `blocked`.

Examples:

- copied records still point at `/root/runs/...`
- retained bundle omitted the local artifact copy
- the recorded artifact path is stale or broken

Required action:

- use a relocation-safe retained bundle with readable local artifacts
- or rerun against the authoritative registry root

### Authoritative registry root is unknown

If you do not know whether the inspected root is authoritative, treat the result as insufficient for retirement.

Required action:

- identify the active registry root(s)
- document why the inspected scope is authoritative before retiring or freezing compat windows

---

## Minimal retained-bundle rule

A retained local bundle is acceptable for continuity audit only if:

- the registry JSONs are present
- each active record has a readable artifact, either at the recorded path or in the local bundled `registry/artifacts/` location

A copied bundle with broken record paths but no readable local artifact is evidence for the remote run, not evidence for continuity retirement.

---

## Recommended procedure

1. Identify the registry root you believe is authoritative.
2. Run `quantlab-ml audit-continuity --registry-root <root>`.
3. Inspect `audit_scope_verdict` and `blocking_reasons` before reading the readiness booleans.
4. If the verdict is `blocked`, do not retire anything.
5. If the verdict is `active_dependency_present`, keep the window temporary and explicitly scoped.
6. If the verdict is `clear_in_inspected_scope`, confirm that the inspected root is authoritative before choosing `FREEZE` or `RETIRE`.

---

## Documentation rule

Every continuity closeout must update:

- `docs/PROJECT_STATE.md`
- `docs/BACKLOG.md`
- any affected compatibility or runtime-contract docs

If the audit is blocked, document the blocker explicitly instead of writing optimistic retirement language.
