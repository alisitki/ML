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

Use these terms explicitly:

- `repo-tracked artifact`: a versioned file on current `HEAD`
- `external retained evidence`: non-tracked or ignored retained material, such as a copied run bundle under `outputs/`
- `authoritative evidence`: evidence whose active registry scope has been confirmed for closeout decisions

---

## Inputs

Required:

- inspected registry root
- readable registry records for the scope under review

Preferred:

- the authoritative active registry root

Secondary inspected-scope input:

- a relocation-safe retained bundle that includes readable local artifact copies under `registry/artifacts/`

Command:

```bash
quantlab-ml audit-continuity --registry-root <registry-root>
```

Optional authority metadata:

```bash
quantlab-ml audit-continuity \
  --registry-root <registry-root> \
  --inspected-evidence-kind external-retained-evidence \
  --authority-status unknown
```

---

## Output fields to read first

- `audit_scope_verdict`
- `inspected_evidence_kind`
- `authority_status`
- `closeout_decision_allowed`
- `closeout_blockers`
- `blocking_reasons`
- `artifact_load_failures`
- `active_training_backend_counts`
- `active_legacy_compat_artifact_count`
- `ready_to_close_numpy_continuity_window`
- `ready_to_retire_legacy_compat_window`

Interpret the readiness booleans only after confirming `closeout_decision_allowed=true`.

If `closeout_decision_allowed=false`, the audit may still be useful for inspected-scope truth, but it is not sufficient for a real closeout decision.

---

## Scope verdicts

- `blocked`: the audit cannot support a continuity decision on this scope
- `active_dependency_present`: active records were readable and at least one continuity dependency is still present
- `clear_in_inspected_scope`: active records were readable and the inspected scope shows no remaining dependency for the audited windows

`clear_in_inspected_scope` is still not global retirement proof unless the inspected root is confirmed to be authoritative.

---

## Evidence-kind and authority interpretation

- `repo_tracked_artifact` typically means the inspected files are versioned on current `HEAD`, but that still does not prove they are the authoritative active scope.
- `external_retained_evidence` means the inspected files live outside the repo-tracked surface. A clean result here is still not authoritative by default.
- `authoritative_evidence` means the inspected scope has already been confirmed as the active closeout surface. Only this class may default to `authority_status=confirmed`.
- A relocation-safe retained bundle may narrow blockers or prove `clear_in_inspected_scope`, but it remains `external_retained_evidence` unless an external active registry root is separately confirmed. Absent that external root confirmation, do not relabel the retained bundle as `authoritative_evidence`.

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
- document why an external active registry root is authoritative before retiring or freezing compat windows

---

## Minimal retained-bundle rule

A retained local bundle is acceptable for continuity audit only if:

- the registry JSONs are present
- each active record has a readable artifact, either at the recorded path or in the local bundled `registry/artifacts/` location

A copied bundle with broken record paths but no readable local artifact is evidence for the remote run, not evidence for continuity retirement.

---

## Recommended procedure

1. Identify the external active registry root you believe is authoritative.
2. If that external root is unavailable, treat any retained bundle as inspected-scope truth only and keep it `external_retained_evidence`.
3. Run `quantlab-ml audit-continuity --registry-root <root>`.
4. Inspect `audit_scope_verdict` and `blocking_reasons` before reading the readiness booleans.
5. If the verdict is `blocked`, do not retire anything.
6. If the verdict is `active_dependency_present`, keep the window temporary and explicitly scoped.
7. If the verdict is `clear_in_inspected_scope`, confirm that the inspected root is the external active authoritative root before choosing `FREEZE` or `RETIRE`.

---

## Documentation rule

Every continuity closeout must update:

- `docs/PROJECT_STATE.md`
- `docs/BACKLOG.md`
- any affected compatibility or runtime-contract docs

If the audit is blocked, document the blocker explicitly instead of writing optimistic retirement language.
