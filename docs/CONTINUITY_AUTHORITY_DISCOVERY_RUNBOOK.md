---
status: operational
owner: quantlab
last_reviewed: 2026-04-18
read_when:
  - before_treating_any_external_registry_root_as_authoritative_evidence
  - before_running_authoritative_continuity_reruns
supersedes: []
superseded_by: []
---

# Continuity Authority Discovery Runbook

## Purpose

This runbook defines the conditional discovery procedure for external active registry roots that may qualify for continuity closeout.

It exists to prevent a false-positive authority path:

- repo-local retained bundles under `outputs/` may be readable and even `clear_in_inspected_scope`
- that still does not make them authoritative evidence
- authoritative reruns are allowed only after discovery narrows the external candidate set to exactly one eligible external root

The internal helper for this procedure is:

```bash
./.venv/bin/python scripts/discover_continuity_authority.py
```

That helper emits discovery summaries only. It must not run `authoritative-evidence` reruns and it must not grant authority on its own.

Workspace stop rule:

- if historical remote roots are gone or inaccessible in the current workspace
- and no concrete external root is already present here
- stop the local discovery loop
- leave continuity closeout pending until fresh authoritative evidence exists

This runbook is therefore a conditional reference surface, not the active next operational task in every workspace.

---

## Search order

Search roots must be evaluated in this order:

1. operator-supplied roots
2. `/workspace/runs/*/registry`
3. `/root/runs/*/registry`

The helper accepts repeatable operator roots:

```bash
./.venv/bin/python scripts/discover_continuity_authority.py \
  --search-root <root-a> \
  --search-root <root-b> \
  --output outputs/analysis/continuity-authority-discovery-<date>.json
```

Operator roots may point at:

- a base runs directory
- a specific run directory
- a specific `registry` directory

Repo `outputs/` roots may be searched to sharpen blockers, but they are never eligible for authority confirmation.

Do not keep adding speculative search roots in a workspace where the historical remote environment is no longer available.

---

## Required candidate summary fields

For each discovered `<candidate>/registry`, capture:

- `audit_scope_verdict`
- `active_record_count`
- `readable_active_artifact_count`
- `artifact_load_failures`
- `active_training_backend_counts`
- `active_legacy_compat_artifact_count`
- `registry_local_fallback_policy_ids`
- `latest_record_updated_at`
- sibling run root
- any observed `build/train/evaluate/score/export` exit codes

The helper is allowed to summarize those fields using inspected-scope audit semantics only.

---

## Candidate classification

Only these classifications are valid:

- `eligible_for_authority_confirmation`
- `not_eligible`
- `retained_bundle_only`

Rules:

- `eligible_for_authority_confirmation`
  - candidate is external to the repo `outputs/` retained-bundle surface
  - `active_record_count > 0`
  - `readable_active_artifact_count == active_record_count`
  - `artifact_load_failures = []`
  - any observed stage exit files are all `0`
- `not_eligible`
  - zero active records, unreadable artifacts, nonzero stage exits, invalid exit files, or broken root
- `retained_bundle_only`
  - candidate lives under repo `outputs/.../registry`
  - it may still provide inspected-scope truth, but it is not an authority candidate

`retained_bundle_only` takes precedence over eligibility. A repo-local retained bundle must stay non-authoritative even if every artifact is readable and every stage exit file is `0`.

---

## Decision rule

- `0 eligible external candidates => blocked`
- `>1 eligible external candidates => blocked as ambiguity`
- `1 eligible external candidate => authority confirmation step allowed`

Allowed next step only for the third case:

```bash
quantlab-ml audit-continuity \
  --registry-root <confirmed-external-root> \
  --inspected-evidence-kind authoritative-evidence
```

If the helper returns `0` or `>1` eligible external candidates, stop there. Document the blocker and keep the closeout records `pending_authoritative_evidence`.

If the blocked result reflects a workspace where historical roots are no longer accessible, close the local loop and wait for fresh authoritative evidence rather than repeating the same discovery attempt.

---

## Documentation output

Every authority discovery batch must produce:

- one discovery summary JSON, typically under `outputs/analysis/`
- one repo-tracked historical record under `docs/history/2026Q2/`

The historical record must include:

- searched roots
- discovered candidates
- chosen candidate or blocker
- discovery summary path
- authoritative rerun command/output
- sprint closeout sentence

That historical record is the repo-tracked authority-discovery evidence surface for the batch.

---

## Guardrail

Do not promote a retained bundle to `authoritative_evidence` just because:

- the bundle is relocation-safe
- `audit_scope_verdict=clear_in_inspected_scope`
- local fallback artifacts are readable
- stage exit files are all `0`

Without a separately confirmed external active registry root, continuity closeout remains pending.
