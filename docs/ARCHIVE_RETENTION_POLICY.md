---
status: canonical
owner: quantlab
last_reviewed: 2026-04-23
read_when:
  - before_remote_runs
  - before_local_output_cleanup
supersedes: []
superseded_by: []
---

# Archive Retention Policy

## Purpose

This policy defines how QuantLab stores and prunes heavy operational artifacts.

It is operational hardening, not model work, Phase C, or selector redesign. It does
not change market scope, observation semantics, runtime behavior, promotion status, or
live-readiness interpretation.

## Remote-first default

Heavy work defaults to remote compute:

- heavy proof builds
- heavy artifact validation
- same-root reruns
- remote smoke runs
- search runs

The local workstation is for code, docs, small reports, thin receipts, manifests, and
summaries. It is not the primary home for completed heavy proof roots.

## Canonical archive home

The canonical heavy artifact archive is:

```text
s3://quantlab-archive/quantlab/...
```

Prefix layout:

- local ignored output roots: `s3://quantlab-archive/quantlab/local-outputs/...`
- remote run roots: `s3://quantlab-archive/quantlab/remote-runs/...`

Use source-path-oriented prefixes so an archived object can be traced back to the
operator-visible source root.

## Allowed roots

Archive and prune tooling may operate only on:

- untracked heavy roots under repo-local `outputs/`
- explicit remote run roots under `/workspace/runs/...` or `/root/runs/...` when
  running on the remote host

Everything else is outside the tooling boundary.

## Hard denylist

The following must never enter archive or prune flow:

- `.env`
- SSH keys and SSH config material, including `.ssh`, `id_*`, `*.pem`, and `*.key`
- `.venv`
- `.git`
- repo-tracked source, docs, configs, tests, scripts, and metadata
- personal or local config caches, including `.aws`, `.config`, `.cache`,
  `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, `__pycache__`, and `.DS_Store`

If a candidate root contains denylisted material, the tool must fail closed for that
root. The operator should remove or isolate the denied material outside the archive
flow, then rerun dry-run inventory.

## Required sequence

1. Verify `s3://quantlab-archive` credentials using a non-mutating bucket/list check.
   Dedicated `S3_ARCHIVE_*` credentials are preferred. If they are absent, shared
   `S3_COMPACT_*` credentials from the existing repo S3 path are allowed only when
   they verify successfully against `quantlab-archive`.
2. Produce inventory dry-run only.
3. Review every candidate root with:
   - source path
   - size
   - retained class
   - replayable yes/no
   - proposed S3 destination prefix
   - proposed local thin mirror contents
   - proposed prune list summary
   - blocked or denylisted entries
4. Stop for operator review.
5. Upload only with explicit `archive_run_bundle.py --execute`.
6. Verify receipts and checksum manifests.
7. Produce prune dry-run from verified receipts.
8. Stop again if the prune plan differs from the reviewed inventory.
9. Prune only with explicit `prune_local_outputs.py --execute`.

## Receipt requirements

Every archived run or proof root needs an explicit receipt with:

- source root
- archive destination prefix
- timestamp
- file inventory
- SHA256 or equivalent digest manifest
- retained class: `full`, `slim`, or `partial`
- replayable yes/no
- what was kept locally
- what was pruned locally
- what was pruned remotely

Pruning requires a verified receipt. A successful upload without verification is not a
deletion permit.

## Retained classes

- `full`: replayable payloads remain present in the archived root.
- `slim`: non-replayable proof/control surface with only decision evidence,
  manifests, summaries, logs, and receipts.
- `partial`: incomplete or interrupted run state that may still be useful for
  diagnostics but cannot be treated as full closure evidence.

Retained class does not change evidence authority. Repo-local retained bundles remain
external retained evidence unless a separate authoritative external active root is
confirmed under continuity rules.

## Thin local mirrors

After verified archive and approved prune, local mirrors should keep only:

- archive receipts and prune receipts
- manifest and checksum files
- summary JSON/Markdown
- logs and exit files
- small configs and small evidence files
- final reports needed for current docs or blocker interpretation

Large replay payloads, tensor/event cache shard files, large policy copies, duplicate
registry artifact payloads, and proof-slice data payloads should not remain local unless
explicitly pinned.

When cache payloads are pruned locally, canonical cache manifests that would become
dangling must be replaced by summary artifacts rather than left as replayable-looking
manifests.

## Supported tooling

Primary operator scripts:

- `scripts/archive_run_bundle.py`
- `scripts/prune_local_outputs.py`

Dry-run is the default mode. `--execute` is required for upload or deletion.

Remote pruning is supported safely only when the same verified-receipt rules are used
on an explicit remote root under `/workspace/runs/...` or `/root/runs/...`.
