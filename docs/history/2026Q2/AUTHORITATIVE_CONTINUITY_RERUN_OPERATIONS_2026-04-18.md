# Authoritative Continuity Rerun Operations 2026-04-18

**Status:** complete  
**Scope:** operational record for the authoritative rerun at `/workspace/runs/ql016-ql004-authoritative-20260418`

This document preserves the real operational issues, fixes, and preflight rules observed while running and retaining the authoritative continuity rerun that closed `QL-016` and `QL-004`.

It is a durable operator note.
It does not change runtime behavior, market scope, parity semantics, or continuity authority rules.

## Verified run facts

- authoritative run root: `/workspace/runs/ql016-ql004-authoritative-20260418`
- retained minimum bundle: `outputs/ql016-ql004-authoritative-minimum-20260418`
- retained manifest: `outputs/ql016-ql004-authoritative-minimum-20260418/bundle_manifest.json`
- retained checksums: `outputs/ql016-ql004-authoritative-minimum-20260418/SHA256SUMS`
- retained bundle disk usage: `192M`
- retained bundle unique bytes: `190676858`
- source repo commit SHA recorded in the retained manifest: `8aabc6a39056e691d70b7ce517e4c7f472488bdd`
- GPU observed in `train.log`: `NVIDIA GeForce RTX 5090`
- training summary: `training_backend=pytorch`, `training_device=cuda`, `tensor_cache_used=true`, `jsonl_fallback_used=false`
- authority summary: `eligible_external_candidate_count=1`, `authority_status=confirmed`

Retained-copy honesty:

- the authoritative source was the external run root and its active registry scope
- the retained-local bundle is a preserved copy derived from that run
- the retained-local bundle is not relabeled as `authoritative_evidence`

## Operational planning minimum

Exact operational requirement for the current one-day rerun shape:

- plan for `~150 GB` free disk inside the remote instance before `build-trajectories` starts

Why:

- `trajectories/development.jsonl` reached `21G`
- `trajectories/train.jsonl` reached `17G`
- `trajectories/validation.jsonl` reached `4.1G`
- `trajectories/final_untouched_test.jsonl` reached `4.1G`
- `trajectories/tensor_cache_v1/` reached `67G`
- registry, policy export, evaluation export, score export, logs, repo sync, venv, and shutdown-time copy work need extra headroom

Operator rule:

- request `250 GB` instance disk for this workflow

Do not confuse:

- remote run disk requirement: `~150 GB` free space minimum
- retained bundle size after shutdown: `192M`

## Encountered issues

| Encountered issue | Symptom | Root cause | Fix / workaround used in this run | Future prevention |
| --- | --- | --- | --- | --- |
| Remote workspace was not a Git checkout | `git rev-parse HEAD` on the remote workspace failed with `fatal: not a git repository` | The repo was synced onto the instance as an execution copy without `.git` metadata | The source commit SHA was taken from the local repo and written into the retained bundle manifest | Record the local commit SHA before sync or before shutdown. Do not rely on remote Git metadata unless the instance is a real clone. |
| Overlapping `rsync` of the same large artifact | Multiple `rsync`/`ssh` processes were copying `policy.json` at the same time and the retained bundle showed a growing hidden temp file | The same copy path was restarted before the previous transfer finished | Duplicate `rsync` processes were terminated and only one transfer was allowed to complete | Copy each large artifact exactly once. Before retrying, check running `rsync` processes and the destination directory for partial temp files. |
| `rsync` temp-file staging looked like a missing artifact | The retained bundle briefly showed hidden files such as `.policy.json.*` and `.inference_artifact.json.*` while the final target file was absent | `rsync` writes a temporary file until the transfer finishes and then renames it into place | The transfer was allowed to finish, then leftover hidden temp files were removed and the final file plus checksums were verified | Treat hidden `.<name>.*` files as in-progress transfer state, not as evidence completion. Do not shut the instance down until the final filenames exist and checksums have been written. |
| Zsh cleanup failed on an unmatched glob | Cleanup returned `zsh: no matches found: .../.inference_artifact.json.*` | Zsh expands unmatched globs as an error by default | Cleanup was rerun with `setopt null_glob` so absent temp files no longer aborted the command | Use `setopt null_glob` before wildcard cleanup in zsh or avoid bare unmatched globs entirely. |

## Preflight requirements

These were required for this run.
They are not recorded here as encountered failures unless the table above says so.

- Rent a fresh external single-GPU instance with direct SSH and on-demand lifetime.
- Request `250 GB` disk so the run still has more than the `~150 GB` free-space minimum after repo sync and environment setup.
- Use the provider-issued SSH host and high port exactly as assigned for that instance.
- Pin the SSH key path and force identity selection in every `ssh` and `rsync` command.
- Keep the authoritative run root outside repo-local `outputs/`.
- Create the registry directory explicitly before the chain starts: `mkdir -p "$RUN_ROOT/registry"`.
- Run `inspect-s3-compact` first and require `matched_partition_count > 0` before building trajectories.
- Capture the local source commit SHA before shutdown because the remote execution copy may not contain `.git`.

Verified command shapes from this run:

```bash
ssh -i ~/.ssh/quantlab_hetzner -o IdentitiesOnly=yes -p 11422 root@ssh1.vast.ai
```

```bash
ssh -i ~/.ssh/quantlab_hetzner -o IdentitiesOnly=yes -p 11422 root@ssh1.vast.ai \
  'tail -f /workspace/runs/ql016-ql004-authoritative-20260418/build.log'
```

```bash
rsync -az -e 'ssh -i ~/.ssh/quantlab_hetzner -o IdentitiesOnly=yes -p 11422' \
  root@ssh1.vast.ai:/workspace/runs/ql016-ql004-authoritative-20260418/policy.json \
  outputs/ql016-ql004-authoritative-minimum-20260418/
```

## What to retain before shutdown

Retain the minimum evidence bundle that preserves the authoritative closeout decision path:

- `continuity_audit_authoritative.json`
- `continuity_authority_discovery.json`
- `inspect_s3.json`
- `policy.json`
- `evaluation.json`
- `score.json`
- `inference_artifact.json`
- `trajectories/manifest.json`
- `trajectories/tensor_cache_v1/tensor_cache_manifest.json`
- `registry/index.json`
- active `registry/records/*`
- active `registry/evaluations/*`
- active `registry/scores/*`
- active `registry/artifacts/*` with duplicate bytes avoided when a hardlink is enough
- `build.log`, `train.log`, `evaluate.log`, `score.log`, `export.log`
- `build.exit`, `train.exit`, `evaluate.exit`, `score.exit`, `export.exit`
- exact copies of the configs used for the run
- a retained manifest with source commit SHA, run root, timestamps, training summary, and authority summary
- a retained checksum file such as `SHA256SUMS`

Integrity steps used in this run:

- final file set was recorded in `bundle_manifest.json`
- SHA-256 hashes were written to `SHA256SUMS`
- `registry/artifacts/policy-fd389f520ad3.json` was stored as a hardlink to `policy.json` to avoid duplicate bytes
- stage exit files were preserved and remained `0`

## What not to copy

Do not copy these large surfaces into the retained bundle:

- raw market data
- `trajectories/development.jsonl`
- `trajectories/train.jsonl`
- `trajectories/validation.jsonl`
- `trajectories/final_untouched_test.jsonl`
- full `trajectories/tensor_cache_v1/` shard payload
- temporary transfer files
- duplicate large artifact bytes with no extra decision value

The retained bundle is for future truth, audit, closeout, and docs verification.
It is not a full rerun reconstruction pack.

## How to use this note later

Use this document in three places:

- future authoritative reruns: follow the preflight rules and avoid the same shutdown-time copy mistakes
- future retained-bundle work: use the retention list and checksum rule exactly, keep the retained-copy honesty language unchanged
- future closeout verification: pair this note with `docs/history/2026Q2/AUTHORITATIVE_CONTINUITY_RERUN_2026-04-18.md`, the retained `bundle_manifest.json`, and `SHA256SUMS`

## Not verified in this run

These items were not observed as failures in this run and are therefore not recorded as encountered issues:

- provider preemption during execution
- SSH permission-denied recovery
- checkpoint/resume behavior after instance loss
