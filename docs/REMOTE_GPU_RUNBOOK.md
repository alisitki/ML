# Remote GPU Runbook

## Purpose

This runbook defines the official first controlled remote-GPU training workflow for
QuantLab real training.

It exists to:
- keep the first Vast run controlled and reproducible
- keep leakage-sensitive split discipline intact
- separate repo implementation work from external continuity follow-up
- avoid treating local smoke workflows as the real-training default

## Interpretation rule

For QuantLab, real training and closure-grade retained evidence generation are remote-GPU tasks by default.

Apply this rule strictly:

- lack of local disk is not a reason to attempt the run locally
- lack of local CUDA is not a reason to attempt the run locally
- lack of local throughput is not a reason to downscope the task into a local workflow
- if the current machine cannot satisfy the run requirements, provision the remote GPU environment and run there

Local execution remains continuity-only and must not be treated as the fallback execution target for meaningful runs.

This runbook is provider-agnostic in principle. Vast.ai is used here as the concrete
example workflow because the current operating target is a rented single-GPU instance.

For the authoritative continuity rerun that closed `QL-016` and `QL-004`, see:
- `docs/history/2026Q2/AUTHORITATIVE_CONTINUITY_RERUN_2026-04-18.md`
- `docs/history/2026Q2/AUTHORITATIVE_CONTINUITY_RERUN_OPERATIONS_2026-04-18.md`

## Scope

This runbook covers:
- remote bootstrap
- preflight checks
- controlled-snapshot build/train/evaluate/score/export flow
- artifact and log collection
- first-failure triage

This runbook does not cover:
- launcher automation
- orchestration or schedulers
- reserved or interruptible policy optimization
- checkpoint/resume design
- external `audit-continuity` closure against active runtime registries
- reframing remote-GPU work as local execution because the current machine is resource-constrained

## Retained evidence honesty

This runbook defines how a controlled remote-GPU run should be executed and what evidence should be retained.

It does not, by itself, prove that any previously retained bundle in the repo is closure-grade for continuity retirement.

In particular:

- retained evidence must remain readable from the local bundle
- `acceptance_evidence.json` is an index, not the source of truth
- copied registry JSON that still points at unreadable `/root/runs/...` paths is not continuity-retirement proof by itself
- continuity closeout still depends on `docs/CONTINUITY_AUDIT_RUNBOOK.md`

## Archive-first retention rule

Remote GPU runs are remote-first for execution and archive-first for heavy artifact
storage. The canonical storage home for heavy run outputs is:

```text
s3://quantlab-archive/quantlab/...
```

Use `s3://quantlab-archive/quantlab/remote-runs/<run-id>/` for remote run roots and
`s3://quantlab-archive/quantlab/local-outputs/<relative-output-root>/` for local
ignored retained roots that must be preserved before cleanup.

Do not keep completed heavy roots indefinitely on the local workstation. Local mirrors
should be thin unless a root is explicitly pinned for active work: receipts, manifests,
checksum files, logs, configs, summaries, reports, and small evidence files only.

Archive-before-delete is mandatory. A remote or local root may be pruned only after:

- successful upload to `s3://quantlab-archive`
- checksum or receipt verification
- an archive manifest and receipt are written
- the receipt records source root, destination prefix, timestamp, file inventory,
  digest manifest, retained class, replayability, local keep/prune lists, and remote
  prune lists

Failure evidence is first-class. If a remote proof fails before completion, including
baseline build failure, selector rerun failure, build-time gate breach, skipped
determinism rerun, or analyzer failure, archive the partial `/workspace/runs/...`
root to `s3://quantlab-archive/quantlab/remote-runs/...` with logs, exit files,
partial manifests, time logs, profiling outputs, checksum manifest, and receipt.
Do not prune that remote root unless archive verification succeeds.

Hard denylist for archive and prune tooling:

- `.env`
- SSH keys and SSH config material, including `.ssh`, `id_*`, `*.pem`, and `*.key`
- `.venv`
- `.git`
- repo-tracked source, docs, configs, tests, scripts, and metadata
- local/personal caches including `.aws`, `.config`, `.cache`, `.mypy_cache`,
  `.pytest_cache`, `.ruff_cache`, `__pycache__`, and `.DS_Store`

Operator sequence:

1. verify `s3://quantlab-archive` credentials with a non-mutating check; dedicated
   `S3_ARCHIVE_*` credentials are preferred, while shared `S3_COMPACT_*` credentials
   are allowed only if they verify successfully against `quantlab-archive`
2. run archive inventory dry-run
3. review every source root, destination prefix, thin local mirror, prune summary, and
   blocked entry
4. upload only with explicit `--execute`
5. verify receipts
6. run prune dry-run from verified receipts
7. prune only with explicit `--execute`

## Controlled first-run posture

Path classification:
- `core direction`

Goals:
- prove the PyTorch path actually runs on GPU when CUDA is available
- prove the repo can build a production-profile learning surface remotely
- prove the artifact chain completes end-to-end on remote GPU

Non-goals:
- full-scale search
- throughput optimization
- promotion evidence
- paper/sim

For `QL-031` same-root proof runs, the scope is intentionally narrower than broad research but broader than the first controlled acceptance rerun:
- candidate search is allowed when it stays inside the same retained root
- promotion evidence is required for the selected same-root champion
- paper/sim linkage is required for both the same-root champion and the compared challenger

Risk posture:
- leakage tolerance remains zero
- walk-forward selection, purge, and final untouched test discipline remain unchanged
- first-run success is operational readiness evidence, not economic validation

## Controlled snapshot

Use:
- `configs/training/production.yaml`
- `configs/reward/default.yaml`
- `configs/evaluation/default.yaml`
- `configs/data/controlled-remote-day.yaml`

`configs/data/controlled-remote-day.yaml` is an example first controlled snapshot.
It pins a single full successful day that is currently readable from the compact
bucket. If that day becomes operationally unsuitable, replace all split windows
together with another single full successful day rather than widening scope.

For `QL-031` same-root proof runs:
- keep `configs/data/controlled-remote-day.yaml` as the default `2026-01-25` proof surface
- switch the training config to `configs/training/production-phase1a-flat-v2-search.yaml` for the current `Parallel V2 Phase 1A` run
- keep the run inside one external registry root until champion promotion, challenger comparison, and paper/sim linkage are all recorded
- keep every retained summary explicit that the resulting local bundle is still `external_retained_evidence`

## Vast instance guidance

Start with:
- verified or secure-cloud offer
- direct SSH
- on-demand instance
- high reliability score
- high-end CPU host with strong single-node throughput for trajectory build and tensor-cache writes

For `QL-031` same-root proof runs, treat the instance filter as performance-first rather than price-first:
- minimum GPU floor: `RTX 5090 Ti`-class single GPU or stronger
- minimum disk capacity: `500 GB`
- prefer the highest write-throughput NVMe host available; target `~10000 MB/s`-class sequential write performance when the listing exposes that signal
- if the listing does not expose an exact write-throughput figure, prefer the host with the strongest storage-performance evidence available
- prefer higher-vCPU hosts to reduce `build-trajectories`, tensor-cache write, and evaluation wall-clock time

Avoid for the first run:
- multi-GPU
- interruptible instances
- A100/H100-class cost

If the Vast listing does not literally spell the GPU as `5090 Ti`, do not step down below that floor; select the nearest equivalent or stronger single-card option and record the exact booked model in the retained manifest.

## Operational planning minimum

For the current one-day controlled rerun shape, plan for at least `~150 GB` free disk
inside the remote instance before `build-trajectories` starts.

Why this floor exists:
- `trajectories/development.jsonl` reached `21G`
- `trajectories/train.jsonl` reached `17G`
- `trajectories/validation.jsonl` reached `4.1G`
- `trajectories/final_untouched_test.jsonl` reached `4.1G`
- `trajectories/tensor_cache_v1/` reached `67G`
- registry, policy, evaluation, score, inference export, logs, and the repo checkout add more write pressure

This is an operational planning minimum, not a retained-bundle size estimate.
The retained minimum evidence bundle for the authoritative rerun was only `192M`.

Provider-level rule:
- request `500 GB` instance disk for `QL-031` same-root proof runs so the instance still has headroom for the repo, venv, search sidecars, logs, and shutdown-time evidence handling

## Bootstrap

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ml]"
```

Verify PyTorch sees CUDA before spending more time:

```bash
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA PyTorch tarafında görünmüyor"
print(torch.__version__)
print(torch.cuda.get_device_name(0))
PY
```

Secrets:
- copy `.env` onto the instance
- do not treat `VAST_API_KEY` as a repo runtime dependency; the repo does not consume it

## SSH and copy practicals

Use the provider-issued SSH host and port exactly as given for that instance.
For Vast direct SSH, pin the key path and force identity selection in every command:

```bash
ssh -i ~/.ssh/quantlab_hetzner -o IdentitiesOnly=yes -p <port> root@<host>
```

Verified example from the authoritative rerun:

```bash
ssh -i ~/.ssh/quantlab_hetzner -o IdentitiesOnly=yes -p 11422 root@ssh1.vast.ai
```

For live monitoring, `tail -f` on the remote log is sufficient:

```bash
ssh -i ~/.ssh/quantlab_hetzner -o IdentitiesOnly=yes -p <port> root@<host> \
  'tail -f /workspace/runs/<run-id>/build.log'
```

For evidence copy, reuse the same SSH options in `rsync`:

```bash
rsync -az -e 'ssh -i ~/.ssh/quantlab_hetzner -o IdentitiesOnly=yes -p <port>' \
  root@<host>:/workspace/runs/<run-id>/policy.json \
  outputs/<retained-bundle>/
```

If the remote workspace was synced without `.git`, remote `git rev-parse HEAD` is not available.
Capture the local commit SHA before sync or before instance shutdown and record it in the retained manifest.

## Preflight

Inspect compact state against the controlled snapshot before building trajectories:

```bash
export RUN_ROOT=/workspace/runs/controlled-prod-v1
mkdir -p "$RUN_ROOT/registry"

quantlab-ml inspect-s3-compact \
  --env-file .env \
  --data-config configs/data/controlled-remote-day.yaml \
  > "$RUN_ROOT/inspect_s3.json"
```

Preflight must show:
- matched partitions greater than zero
- the intended full-day coverage
- no immediate object-readability failure

## Authoritative root rule

For authoritative continuity reruns, the active run root must be outside repo-local `outputs/`.

Use an external operator-supplied path such as:

```bash
export RUN_ROOT=/workspace/runs/<authoritative-rerun-id>
mkdir -p "$RUN_ROOT/registry"
```

Do not point an authoritative rerun at repo-local `outputs/registry`.
Repo-local retained bundles remain retained copies and are not the active authoritative root.

## Official command flow

```bash
export RUN_ID="$(basename "$RUN_ROOT")"

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase BUILD_STARTED \
  --log "$RUN_ROOT/build.log" \
  --exit-file "$RUN_ROOT/build.exit" \
  -- \
  quantlab-ml build-trajectories \
    --source s3-compact \
    --s3-env-file .env \
    --data-config configs/data/controlled-remote-day.yaml \
    --training-config configs/training/production.yaml \
    --reward-config configs/reward/default.yaml \
    --output "$RUN_ROOT/trajectories"

# NOTE: --output is a directory.
# The directory will contain canonical JSONL plus a tensor_cache_v1 sidecar.
# The prod train/evaluate commands auto-detect the directory format and
# must use tensor-cache fast paths unless explicit compat fallback is requested.

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase TRAIN_STARTED \
  --log "$RUN_ROOT/train.log" \
  --exit-file "$RUN_ROOT/train.exit" \
  -- \
  quantlab-ml train \
    --trajectories "$RUN_ROOT/trajectories" \
    --training-config configs/training/production.yaml \
    --registry-root "$RUN_ROOT/registry" \
    --output "$RUN_ROOT/policy.json"

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase EVAL_STARTED \
  --log "$RUN_ROOT/evaluate.log" \
  --exit-file "$RUN_ROOT/evaluate.exit" \
  -- \
  quantlab-ml evaluate \
    --trajectories "$RUN_ROOT/trajectories" \
    --policy "$RUN_ROOT/policy.json" \
    --evaluation-config configs/evaluation/default.yaml \
    --output "$RUN_ROOT/evaluation.json"

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase SCORE_STARTED \
  --log "$RUN_ROOT/score.log" \
  --exit-file "$RUN_ROOT/score.exit" \
  -- \
  quantlab-ml score \
    --policy "$RUN_ROOT/policy.json" \
    --evaluation "$RUN_ROOT/evaluation.json" \
    --registry-root "$RUN_ROOT/registry" \
    --output "$RUN_ROOT/score.json"

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase EXPORT_STARTED \
  --log "$RUN_ROOT/export.log" \
  --exit-file "$RUN_ROOT/export.exit" \
  -- \
  quantlab-ml export-policy \
    --policy "$RUN_ROOT/policy.json" \
    --score "$RUN_ROOT/score.json" \
    --output "$RUN_ROOT/inference_artifact.json"
```

Operational requirement:
- `build.log` must be created immediately at stage launch.
- the first visible line must appear within a few seconds and begin with `[STARTED]`
- the wrapper forces unbuffered child execution and emits `[HEARTBEAT]` lines during otherwise silent phases
- stage markers such as `BUILD_STARTED`, `EVAL_STARTED`, `COMPLETED`, and `FAILED` must remain visible in the stage log

## QL-031 same-root proof chain overlay

Use this overlay only for the current `QL-031` `Parallel V2 Phase 1A` same-root run.

Command deltas:
- replace `configs/training/production.yaml` with `configs/training/production-phase1a-flat-v2-search.yaml`
- keep `RUN_ROOT` external, for example `/workspace/runs/ql031-phase1a-same-root-<date>`
- keep one registry root for the selected champion and the compared challenger
- pass `--split final_untouched_test` explicitly to both selected and challenger `evaluate` commands
- choose the challenger as the highest-ranked non-selected candidate from `policy_search.json`
- split execution into `Stage A` and `Stage B`

`H=4` wording rule for this overlay:
- `H=4` means a 4-row local oracle horizon, `t..t+3`
- labels must be masked out unless all four rows stay inside the same split and the same trajectory chunk

### Operator survivability shell baseline

Use the following shell helpers for every `QL-031` `Parallel V2 Phase 1A` remote run. These helpers are not optional policy text; they are the canonical operator path for incomplete-run retention, periodic sync, and hard abort handling.

```bash
set -euo pipefail

SSH_CMD=(ssh -i ~/.ssh/quantlab_hetzner -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -p "$SSH_PORT" "root@$SSH_HOST")
RSYNC_RSH="ssh -i ~/.ssh/quantlab_hetzner -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -p $SSH_PORT"

require_file() {
  local path="$1"
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; return 1; }
}

require_exit_zero() {
  local exit_file="$1"
  require_file "$exit_file"
  [[ "$(tr -d '[:space:]' < "$exit_file")" == "0" ]]
}

require_profile_summary() {
  local profile_path="$1"
  local key="$2"
  local expected="$3"
  python3 - <<'PY' "$profile_path" "$key" "$expected"
import json, sys
path, key, expected = sys.argv[1:4]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
value = payload["summary"].get(key)
normalized = str(value).lower() if isinstance(value, bool) else str(value)
if normalized != expected.lower():
    raise SystemExit(f"profile summary mismatch: {key}={value!r} expected {expected!r}")
PY
}

require_profile_number_max() {
  local profile_path="$1"
  local key="$2"
  local threshold="$3"
  python3 - <<'PY' "$profile_path" "$key" "$threshold"
import json, sys
path, key, threshold = sys.argv[1:4]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
value = payload["summary"].get(key)
if value is None:
    raise SystemExit(f"profile summary missing numeric field: {key}")
if float(value) > float(threshold):
    raise SystemExit(f"profile summary threshold exceeded: {key}={value} max={threshold}")
PY
}

sync_partial_bundle() {
  mkdir -p "$LOCAL_BUNDLE_ROOT"
  rsync -az -e "$RSYNC_RSH" \
    --include='*/' \
    --include='*.log' \
    --include='*.exit' \
    --include='phase1a_profile.json' \
    --include='policy.json' \
    --include='policy_search.json' \
    --include='policy_search.partial.json' \
    --include='policy_candidates_partial/***' \
    --include='checkpoints/***' \
    --include='evaluation.json' \
    --include='score.json' \
    --include='trajectories/manifest.json' \
    --include='trajectories/phase1a_supervision_v1/manifest.json' \
    --include='registry/***' \
    --exclude='*' \
    "root@$SSH_HOST:$RUN_ROOT/" "$LOCAL_BUNDLE_ROOT/"
}

retain_incomplete_bundle() {
  python /Users/stk/Desktop/ML/scripts/retain_remote_run_bundle.py \
    --allow-incomplete \
    --bundle-root "$LOCAL_BUNDLE_ROOT" \
    --source-run-root "$RUN_ROOT" \
    --source-registry-root "$RUN_ROOT/registry" \
    --instance-metadata "$LOCAL_INSTANCE_METADATA" \
    --config-copy /Users/stk/Desktop/ML/configs/data/controlled-remote-day.yaml:configs/data/controlled-remote-day.yaml \
    --config-copy /Users/stk/Desktop/ML/configs/training/production-phase1a-flat-v2-search.yaml:configs/training/production-phase1a-flat-v2-search.yaml \
    --config-copy /Users/stk/Desktop/ML/configs/reward/default.yaml:configs/reward/default.yaml \
    --config-copy /Users/stk/Desktop/ML/configs/evaluation/default.yaml:configs/evaluation/default.yaml
}

checkpoint_sync() {
  sync_partial_bundle
  retain_incomplete_bundle
  date +%s > "$LOCAL_BUNDLE_ROOT/.last_sync_epoch"
}

abort_and_retain() {
  local reason="$1"
  echo "ABORT: $reason" >&2
  checkpoint_sync || true
  exit 1
}

periodic_sync_loop() {
  while sleep 900; do
    local last=0
    [[ -f "$LOCAL_BUNDLE_ROOT/.last_sync_epoch" ]] && last="$(cat "$LOCAL_BUNDLE_ROOT/.last_sync_epoch")"
    local now
    now="$(date +%s)"
    if (( now - last >= 900 )); then
      checkpoint_sync || true
    fi
  done
}

watch_progress_and_sync() {
  "${SSH_CMD[@]}" "tail -n 0 -F '$RUN_ROOT/materialize.log' '$RUN_ROOT/train.log'" | \
  while IFS= read -r line; do
    case "$line" in
      *"[PROGRESS]"*"marker=materialization_completed"*|*"[PROGRESS]"*"marker=fold_completed"*|*"[PROGRESS]"*"marker=candidate_completed"*)
        checkpoint_sync || true
        ;;
    esac
  done
}
```

Sync/retain trigger points:
- immediately after `build` completes and `build.exit == 0`
- immediately after `materialize` completes and `materialize.exit == 0`
- immediately after any `[PROGRESS] marker=fold_completed`
- immediately after any `[PROGRESS] marker=candidate_completed`
- every `15 minutes` if no fold/candidate completion marker has arrived
- immediately after `evaluate` completion
- immediately after `score` completion
- immediately before every hard abort through `abort_and_retain`

Partial-retain integrity rule for the helper above:
- do not copy `trajectories/tensor_cache_v1/tensor_cache_manifest.json` unless the corresponding shard payloads are copied too
- the default `sync_partial_bundle` helper intentionally keeps incomplete local bundles `slim`
- if tensor-cache summary metadata is needed in a `slim` retained copy, write a non-dangling summary artifact instead of copying the canonical manifest alone

### Smallest paid remote scope first

Do not jump directly to full `Stage A`. Once local profiling gates are green, run only the smallest paid remote scope:
- full same-root surface
- `1 candidate x 1 epoch`
- selected `evaluate`
- selected `score`

```bash
set -euo pipefail

export RUN_ID="$(basename "$RUN_ROOT")"
mkdir -p "$RUN_ROOT/registry"

periodic_sync_loop &
PERIODIC_SYNC_PID=$!
watch_progress_and_sync &
PROGRESS_SYNC_PID=$!
trap 'kill "$PERIODIC_SYNC_PID" "$PROGRESS_SYNC_PID" 2>/dev/null || true' EXIT

quantlab-ml inspect-s3-compact \
  --env-file .env \
  --data-config configs/data/controlled-remote-day.yaml \
  > "$RUN_ROOT/inspect_s3.json"

python - <<'PY' "$RUN_ROOT/inspect_s3.json"
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)

matched = int(payload.get("matched_partition_count", 0))
if matched <= 0:
    raise SystemExit("inspect-s3-compact found zero matched partitions")
if payload.get("readability_failures"):
    raise SystemExit("inspect-s3-compact reported readability failures")
PY

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase BUILD_STARTED \
  --log "$RUN_ROOT/build.log" \
  --exit-file "$RUN_ROOT/build.exit" \
  -- \
  quantlab-ml build-trajectories \
    --source s3-compact \
    --s3-env-file .env \
    --data-config configs/data/controlled-remote-day.yaml \
    --training-config configs/training/production-phase1a-flat-v2-search.yaml \
    --reward-config configs/reward/default.yaml \
    --output "$RUN_ROOT/trajectories"

checkpoint_sync
require_exit_zero "$LOCAL_BUNDLE_ROOT/build.exit" || abort_and_retain "build failed"
require_file "$LOCAL_BUNDLE_ROOT/trajectories/manifest.json" || abort_and_retain "missing trajectories manifest"
require_file "$LOCAL_BUNDLE_ROOT/trajectories/tensor_cache_v1/tensor_cache_manifest.json" || abort_and_retain "missing tensor cache manifest"

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase MATERIALIZE_STARTED \
  --log "$RUN_ROOT/materialize.log" \
  --exit-file "$RUN_ROOT/materialize.exit" \
  -- \
  quantlab-ml materialize-phase1a-supervision \
    --trajectories "$RUN_ROOT/trajectories" \
    --training-config configs/training/production-phase1a-flat-v2-search.yaml \
    --output "$RUN_ROOT/trajectories/phase1a_supervision_v1" \
    --profile-output "$RUN_ROOT/phase1a_profile.json"

checkpoint_sync
require_exit_zero "$LOCAL_BUNDLE_ROOT/materialize.exit" || abort_and_retain "materialization failed"
require_file "$LOCAL_BUNDLE_ROOT/trajectories/phase1a_supervision_v1/manifest.json" || abort_and_retain "missing supervision manifest"
require_file "$LOCAL_BUNDLE_ROOT/phase1a_profile.json" || abort_and_retain "missing phase1a_profile.json after materialize"
require_profile_summary "$LOCAL_BUNDLE_ROOT/phase1a_profile.json" "tensor_cache_used" "true" || abort_and_retain "tensor_cache_used=false after materialize"
require_profile_summary "$LOCAL_BUNDLE_ROOT/phase1a_profile.json" "phase1a_supervision_used" "true" || abort_and_retain "phase1a_supervision_used=false after materialize"
require_profile_number_max "$LOCAL_BUNDLE_ROOT/phase1a_profile.json" "materialization_wall_sec" "1500" || abort_and_retain "materialization exceeded 25-minute threshold"

python3 - <<'PY' > "$RUN_ROOT/profile-1cand-1epoch.yaml"
import yaml
cfg = yaml.safe_load(open('configs/training/production-phase1a-flat-v2-search.yaml'))
cfg['trainer']['epochs'] = 1
cfg['trainer']['candidate_search'] = {'seeds': [7], 'learning_rates': [0.05], 'l2_weights': [0.00005]}
print(yaml.safe_dump(cfg, sort_keys=False))
PY

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase TRAIN_STARTED \
  --log "$RUN_ROOT/train.log" \
  --exit-file "$RUN_ROOT/train.exit" \
  -- \
  quantlab-ml train \
    --trajectories "$RUN_ROOT/trajectories" \
    --training-config "$RUN_ROOT/profile-1cand-1epoch.yaml" \
    --profile-output "$RUN_ROOT/phase1a_profile.json" \
    --registry-root "$RUN_ROOT/registry" \
    --output "$RUN_ROOT/policy.json"

checkpoint_sync
require_exit_zero "$LOCAL_BUNDLE_ROOT/train.exit" || abort_and_retain "train failed"
require_file "$LOCAL_BUNDLE_ROOT/policy.json" || abort_and_retain "missing policy.json after train"
require_profile_summary "$LOCAL_BUNDLE_ROOT/phase1a_profile.json" "tensor_cache_used" "true" || abort_and_retain "tensor_cache_used=false after train"
require_profile_summary "$LOCAL_BUNDLE_ROOT/phase1a_profile.json" "phase1a_supervision_used" "true" || abort_and_retain "phase1a_supervision_used=false after train"
require_profile_summary "$LOCAL_BUNDLE_ROOT/phase1a_profile.json" "jsonl_fallback_used" "false" || abort_and_retain "jsonl_fallback_used=true after train"
require_profile_number_max "$LOCAL_BUNDLE_ROOT/phase1a_profile.json" "candidate_wall_sec" "480" || abort_and_retain "1 candidate x 1 epoch exceeded 8-minute threshold"

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase EVAL_STARTED \
  --log "$RUN_ROOT/evaluate.log" \
  --exit-file "$RUN_ROOT/evaluate.exit" \
  -- \
  quantlab-ml evaluate \
    --trajectories "$RUN_ROOT/trajectories" \
    --policy "$RUN_ROOT/policy.json" \
    --evaluation-config configs/evaluation/default.yaml \
    --split final_untouched_test \
    --profile-output "$RUN_ROOT/phase1a_profile.json" \
    --output "$RUN_ROOT/evaluation.json"

checkpoint_sync
require_exit_zero "$LOCAL_BUNDLE_ROOT/evaluate.exit" || abort_and_retain "evaluate failed"
require_file "$LOCAL_BUNDLE_ROOT/evaluation.json" || abort_and_retain "missing evaluation.json"
require_profile_summary "$LOCAL_BUNDLE_ROOT/phase1a_profile.json" "compiled_v2_eval_used" "true" || abort_and_retain "compiled_v2_eval_used=false"

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase SCORE_STARTED \
  --log "$RUN_ROOT/score.log" \
  --exit-file "$RUN_ROOT/score.exit" \
  -- \
  quantlab-ml score \
    --policy "$RUN_ROOT/policy.json" \
    --evaluation "$RUN_ROOT/evaluation.json" \
    --registry-root "$RUN_ROOT/registry" \
    --output "$RUN_ROOT/score.json"

checkpoint_sync
require_exit_zero "$LOCAL_BUNDLE_ROOT/score.exit" || abort_and_retain "score failed"
require_file "$LOCAL_BUNDLE_ROOT/score.json" || abort_and_retain "missing score.json"
```

Early-abort thresholds for this smallest remote scope:
- `materialization_wall_sec > 1500`
- `candidate_wall_sec > 480`
- any non-zero `*.exit`
- any missing manifest or required artifact
- `tensor_cache_used=false`
- `phase1a_supervision_used=false`
- `compiled_v2_eval_used=false`
- `jsonl_fallback_used=true`

The full `Stage A` same-root chain remains blocked until this smallest remote scope is green and reviewed.

### Stage A

Run only:
- `inspect-s3-compact`
- `build-trajectories`
- `train`
- selected `evaluate`
- selected `score`

```bash
set -euo pipefail

require_file() {
  local path="$1"
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 1; }
}

require_exit_zero() {
  local exit_file="$1"
  require_file "$exit_file"
  [[ "$(tr -d '[:space:]' < "$exit_file")" == "0" ]] || {
    echo "stage failed: $exit_file" >&2
    exit 1
  }
}

export RUN_ID="$(basename "$RUN_ROOT")"
mkdir -p "$RUN_ROOT/registry"

quantlab-ml inspect-s3-compact \
  --env-file .env \
  --data-config configs/data/controlled-remote-day.yaml \
  > "$RUN_ROOT/inspect_s3.json"

python - <<'PY' "$RUN_ROOT/inspect_s3.json"
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)

matched = int(payload.get("matched_partition_count", 0))
if matched <= 0:
    raise SystemExit("inspect-s3-compact found zero matched partitions")
if payload.get("readability_failures"):
    raise SystemExit("inspect-s3-compact reported readability failures")
PY

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase BUILD_STARTED \
  --log "$RUN_ROOT/build.log" \
  --exit-file "$RUN_ROOT/build.exit" \
  -- \
  quantlab-ml build-trajectories \
    --source s3-compact \
    --s3-env-file .env \
    --data-config configs/data/controlled-remote-day.yaml \
    --training-config configs/training/production-phase1a-flat-v2-search.yaml \
    --reward-config configs/reward/default.yaml \
    --output "$RUN_ROOT/trajectories"
require_exit_zero "$RUN_ROOT/build.exit"

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase TRAIN_STARTED \
  --log "$RUN_ROOT/train.log" \
  --exit-file "$RUN_ROOT/train.exit" \
  -- \
  quantlab-ml train \
    --trajectories "$RUN_ROOT/trajectories" \
    --training-config configs/training/production-phase1a-flat-v2-search.yaml \
    --registry-root "$RUN_ROOT/registry" \
    --output "$RUN_ROOT/policy.json"
require_exit_zero "$RUN_ROOT/train.exit"

require_file "$RUN_ROOT/policy.json"
require_file "$RUN_ROOT/policy_search.json"

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase EVAL_STARTED \
  --log "$RUN_ROOT/evaluate.log" \
  --exit-file "$RUN_ROOT/evaluate.exit" \
  -- \
  quantlab-ml evaluate \
    --trajectories "$RUN_ROOT/trajectories" \
    --policy "$RUN_ROOT/policy.json" \
    --evaluation-config configs/evaluation/default.yaml \
    --split final_untouched_test \
    --output "$RUN_ROOT/evaluation.json"
require_exit_zero "$RUN_ROOT/evaluate.exit"

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase SCORE_STARTED \
  --log "$RUN_ROOT/score.log" \
  --exit-file "$RUN_ROOT/score.exit" \
  -- \
  quantlab-ml score \
    --policy "$RUN_ROOT/policy.json" \
    --evaluation "$RUN_ROOT/evaluation.json" \
    --registry-root "$RUN_ROOT/registry" \
    --output "$RUN_ROOT/score.json"
require_exit_zero "$RUN_ROOT/score.exit"

python - <<'PY' \
  "$RUN_ROOT/policy.json" \
  "$RUN_ROOT/policy_search.json" \
  "$RUN_ROOT/evaluation.json" \
  "$RUN_ROOT/stage-a-summary.json" \
  "$RUN_ROOT/stage-b.allowed"
import json
import sys
from pathlib import Path

policy_path, search_path, evaluation_path, summary_path, allow_flag_path = sys.argv[1:6]

with open(policy_path, encoding="utf-8") as handle:
    policy = json.load(handle)
with open(search_path, encoding="utf-8") as handle:
    search = json.load(handle)
with open(evaluation_path, encoding="utf-8") as handle:
    evaluation = json.load(handle)

training_summary = policy.get("training_summary", {})
diagnostics = evaluation.get("diagnostics") or {}

artifact_checks = {
    "selected_policy_matches_manifest": search.get("selected_policy_id") == policy.get("policy_id"),
    "candidate_manifest_has_runner_up": len(search.get("candidates", [])) >= 2,
    "selection_metric_total_net_return": training_summary.get("selection_metric") == "total_net_return",
    "final_untouched_test_unused_for_selection": training_summary.get("final_untouched_test_used") is False,
    "bootstrap_horizon_steps_h4": training_summary.get("bootstrap_horizon_steps") == 4,
    "oracle_masked_rows_positive": training_summary.get("oracle_masked_row_count", 0) > 0,
    "oracle_label_coverage_partial": 0.0 < training_summary.get("oracle_label_coverage_ratio", 0.0) < 1.0,
}

continuation_checks = {
    "total_net_return_gate": evaluation.get("total_net_return", float("-inf")) >= -0.60,
    "gross_directional_pnl_gate": diagnostics.get("gross_directional_pnl", float("-inf")) > 0.15,
    "trade_rate_gate": diagnostics.get("trade_rate", float("inf")) <= 0.65,
    "fee_slippage_burden_gate": diagnostics.get("fee_slippage_burden", float("inf")) <= 0.75,
    "mean_dwell_steps_gate": diagnostics.get("mean_dwell_steps", float("-inf")) >= 2.0,
    "flip_rate_gate": diagnostics.get("flip_rate", float("inf")) <= 0.20,
    "venue_switch_rate_gate": diagnostics.get("venue_switch_rate", float("inf")) <= 0.08,
}

summary = {
    "total_net_return": evaluation.get("total_net_return"),
    "gross_directional_pnl": diagnostics.get("gross_directional_pnl"),
    "trade_rate": diagnostics.get("trade_rate"),
    "fee_slippage_burden": diagnostics.get("fee_slippage_burden"),
    "mean_dwell_steps": diagnostics.get("mean_dwell_steps"),
    "flip_rate": diagnostics.get("flip_rate"),
    "venue_switch_rate": diagnostics.get("venue_switch_rate"),
    "oracle_masked_row_count": training_summary.get("oracle_masked_row_count"),
    "oracle_label_coverage_ratio": training_summary.get("oracle_label_coverage_ratio"),
    "artifact_checks": artifact_checks,
    "continuation_checks": continuation_checks,
}

continue_to_stage_b = all(artifact_checks.values()) and all(continuation_checks.values())
summary["continue_to_stage_b"] = continue_to_stage_b

Path(summary_path).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))

allow_flag = Path(allow_flag_path)
if continue_to_stage_b:
    allow_flag.write_text("stage_b_allowed\n", encoding="utf-8")
else:
    allow_flag.unlink(missing_ok=True)
PY

echo "Stage A complete. Review $RUN_ROOT/stage-a-summary.json"
test -f "$RUN_ROOT/stage-b.allowed" && echo "Stage B is allowed." || echo "Stage B remains blocked."
```

### Stage B

Run only if `Stage A` produced `$RUN_ROOT/stage-b.allowed`.

```bash
set -euo pipefail

require_file() {
  local path="$1"
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 1; }
}

require_exit_zero() {
  local exit_file="$1"
  require_file "$exit_file"
  [[ "$(tr -d '[:space:]' < "$exit_file")" == "0" ]] || {
    echo "stage failed: $exit_file" >&2
    exit 1
  }
}

require_file "$RUN_ROOT/stage-b.allowed"
require_file "$RUN_ROOT/policy_search.json"
require_file "$RUN_ROOT/policy.json"
require_file "$RUN_ROOT/score.json"

eval "$(python - <<'PY' "$RUN_ROOT/policy_search.json"
import json
import shlex
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)

selected_policy_id = payload["selected_policy_id"]
challengers = [candidate for candidate in payload["candidates"] if not candidate["selected_candidate"]]
if not challengers:
    raise SystemExit("policy_search.json does not contain a non-selected challenger")
challenger = min(
    challengers,
    key=lambda candidate: (candidate["candidate_rank"], candidate["candidate_index"]),
)

exports = {
    "SELECTED_POLICY_ID": selected_policy_id,
    "CHALLENGER_POLICY_ID": challenger["policy_id"],
    "CHALLENGER_POLICY_PATH": challenger["artifact_path"],
}
for key, value in exports.items():
    print(f"export {key}={shlex.quote(str(value))}")
PY
)"

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase EXPORT_STARTED \
  --log "$RUN_ROOT/export.log" \
  --exit-file "$RUN_ROOT/export.exit" \
  -- \
  quantlab-ml export-policy \
    --policy "$RUN_ROOT/policy.json" \
    --score "$RUN_ROOT/score.json" \
    --output "$RUN_ROOT/inference_artifact.json"
require_exit_zero "$RUN_ROOT/export.exit"

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase PAPER_SIM_STARTED \
  --log "$RUN_ROOT/champion-paper-sim.log" \
  --exit-file "$RUN_ROOT/champion-paper-sim.exit" \
  -- \
  quantlab-ml record-paper-sim \
    --registry-root "$RUN_ROOT/registry" \
    --policy-id "$SELECTED_POLICY_ID" \
    --report "$RUN_ROOT/champion-paper-sim.md"
require_exit_zero "$RUN_ROOT/champion-paper-sim.exit"

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase PROMOTION_STARTED \
  --log "$RUN_ROOT/champion-promotion.log" \
  --exit-file "$RUN_ROOT/champion-promotion.exit" \
  -- \
  quantlab-ml promote-policy \
    --registry-root "$RUN_ROOT/registry" \
    --policy-id "$SELECTED_POLICY_ID" \
    --evidence "$RUN_ROOT/champion-promotion-evidence.yaml" \
    --output "$RUN_ROOT/champion-promotion-decision.json"
require_exit_zero "$RUN_ROOT/champion-promotion.exit"

python - <<'PY' "$RUN_ROOT/champion-promotion-decision.json"
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)

if payload.get("decision") != "promote":
    raise SystemExit("selected policy was not promoted; challenger compare chain stays blocked")
PY

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase EVAL_STARTED \
  --log "$RUN_ROOT/challenger-evaluate.log" \
  --exit-file "$RUN_ROOT/challenger-evaluate.exit" \
  -- \
  quantlab-ml evaluate \
    --trajectories "$RUN_ROOT/trajectories" \
    --policy "$CHALLENGER_POLICY_PATH" \
    --evaluation-config configs/evaluation/default.yaml \
    --split final_untouched_test \
    --output "$RUN_ROOT/challenger-evaluation.json"
require_exit_zero "$RUN_ROOT/challenger-evaluate.exit"

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase SCORE_STARTED \
  --log "$RUN_ROOT/challenger-score.log" \
  --exit-file "$RUN_ROOT/challenger-score.exit" \
  -- \
  quantlab-ml score \
    --policy "$CHALLENGER_POLICY_PATH" \
    --evaluation "$RUN_ROOT/challenger-evaluation.json" \
    --registry-root "$RUN_ROOT/registry" \
    --output "$RUN_ROOT/challenger-score.json"
require_exit_zero "$RUN_ROOT/challenger-score.exit"

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase COMPARE_STARTED \
  --log "$RUN_ROOT/challenger-compare.log" \
  --exit-file "$RUN_ROOT/challenger-compare.exit" \
  -- \
  quantlab-ml compare-policies \
    --registry-root "$RUN_ROOT/registry" \
    --challenger-policy-id "$CHALLENGER_POLICY_ID" \
    --output "$RUN_ROOT/challenger-comparison-report.json"
require_exit_zero "$RUN_ROOT/challenger-compare.exit"

export COMPARISON_REPORT_ID="$(python - <<'PY' "$RUN_ROOT/challenger-comparison-report.json"
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)

print(payload["comparison_report_id"])
PY
)"

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase PAPER_SIM_STARTED \
  --log "$RUN_ROOT/challenger-paper-sim.log" \
  --exit-file "$RUN_ROOT/challenger-paper-sim.exit" \
  -- \
  quantlab-ml record-paper-sim \
    --registry-root "$RUN_ROOT/registry" \
    --policy-id "$CHALLENGER_POLICY_ID" \
    --report "$RUN_ROOT/challenger-paper-sim.md" \
    --comparison-report-id "$COMPARISON_REPORT_ID"
require_exit_zero "$RUN_ROOT/challenger-paper-sim.exit"
```

Evidence-file rule:
- `champion-promotion-evidence.yaml` must be explicit; do not synthesize defaults at invocation time
- record exact `paper_sim_evidence_id`, `deployment_artifact_path`, reproducibility metadata, and runtime-boundary booleans in that file
- if the selected challenger is later promoted inside the same root, create a second explicit promotion-evidence file that references the exact same-root `comparison_report_id`

## Expected outputs

The first controlled run should leave behind:
- `inspect_s3.json`
- `trajectories/` (canonical JSONL directory + tensor cache sidecar)
- `trajectories/manifest.json`
- `trajectories/train.jsonl`
- `trajectories/validation.jsonl`
- `trajectories/development.jsonl`
- `trajectories/final_untouched_test.jsonl`
- `trajectories/tensor_cache_v1/tensor_cache_manifest.json`
- split-scoped tensor cache shard files and replay sidecars under `trajectories/tensor_cache_v1/`
- `policy.json`
- `policy_search.json` when `production-phase1a-flat-v2-search.yaml` is used
- `policy_candidates/` when `production-phase1a-flat-v2-search.yaml` is used
- `evaluation.json`
- `score.json`
- `stage-a-summary.json`
- optional `stage-b.allowed`
- `inference_artifact.json`
- `build.log`
- `train.log`
- `evaluate.log`
- `score.log`
- `export.log`
- `registry/`
- optional `acceptance_evidence.json` derived from the retained run files above

For `QL-031` same-root proof runs, also retain:
- `champion-paper-sim.md`
- `champion-promotion-evidence.yaml`
- `champion-promotion-decision.json`
- `challenger-evaluation.json`
- `challenger-score.json`
- `challenger-comparison-report.json`
- `challenger-paper-sim.md`
- comparison, paper/sim, and promotion JSONs under `registry/`
- retained manifest metadata that records the exact Vast.ai instance details used for the run

After copying the retained bundle locally, build `bundle_manifest.json` and `SHA256SUMS` directly from the copied bundle:

Historical note:
- the example below is for the retained `2026-04-19` same-root blocker bundle only
- it intentionally copies the exact historical training config used by that source run
- do not reuse that config-copy line for a new `Parallel V2 Phase 1A` launch

```bash
python scripts/retain_remote_run_bundle.py \
  --bundle-root outputs/ql031-same-root-proof-bundle \
  --source-run-root /workspace/runs/ql031-same-root-proof-20260419 \
  --instance-metadata outputs/ql031-same-root-proof-bundle/vast-instance.json \
  --ql031-status-path outputs/ql031-analysis/ql031_status.json \
  --config-copy configs/data/controlled-remote-day.yaml:configs/data/controlled-remote-day.yaml \
  --config-copy configs/training/production-ql031-search.yaml:configs/training/production-ql031-search.yaml \
  --config-copy configs/reward/default.yaml:configs/reward/default.yaml \
  --config-copy configs/evaluation/default.yaml:configs/evaluation/default.yaml
```

The retained manifest must keep the `external_retained_evidence` interpretation explicit, include the same-root comparison and paper/sim linkage summary, and record the exact Vast.ai instance details rather than generic capacity labels.

Inside `training_summary`, confirm:
- `training_backend = pytorch`
- `training_device = cuda`
- `cuda_available = true`
- `selection_fold_count > 0`
- `final_untouched_test_used = false`
- `learned_normalization_fit_split = train`
- `training_data_flow = tensor_shard_batch`
- `validation_data_flow = tensor_shard_evaluation`
- `normalization_strategy = train_only_two_pass_tensor_cache`
- `proxy_validation_used = false`
- `tensor_cache_used = true`
- `jsonl_fallback_used = false`
- `tensor_cache_format = tensor_cache_v1`
- `tensor_cache_shard_count > 0`
- `effective_batch_size > 0`
- `estimated_batch_bytes > 0`
- `batches_per_epoch > 0`
- `batch_target_bytes = 134217728`
- `validation_wall_sec_history` length matches `epochs`

Inside the logs, confirm:
- `tensor_cache_used=true`
- `jsonl_fallback_used=false`
- `compiled_policy_mode=tensor_cache_linear_policy_batch`
- `train_rows_per_sec`, `validation_rows_per_sec`, and `evaluation_rows_per_sec` are present
- if `acceptance_evidence.json` is present, it must only index the retained files and may not replace them as the source of truth

## Shutdown retention for authoritative reruns

Retained bundles are now explicitly classified:
- `full`: replayable retained bundle with split JSONL and tensor-cache payloads intact
- `slim`: audit-only retained bundle that is non-replayable and does not support Phase 0 empirical closure

Integrity rule:
- a `slim` bundle must not keep `trajectories/tensor_cache_v1/tensor_cache_manifest.json` if the referenced shard payloads are absent
- if tensor-cache summary metadata is needed in a `slim` bundle, keep a non-dangling summary artifact such as `tensor_cache_manifest.summary.json` instead
- retained manifests must declare `bundle_payload_class`, `replayable`, and `supports_phase0_empirical_closure`
- legacy dangling bundles must be normalized into a sibling copy or an explicitly receipted in-place form; the original defect trail must remain auditable via `normalization_receipt.json`

Before instance termination, retain the minimum evidence bundle needed to support future truth, audit, closeout, and docs verification.

Retain for `slim` bundles:
- `continuity_audit_authoritative.json`
- `continuity_authority_discovery.json`
- `inspect_s3.json`
- `policy.json`
- `evaluation.json`
- `score.json`
- `inference_artifact.json`
- `trajectories/manifest.json`
- `trajectories/tensor_cache_v1/tensor_cache_manifest.summary.json` when tensor-cache summary metadata is needed
- `registry/index.json`
- active `registry/records/*`
- active `registry/evaluations/*`
- active `registry/scores/*`
- active `registry/artifacts/*` with duplicate bytes avoided when a hardlink to `policy.json` is sufficient
- active `registry/comparisons/*` for same-root comparison runs
- active `registry/paper_sim/*` for same-root proof runs
- active `registry/promotions/*` for same-root proof runs
- `build.log`, `train.log`, `evaluate.log`, `score.log`, `export.log`
- `build.exit`, `train.exit`, `evaluate.exit`, `score.exit`, `export.exit`
- exact copies of the data, training, reward, and evaluation config files used for the run
- retained manifest metadata with source commit SHA, run root, timestamps, training summary, authority summary, and instance metadata
- retained checksums such as `SHA256SUMS`

Required retained manifest instance metadata:
- exact booked Vast.ai GPU model
- vCPU count
- RAM size
- disk size
- advertised storage or host-throughput note when the listing exposes it
- exact host label / offer identifier used for booking

Do not copy:
- raw market data
- full split JSONL payloads when intentionally producing a `slim` bundle
- full tensor-cache shard payloads when intentionally producing a `slim` bundle
- temporary transfer files
- duplicate large artifacts that carry no additional decision evidence

Retain for `full` bundles:
- everything required by the `slim` bundle
- full split JSONL payloads
- full tensor-cache shard payloads and replay JSONL sidecars
- `trajectories/tensor_cache_v1/tensor_cache_manifest.json`

Use a `full` bundle when replay, JSONL fallback, empirical diagnostics re-materialization, or Phase 0 closure evidence may be needed later.

Retention honesty:
- the retained-local bundle is a preserved copy derived from an authoritative rerun
- the retained-local bundle is not itself re-labeled as authoritative evidence

## Acceptance criteria

The first controlled run is successful when:
- every command exits `0`
- S3 preflight shows matched partitions `> 0`
- the full artifact chain is written successfully
- training records `cuda` as the selected device
- walk-forward and train-only normalization evidence remain intact
- training/evaluate logs prove the tensor-cache hot path is active (`tensor_cache_used=true`, `jsonl_fallback_used=false`)
- training logs expose `effective_batch_size`, `estimated_batch_bytes`, `batches_per_epoch`, `batch_target_bytes`, `train_rows_per_sec`, and `validation_rows_per_sec`
- evaluate logs expose `evaluation_rows_per_sec` and `compiled_policy_mode=tensor_cache_linear_policy_batch`
- train `epoch_wall_sec < 300`
- per-epoch `validation_wall_sec < 60`
- final `evaluate_wall_sec < 180`
- no phase exits with `137` or other OOM-kill evidence

For QL-021-style controlled proof runs, average GPU utilization is diagnostic telemetry only.
Low average utilization does not invalidate a successful controlled proof run when direct hot-path evidence already confirms:
- `training_device=cuda`
- `tensor_cache_used=true`
- `jsonl_fallback_used=false`
- required chain exit codes are all `0`
- explicit `train` / `evaluate` execution evidence is present
- timing gates above are satisfied

This run does not need:
- positive economic score
- promotion readiness
- larger search budget

## Failure triage

Check first:

### `torch.cuda.is_available() = false`
- wrong PyTorch wheel
- driver/runtime mismatch on the host
- CUDA not exposed into the container

### `training_device=cpu`
- CUDA selection fallback triggered
- remote environment is not exposing GPU to PyTorch
- the run is not valid as the first controlled GPU evidence run

### `tensor_cache_used=false` or `jsonl_fallback_used=true`
- prod directory fast path is not active
- tensor cache sidecar is missing or unreadable
- the run is not valid as QL-021 acceptance evidence unless compat fallback was explicitly being debugged

### low average GPU utilization
- treat it as advisory telemetry, not as the primary acceptance truth
- if `training_device=cuda`, the tensor-cache hot path is active, required chain exits are `0`, and timing gates pass, the controlled proof run remains valid
- investigate low utilization only as a bottleneck/throughput diagnostic or future optimization signal

### `build-trajectories` fails or becomes too slow
- controlled snapshot is too large for the host RAM/disk
- compact object readability is degraded
- disk allocation is too small

### registry or artifact chain missing
- wrong `RUN_ROOT`
- wrong `--registry-root`
- build/train/evaluate/score/export chain was interrupted

### evaluation looks nonsensical
- wrong data config
- smoke profile used accidentally
- production profile not actually used

### training or evaluation is still slow despite CUDA
- tensor cache fast path may not be active
- the run may be spending time in compat fallback or another hidden JSONL path
- inspect `train.log` / `evaluate.log` for `tensor_cache_used`, `jsonl_fallback_used`, and throughput fields before changing hardware size

## What comes next

If this run is clean:
- keep the same workflow
- run a slightly wider second controlled run
- optionally widen date span or add a small candidate search budget

Keep separate:
- external `audit-continuity --registry-root <active-runtime-registry-root>` remains a parallel operational follow-up
- it is not a blocker for the first controlled remote GPU run
