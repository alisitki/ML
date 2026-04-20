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
- switch the training config to `configs/training/production-ql031-search.yaml`
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

Use this overlay only for the `QL-031` closure batch.

Command deltas:
- replace `configs/training/production.yaml` with `configs/training/production-ql031-search.yaml`
- keep `RUN_ROOT` external, for example `/workspace/runs/ql031-same-root-proof-<date>`
- keep one registry root for the selected champion and the compared challenger
- evaluate and score the selected artifact plus at least one non-selected candidate from the same `policy_search.json`

Required same-root chain after the initial `train`:

```bash
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

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase PAPER_SIM_STARTED \
  --log "$RUN_ROOT/champion-paper-sim.log" \
  --exit-file "$RUN_ROOT/champion-paper-sim.exit" \
  -- \
  quantlab-ml record-paper-sim \
    --registry-root "$RUN_ROOT/registry" \
    --policy-id <selected-policy-id> \
    --report "$RUN_ROOT/champion-paper-sim.md"

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase PROMOTION_STARTED \
  --log "$RUN_ROOT/champion-promotion.log" \
  --exit-file "$RUN_ROOT/champion-promotion.exit" \
  -- \
  quantlab-ml promote-policy \
    --registry-root "$RUN_ROOT/registry" \
    --policy-id <selected-policy-id> \
    --evidence "$RUN_ROOT/champion-promotion-evidence.yaml" \
    --output "$RUN_ROOT/champion-promotion-decision.json"

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase EVAL_STARTED \
  --log "$RUN_ROOT/challenger-evaluate.log" \
  --exit-file "$RUN_ROOT/challenger-evaluate.exit" \
  -- \
  quantlab-ml evaluate \
    --trajectories "$RUN_ROOT/trajectories" \
    --policy "$RUN_ROOT/policy_candidates/<challenger-policy-id>.json" \
    --evaluation-config configs/evaluation/default.yaml \
    --output "$RUN_ROOT/challenger-evaluation.json"

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase SCORE_STARTED \
  --log "$RUN_ROOT/challenger-score.log" \
  --exit-file "$RUN_ROOT/challenger-score.exit" \
  -- \
  quantlab-ml score \
    --policy "$RUN_ROOT/policy_candidates/<challenger-policy-id>.json" \
    --evaluation "$RUN_ROOT/challenger-evaluation.json" \
    --registry-root "$RUN_ROOT/registry" \
    --output "$RUN_ROOT/challenger-score.json"

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase COMPARE_STARTED \
  --log "$RUN_ROOT/challenger-compare.log" \
  --exit-file "$RUN_ROOT/challenger-compare.exit" \
  -- \
  quantlab-ml compare-policies \
    --registry-root "$RUN_ROOT/registry" \
    --challenger-policy-id <challenger-policy-id> \
    --output "$RUN_ROOT/challenger-comparison-report.json"

python scripts/remote_run_stage.py \
  --run-id "$RUN_ID" \
  --phase PAPER_SIM_STARTED \
  --log "$RUN_ROOT/challenger-paper-sim.log" \
  --exit-file "$RUN_ROOT/challenger-paper-sim.exit" \
  -- \
  quantlab-ml record-paper-sim \
    --registry-root "$RUN_ROOT/registry" \
    --policy-id <challenger-policy-id> \
    --report "$RUN_ROOT/challenger-paper-sim.md" \
    --comparison-report-id <comparison-report-id>
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
- `policy_search.json` when `production-ql031-search.yaml` is used
- `policy_candidates/` when `production-ql031-search.yaml` is used
- `evaluation.json`
- `score.json`
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
