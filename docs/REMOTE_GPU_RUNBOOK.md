# Remote GPU Runbook

## Purpose

This runbook defines the official first controlled remote-GPU training workflow for
QuantLab real training.

It exists to:
- keep the first Vast run controlled and reproducible
- keep leakage-sensitive split discipline intact
- separate repo implementation work from external continuity follow-up
- avoid treating local smoke workflows as the real-training default

This runbook is provider-agnostic in principle. Vast.ai is used here as the concrete
example workflow because the current operating target is a rented single-GPU instance.

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

## Vast instance guidance

Start with:
- single GPU
- verified or secure-cloud offer
- direct SSH
- on-demand instance
- 80-100 GB disk
- high reliability score

Preferred GPU class:
- `24 GB`-class single card such as `L4`, `A10`, `A5000`, or `RTX 4090`

Avoid for the first run:
- multi-GPU
- interruptible instances
- A100/H100-class cost
- candidate search

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

## Official command flow

```bash
quantlab-ml build-trajectories \
  --source s3-compact \
  --s3-env-file .env \
  --data-config configs/data/controlled-remote-day.yaml \
  --training-config configs/training/production.yaml \
  --reward-config configs/reward/default.yaml \
  --output "$RUN_ROOT/trajectories" \
  2>&1 | tee "$RUN_ROOT/build.log"

# NOTE: --output is a directory.
# The directory will contain canonical JSONL plus a tensor_cache_v1 sidecar.
# The prod train/evaluate commands auto-detect the directory format and
# must use tensor-cache fast paths unless explicit compat fallback is requested.

quantlab-ml train \
  --trajectories "$RUN_ROOT/trajectories" \
  --training-config configs/training/production.yaml \
  --registry-root "$RUN_ROOT/registry" \
  --output "$RUN_ROOT/policy.json" \
  2>&1 | tee "$RUN_ROOT/train.log"

quantlab-ml evaluate \
  --trajectories "$RUN_ROOT/trajectories" \
  --policy "$RUN_ROOT/policy.json" \
  --evaluation-config configs/evaluation/default.yaml \
  --output "$RUN_ROOT/evaluation.json" \
  2>&1 | tee "$RUN_ROOT/evaluate.log"

quantlab-ml score \
  --policy "$RUN_ROOT/policy.json" \
  --evaluation "$RUN_ROOT/evaluation.json" \
  --registry-root "$RUN_ROOT/registry" \
  --output "$RUN_ROOT/score.json" \
  2>&1 | tee "$RUN_ROOT/score.log"

quantlab-ml export-policy \
  --policy "$RUN_ROOT/policy.json" \
  --score "$RUN_ROOT/score.json" \
  --output "$RUN_ROOT/inference_artifact.json" \
  2>&1 | tee "$RUN_ROOT/export.log"
```

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
- `evaluation.json`
- `score.json`
- `inference_artifact.json`
- `build.log`
- `train.log`
- `evaluate.log`
- `score.log`
- `export.log`
- `registry/`

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
- average GPU utilization reaches at least `20%`
- no phase exits with `137` or other OOM-kill evidence

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
