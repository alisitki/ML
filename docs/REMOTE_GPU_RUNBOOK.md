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

# NOTE: --output is a directory (streaming JSONL format).
# The directory will contain manifest.json + per-split JSONL files.
# The train and evaluate commands auto-detect the directory format.

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
- `trajectories/` (JSONL streaming directory)
  - `manifest.json`
  - `train.jsonl`
  - `validation.jsonl`
  - `development.jsonl`
  - `final_untouched_test.jsonl`
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
- `training_data_flow = streaming_batch`
- `validation_data_flow = streaming_evaluation`
- `normalization_strategy = train_only_two_pass_streaming`
- `proxy_validation_used = false`
- `effective_batch_size > 0`
- `estimated_batch_bytes > 0`
- `batches_per_epoch > 0`
- `batch_target_bytes = 134217728`

## Acceptance criteria

The first controlled run is successful when:
- every command exits `0`
- S3 preflight shows matched partitions `> 0`
- the full artifact chain is written successfully
- training records `cuda` as the selected device
- walk-forward and train-only normalization evidence remain intact
- training logs expose `effective_batch_size`, `estimated_batch_bytes`, `batches_per_epoch`, and `batch_target_bytes`
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

## What comes next

If this run is clean:
- keep the same workflow
- run a slightly wider second controlled run
- optionally widen date span or add a small candidate search budget

Keep separate:
- external `audit-continuity --registry-root <active-runtime-registry-root>` remains a parallel operational follow-up
- it is not a blocker for the first controlled remote GPU run
