# QuantLab ML

QuantLab ML is an offline policy-discovery scaffold for raw multi-exchange market
streams. The repository is not a live execution engine, a generic backtester, or a
rule-strategy host. Its intended system boundary is:

`raw multi-exchange streams -> trajectories / learning surface -> offline policy discovery -> policy registry -> runtime selector / inference -> execution intent -> thin executor`

## Core Guarantees

- Exchange-aware dataset contract with explicit stream availability per venue.
- Single-asset policy samples built from full cross-symbol and cross-exchange context.
- Explicit `abstain` / `no-trade` action in the action contract.
- Versioned policy artifacts for runtime selector / inference, with explicit execution-intent handoff to the thin executor.
- Shared economic reward semantics between training and evaluation replay.
- Registry lineage, score history, and champion-vs-challenger state.

## Config Profiles

- `configs/data/default.yaml` is the target-universe contract.
- `configs/data/fixture.yaml` is smoke-only and is the profile tests and CLI smoke runs
  should use explicitly.
- `configs/data/controlled-remote-day.yaml` is the official first controlled remote-training snapshot example for a single full successful day. It is intentionally larger than smoke, but still bounded for a first readiness run.
- `configs/training/default.yaml` is a smoke/fixture-oriented PyTorch training profile kept as the current continuity default for CLI/tests. It is not the production preset and it does not define long-term strategic direction.
- `configs/training/production.yaml` is the explicit canonical production observation profile: `1m×8`, `5m×8`, `15m×8`, `60m×12`.
- `configs/training/search-small.yaml` is an optional small candidate-search profile for smoke-scale verification. It is not the default core architecture.
- `stream_universe` is the union of stream families; `available_streams_by_exchange`
  defines which exchange-stream coordinates are structurally unavailable.

## Execution Modes

- `local smoke/debug mode`: local CPU/laptop execution is acceptable for smoke runs, debugging, config sanity-checks, and short verification loops.
- `continuity baseline mode`: small local runs are acceptable for temporary continuity baselines and workflow validation, but this mode is not the long-term optimization target for core training design.
- `real training mode`: core model training is PyTorch-first and the expected compute target is remote GPU when available. Provider examples include Vast.ai or equivalent rented GPU providers, but the repo is not locked to any one provider. Secrets, provisioning, and orchestration are outside the scope of this document.

Runtime inference acceleration choices such as ONNX/TensorRT remain separate from training compute decisions.

## Repository Layout

```text
configs/               Dataset, reward, training, evaluation, registry defaults
docs/                  Canonical governance, contracts, runbooks, and state
src/quantlab_ml/
  contracts/           Stable IO models
  data/                Raw source adapters and event normalization
  trajectories/        Time alignment, context assembly, trajectory persistence
  rewards/             Reward snapshots and replay application semantics
  models/              Baseline model and runtime decision types
  training/            Baseline training orchestration
  scoring/             Policy scoring surface
  selection/           Champion/challenger ranking
  policies/            Runtime bridge and executor export contract
  evaluation/          Offline evaluation replay
  registry/            Artifact, lineage, and coverage store
  cli/                 Typer CLI entrypoint
tests/                 Fixture-driven contract and smoke coverage
```

## Canonical Documentation

Use these as source of truth:

- `docs/QUANTLAB_CONSTITUTION.md` and `docs/RUNTIME_BOUNDARY.md` for system identity and boundary.
- `docs/CANONICAL_MARKET_DATA_CONTRACT.md` and `docs/OBSERVATION_SCHEMA.md` for market-data and observation contracts.
- `docs/ACTION_SPACE.md`, `docs/REWARD_SPEC_V1.md`, and `docs/SPLIT_POLICY_V1.md` for action, reward, and split semantics.
- `docs/POLICY_ARTIFACT_SCHEMA.md`, `docs/REGISTRY_SCHEMA.md`, and `docs/EXECUTION_INTENT_SCHEMA.md` for artifact, registry, and executor handoff.
- `docs/REMOTE_GPU_RUNBOOK.md` for the official first controlled remote-GPU training workflow.
- `AGENTS.md`, `docs/PROJECT_STATE.md`, `docs/ROADMAP.md`, and `docs/BACKLOG.md` for repo operating flow.

## Quickstart

This scaffold targets Python 3.12.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ml]"
pytest
```

Core training and the current verification suite both require the ML extra. A lighter install is only useful for docs or non-training inspection work:

```bash
python -m pip install -e ".[dev]"
```

## CLI Flow

Smoke and local verification should use the fixture profile explicitly:

The commands below are local smoke/continuity examples, not the default execution target for meaningful real training. `train` now uses the PyTorch core backend and requires the `.[dev,ml]` environment.

```bash
quantlab-ml build-trajectories \
  --input tests/fixtures/market_events.ndjson \
  --data-config configs/data/fixture.yaml \
  --training-config configs/training/default.yaml \
  --output outputs/trajectories.json

quantlab-ml train \
  --trajectories outputs/trajectories.json \
  --training-config configs/training/default.yaml \
  --output outputs/policy.json

quantlab-ml evaluate \
  --trajectories outputs/trajectories.json \
  --policy outputs/policy.json \
  --output outputs/evaluation.json

quantlab-ml score \
  --policy outputs/policy.json \
  --evaluation outputs/evaluation.json \
  --output outputs/score.json

quantlab-ml export-policy \
  --policy outputs/policy.json \
  --score outputs/score.json \
  --output outputs/executor-policy.json
```

For opt-in candidate search, use `configs/training/search-small.yaml` or another
training config that sets `trainer.candidate_search`:

```bash
quantlab-ml train \
  --trajectories outputs/trajectories.json \
  --training-config configs/training/search-small.yaml \
  --output outputs/search-policy.json
```

The selected artifact still lands at `--output`. When the search produces more
than one candidate, `train` also writes:

- `<output stem>_search.json` with run-level search metadata and ranked candidates
- `<output stem>_candidates/` with the non-selected candidate artifacts

These default/search configs are continuity-oriented local profiles, not proof of production-surface readiness.
`configs/training/default.yaml` remains the smoke/continuity default and is intentionally not the production preset.
Use `configs/training/production.yaml` when you want the canonical production observation surface explicitly.

For the explicit production observation surface:

```bash
quantlab-ml build-trajectories \
  --input tests/fixtures/market_events.ndjson \
  --data-config configs/data/fixture.yaml \
  --training-config configs/training/production.yaml \
  --output outputs/production-trajectories.json
```

`evaluate`, `score`, and `export-policy` remain backward-compatible and continue
to consume the selected artifact path.

To inspect temporary continuity windows in a registry:

```bash
quantlab-ml audit-continuity --registry-root outputs/registry
```

This command reports active training backend counts plus active legacy-compat dependency counts. New core-path artifacts should report `training_backend=pytorch`; any remaining `numpy` count is continuity debt.

## S3 Compact Usage

The compact bucket is state-driven. `compacted/_state.json` is read first to determine
which logical partitions and days exist, then data objects are discovered from the
storage layout under the bucket root.

Required `.env` entries:

```bash
S3_COMPACT_ENDPOINT=...
S3_COMPACT_BUCKET=quantlab-compact
S3_COMPACT_ACCESS_KEY=...
S3_COMPACT_SECRET_KEY=...
S3_COMPACT_REGION=us-east-1
S3_COMPACT_STATE_KEY=compacted/_state.json
```

Inspect the current state and day coverage:

```bash
quantlab-ml inspect-s3-compact --env-file .env
```

For the currently observed bucket layout, a ready-to-run state-backed example profile
is provided at `configs/data/s3-current.yaml`:

```bash
python -m pip install -e ".[dev,ml]"

quantlab-ml build-trajectories \
  --source s3-compact \
  --s3-env-file .env \
  --data-config configs/data/s3-current.yaml \
  --training-config configs/training/default.yaml \
  --reward-config configs/reward/default.yaml \
  --output outputs/s3-trajectories.json
```

This example keeps the continuity/smoke baseline training profile for convenience.
It is not evidence that `configs/training/default.yaml` is the production training default.

For the first controlled remote-GPU readiness run, use the bounded full-day example
snapshot at `configs/data/controlled-remote-day.yaml` together with
`configs/training/production.yaml`. The official command flow, acceptance criteria,
artifact layout, and failure triage are documented in `docs/REMOTE_GPU_RUNBOOK.md`.

The current compact storage layout discovered from the bucket is:

```text
exchange=<exchange>/stream=<stream>/symbol=<symbol>/date=<YYYYMMDD>/data.parquet
exchange=<exchange>/stream=<stream>/symbol=<symbol>/date=<YYYYMMDD>/meta.json
exchange=<exchange>/stream=<stream>/symbol=<symbol>/date=<YYYYMMDD>/quality_day.json
```

## Canonical Learning Surface

See `docs/CANONICAL_MARKET_DATA_CONTRACT.md` and `docs/OBSERVATION_SCHEMA.md` for observation axes, contract availability semantics, mask behavior, and compatibility boundaries.

## Canonical Evaluation

See `docs/REWARD_SPEC.md`, `docs/REWARD_SPEC_V1.md`, `docs/SPLIT_POLICY.md`, `docs/SPLIT_POLICY_V1.md`, and `docs/EVALUATION_RUNBOOK.md` for reward math, split discipline, and official evaluation flow.

## Non-Goals

- Live order execution or routing.
- Explainable rule export.
- Production S3 credentials or infrastructure.
- RL-family-specific abstractions before the learning surface contract is stable.
