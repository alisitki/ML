# QuantLab ML

QuantLab ML is an offline policy-discovery scaffold for raw multi-exchange market
streams. The repository is not a live execution engine, a generic backtester, or a
rule-strategy host. Its intended system boundary is:

`raw multi-exchange streams -> trajectories / learning surface -> offline policy discovery -> policy registry -> thin executor contract`

## Core Guarantees

- Exchange-aware dataset contract with explicit stream availability per venue.
- Single-asset policy samples built from full cross-symbol and cross-exchange context.
- Explicit `abstain` / `no-trade` action in the action contract.
- Opaque policy payloads plus thin executor-facing applicability metadata.
- Shared reward semantics between training snapshots and evaluation replay.
- Registry lineage, coverage metadata, and champion-vs-challenger state.

## Config Profiles

- `configs/data/default.yaml` is the target-universe contract.
- `configs/data/fixture.yaml` is smoke-only and is the profile tests and CLI smoke runs
  should use explicitly.
- `stream_universe` is the union of stream families; `available_streams_by_exchange`
  defines which exchange-stream coordinates are structurally unavailable.

## Repository Layout

```text
configs/               Dataset, reward, training, evaluation, registry defaults
docs/                  Architecture, artifact, data-contract, learning-surface notes
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

## Quickstart

This scaffold targets Python 3.12.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

To install the heavier ML stack later:

```bash
python -m pip install -e ".[dev,ml]"
```

## CLI Flow

Smoke and local verification should use the fixture profile explicitly:

```bash
quantlab-ml build-trajectories \
  --input tests/fixtures/market_events.ndjson \
  --data-config configs/data/fixture.yaml \
  --output outputs/trajectories.json

quantlab-ml train \
  --trajectories outputs/trajectories.json \
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

The current compact storage layout discovered from the bucket is:

```text
exchange=<exchange>/stream=<stream>/symbol=<symbol>/date=<YYYYMMDD>/data.parquet
exchange=<exchange>/stream=<stream>/symbol=<symbol>/date=<YYYYMMDD>/meta.json
exchange=<exchange>/stream=<stream>/symbol=<symbol>/date=<YYYYMMDD>/quality_day.json
```

## Learning Surface Notes

- Observation yüzeyi `[scale, time, symbol, exchange, stream, field]` boyutlu çok ölçekli
  ham tensor bloklarından oluşur; exchange ortalaması veya stream'den tek scalar collapse yoktur.
- Emitted train/eval steps kendi split pencereleri içinde kalır.
- Observation lookback causal pre-split geçmişten yararlanır; eval sabit-sıfır reset yapmaz.
- **`padding`**: yalnızca yetersiz tarihçe — `history_start`'tan önceki bucket'lar.
- **`unavailable_by_contract`**: yapısal olarak erişilemez `(exchange, stream)` koordinatı.
- **`missing`**: erişilebilir ama bu adımda hiç event gelmedi.
- **`stale`**: erişilebilir, event var, freshness bound'u geçmiş.
- Bu dört mask birbirini dışlar ve karıştırılmaz.
- Action feasibility `action × venue × size_band × leverage_band` matrisinden türer;
  yalnızca decision-time bilgi kullanılır.
- Reward context ve timeline venue-specific'tir; exchange-average kullanılmaz.
- V1 reduction mantığı (`target_stream_series`, scalar collapse) yalnızca
  `contracts/compat.py` ve `training/compat_adapter.py`'de yaşar.

## Evaluation Boundary

The v1 replay engine currently supports a narrow boundary:

- fill assumption: `next_mark_price`
- fee handling: `shared_reward_contract`
- funding handling: `carry_from_funding_stream`
- slippage handling: `fixed_bps`
- terminal semantics: `trajectory_boundary_is_terminal`
- timeout semantics: `force_terminal_at_data_end`
- infeasible action treatment: `force_abstain`, with the abstain path applied and the
  configured infeasible penalty still counted

## Non-Goals

- Live order execution or routing.
- Explainable rule export.
- Production S3 credentials or infrastructure.
- RL-family-specific abstractions before the learning surface contract is stable.
