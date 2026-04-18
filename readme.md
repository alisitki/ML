# QuantLab ML

QuantLab ML targets an end-to-end multi-exchange futures ML trading system for futures markets. The destination is a system that ingests websocket market data, builds exchange-aware canonical state, trains and evaluates policies offline, runs runtime inference, and hands controlled execution intent to a thin executor on the path toward live capital deployment.

## Current implemented scope

The repository materially implements the foundation-heavy half of that system today:

- canonical market-data, observation, reward, split, artifact, and registry contracts for the declared market scope
- trajectory building and offline training/evaluation flows
- runtime-facing inference-artifact export and execution-intent contracts
- governance, runbooks, and evidence discipline around reproducible offline work

This is not just a notebook sandbox, but it is also not yet a full live-operating trading system.

## Not yet implemented as current repo reality

The repository does not yet materially implement all of:

- production websocket ingestion across the active venue scope
- a long-running online state / feature service
- replay-vs-live parity proof on live feeds
- a selector runtime daemon consuming live state
- thin executor integration and live control loops
- a shadow/paper operating loop
- system-generated commercialization evidence across live-facing gates

Those are valid next-phase targets, not current capabilities.

## Why this phase matters

A live ML trading system is only commercially credible if canonical semantics, offline evaluation discipline, artifact lineage, and runtime boundaries are correct before live plumbing is added. QuantLab's current phase matters because it hardens the part of the stack that makes later live behavior interpretable, comparable, and governable.

## Next build phase

The next major build phase is the live-operating half:

- websocket ingestion
- online state / feature service
- replay-vs-live parity tooling
- degraded-input, stale-state, and recovery behavior
- selector runtime
- shadow/paper loop
- thin executor integration and live controls

## Current repository boundary

```text
implemented today:
  canonical contracts -> trajectories -> offline training/evaluation -> artifacts/registry -> exported inference artifacts and execution-intent contracts

planned next:
  websocket ingestion -> online state -> runtime selector daemon -> thin executor -> shadow/paper evidence -> commercialization gates
```

## Operational entry points

Data and training configs:

- `configs/data/default.yaml`
- `configs/data/fixture.yaml`
- `configs/data/s3-current.yaml`
- `configs/data/controlled-remote-day.yaml`
- `configs/training/production.yaml`

CLI surfaces:

- `quantlab-ml build-trajectories`
- `quantlab-ml train`
- `quantlab-ml evaluate`
- `quantlab-ml score`
- `quantlab-ml export-policy`
- `quantlab-ml inspect-s3-compact --env-file .env`
- `quantlab-ml audit-continuity --registry-root outputs/registry`

## Read first

- `AGENTS.md`
- `docs/DOCS_INDEX.md`
- `docs/PRODUCT_THESIS.md`
- `docs/MARKET_SCOPE.md`
- `docs/ONLINE_RUNTIME_MODEL.md`
- `docs/COMMERCIALIZATION_GATES.md`
- `docs/PROJECT_STATE.md`

## Canonical technical docs

- `docs/CANONICAL_MARKET_DATA_CONTRACT.md`
- `docs/OBSERVATION_SCHEMA.md`
- `docs/POLICY_ARTIFACT_SCHEMA.md`
- `docs/EXECUTION_INTENT_SCHEMA.md`
- `docs/QUANTLAB_CONSTITUTION.md`
- `docs/RUNTIME_BOUNDARY.md`
- `docs/REMOTE_GPU_RUNBOOK.md`
