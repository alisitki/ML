# QuantLab ML

QuantLab ML targets an end-to-end multi-exchange futures ML trading system for futures markets. The target system ingests websocket market data, builds exchange-aware canonical state, trains and evaluates policies offline, runs runtime inference, and hands controlled execution intent to a thin executor on the path toward live capital deployment.

## Current implemented scope

The repository materially implements the offline-first foundation-heavy half of that target today:

- canonical market-data, observation, reward, split, artifact, registry, and execution-intent contracts for the declared market scope
- trajectory building plus offline training, evaluation, scoring, and policy export flows
- registry lineage, promotion discipline, and continuity-audit tooling for temporary compatibility windows
- governance docs, runbooks, and repo-tracked truth surfaces for controlled offline work

This is not a notebook sandbox, but it is also not yet a live-operating trading system.

## Current closure verdict

QuantLab is currently `offline operational but not professionally closed`.

- The offline engine is real on current HEAD: `build-trajectories`, `train`, `evaluate`, `score`, and `export-policy` exist and are test-covered.
- Current HEAD also carries repo-tracked continuity closeout records and an authority-aware continuity audit surface, but that is not the same thing as authoritative closeout evidence.
- Current HEAD does not include repo-tracked QL-021 retained bundles under `outputs/`; any such bundle must be treated as external retained evidence until its provenance and authority are confirmed.
- Offline closure is still incomplete because continuity retirement depends on authoritative evidence, relocation-safe retained artifacts, and explicit evidence-based closure criteria.
- Until those blockers are cleared or explicitly frozen, live/runtime buildout is a planned next phase, not the current main focus.

## Not yet implemented as current repo reality

The repository does not yet materially implement all of:

- production websocket ingestion across the active venue scope
- a long-running online state / feature service
- replay-vs-live parity proof on live feeds
- a selector runtime daemon consuming live state
- thin executor integration and live control loops
- a shadow/paper operating loop
- system-generated commercialization evidence across live-facing gates

Those are target-state layers, not current capabilities.

## Current focus before live/runtime work

- keep repo-truth docs aligned with current HEAD rather than planned architecture
- keep `audit-continuity` authority-aware so inspected-scope truth is not mistaken for authoritative closeout truth
- keep repo-tracked closeout records explicit and pending until authoritative evidence is attached
- define explicit offline-closure evidence gaps before Phase 2 becomes the main execution track

## Next build phase

The planned next major build phase remains the live-operating half, but it should not become the main focus until the current offline-closure blockers are explicit:

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

current main focus:
  truthful offline-closure hardening -> continuity audit semantics -> evidence / state honesty

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

`outputs/` is a local ignored workspace path in this repo. An audit run under `outputs/` is neither a repo-tracked artifact nor authoritative evidence by itself.

`audit-continuity` is scope-limited: zero active records, unreadable retained artifact paths, or unknown authoritative registry scope are not retirement proof.

## Read first

- `AGENTS.md`
- `docs/DOCS_INDEX.md`
- `docs/PRODUCT_THESIS.md`
- `docs/MARKET_SCOPE.md`
- `docs/ONLINE_RUNTIME_MODEL.md`
- `docs/COMMERCIALIZATION_GATES.md`
- `docs/PROJECT_STATE.md`
- `docs/OFFLINE_CLOSURE_CRITERIA.md`
- `docs/CONTINUITY_AUDIT_RUNBOOK.md`
- `docs/CONTINUITY_CLOSEOUT_RECORDS.md`

## Canonical technical docs

- `docs/CANONICAL_MARKET_DATA_CONTRACT.md`
- `docs/OBSERVATION_SCHEMA.md`
- `docs/POLICY_ARTIFACT_SCHEMA.md`
- `docs/EXECUTION_INTENT_SCHEMA.md`
- `docs/QUANTLAB_CONSTITUTION.md`
- `docs/RUNTIME_BOUNDARY.md`
- `docs/REMOTE_GPU_RUNBOOK.md`
