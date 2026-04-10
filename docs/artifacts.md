# Artifact Contract

## Policy Artifact

Each policy artifact is split into two parts:

- `policy_payload`: opaque serialized runtime payload. The executor does not inspect
  internals; the runtime bridge is responsible for interpreting it.
- `executor_metadata`: applicability and ranking metadata needed by the thin executor.

Required executor metadata fields:

- `asset_universe`
- `venue_compatibility`
- `instrument_compatibility`
- `min_capital_requirement`
- `size_bounds`
- `leverage_bounds`
- `liquidity_flags`
- `applicability_flags`
- `expected_return`
- `risk_score`
- `turnover_score`
- `confidence_score`
- `artifact_version`
- `lineage_pointer`

## Registry Record

The registry stores more than blobs:

- Dataset hash and slice id
- Reward config hash
- Training config hash
- Parent policy id and lineage chain
- Champion/challenger status
- Train and eval windows
- Score history
- Coverage and experience statistics
