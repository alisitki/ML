# Policy Artifact Schema

## Purpose

This document defines the canonical policy artifact contract for QuantLab.

A policy artifact is what survives offline training and becomes eligible for:
- registry entry
- runtime selector / inference usage
- champion / challenger comparison
- paper/sim evaluation
- possible deployment

## 1. Core principle

Training artifact and deployment artifact are separate.

A policy artifact must be:
- versioned
- identifiable
- comparable
- reproducible
- consumable by runtime selector / inference

The executor must not be asked to interpret raw model internals.

## 2. Artifact families

### A. Training artifact
Contains whatever is needed to reproduce or continue offline work.

May include:
- checkpoints
- optimizer state
- trainer metadata
- extra diagnostics

### B. Deployment / inference artifact
Contains what runtime selector / inference needs to score or decide.

Must not require training code paths.

## 3. Mandatory top-level identity

Every policy artifact must contain:

- `policy_id`
- `artifact_version`
- `policy_family`
- `training_snapshot_id`
- `training_config_hash`
- `code_commit_hash`
- `reward_version`
- `evaluation_surface_id`

## 4. Policy ownership

Every policy artifact must declare:

- `target_asset`
- `allowed_venues`
- `allowed_action_family`
- `required_context` if relevant

Cross-symbol context may be used in inference,
but action ownership is always single-asset.

## 5. Payload separation

Policy artifact is divided into:

### A. `policy_payload`
Opaque model or rule payload.

May be:
- model weights reference
- serialized parameters
- compiled runtime payload
- structured rule payload

### B. `runtime_metadata`
Non-opaque metadata for runtime selection and safety.

Must include enough information to:
- decide compatibility
- decide applicability
- compare candidates
- gate invalid deployment

## 6. Mandatory runtime metadata

Runtime metadata must include:

- `target_asset`
- `allowed_venues`
- `action_space_version`
- `required_streams`
- `required_field_families`
- `required_scale_preset`
- `observation_schema_version`
- `reward_version`
- `policy_state_requirements`
- `expected_return_score`
- `risk_score`
- `turnover_score`
- `confidence_or_quality_score`
- `min_capital_requirement`
- `size_bounds`
- `leverage_bounds`
- `artifact_compatibility_tags`

New artifacts must also carry a structured strict runtime contract inside `runtime_metadata`.

That strict contract must include:
- `runtime_contract_version`
- `policy_kind`
- `required_scale_specs`
- `required_raw_surface_shapes`
- `derived_contract_version`
- `derived_channel_templates`
- `derived_channel_template_signature`
- `expected_feature_dim`

## 7. Compatibility requirements

A policy artifact must be rejected by runtime selector if incompatible with:
- current observation schema
- current reward interpretation where relevant
- available streams or fields
- allowed venue set
- runtime adapter
- required policy state

Runtime rejection must be explicit for at least:
- observation schema version mismatch
- scale-spec mismatch
- raw tensor shape mismatch
- derived contract/version mismatch
- derived channel identity/order mismatch
- feature-dimension mismatch
- deprecated or unsupported runtime adapter

Compatibility must be explicit, not guessed.

## 8. Versioning

Every policy artifact must carry:
- schema version
- payload format version
- compatibility tags
- optional digest / checksum

Version strings without enforcement are insufficient.

For new artifacts, missing strict runtime-contract metadata is itself a runtime rejection condition.
Legacy acceptance, if any, must be explicit, version-bounded, logged, and deprecated rather than silent.

## 9. Runtime selector relation

The runtime selector consumes policy artifacts and produces final decision candidates.

The selector may use:
- one policy
- many policies
- ensemble logic
- ranking logic
- arbitration logic

But the executor must receive only final executable intent, not raw artifact internals.

## 10. Executor-facing output

The executor should receive a narrowed execution intent, such as:

- target asset
- venue
- action
- size / notional
- leverage
- ttl / validity window
- confidence if needed
- trace id / policy id

The executor must not need to parse arbitrary training payloads.

## 11. Required lineage hooks

Every artifact must support lineage through:
- parent artifact id if applicable
- training run id
- evaluation report linkage
- paper/sim report linkage

## 12. Prohibited artifact failures

Forbidden:
- unversioned payload
- payload with no compatibility tags
- deployment artifact that requires training code path
- executor-facing artifact that hides venue or asset ownership
- policy artifact that cannot be linked back to training/evaluation evidence
- new deployment artifact with no strict runtime contract
- silent legacy fallback when compatibility must be inferred
