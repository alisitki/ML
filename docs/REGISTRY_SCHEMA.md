# Registry Schema

## Purpose

This document defines the canonical registry model for QuantLab.

The registry is mandatory.
It is the system of record for:
- policy candidates
- champion / challenger state
- lineage
- evaluation evidence
- paper/sim linkage
- deployment eligibility

## 1. Core principle

If it is not in the registry, it does not exist for promotion purposes.

## 2. Required record types

Registry must support at least:

- candidate policy record
- evaluation record
- paper/sim record linkage
- champion / challenger state
- lineage linkage
- artifact linkage

## 3. Required policy record fields

Every policy record must include:

- `policy_id`
- `artifact_id`
- `status`
- `policy_family`
- `target_asset`
- `training_snapshot_id`
- `training_config_hash`
- `reward_version`
- `evaluation_surface_id`
- `search_budget_summary`
- `created_at`

## 4. Allowed statuses

At minimum:
- `candidate`
- `challenger`
- `champion`
- `rejected`
- `retired`
- `archived`

## 5. Champion / challenger rules

- unscored policy cannot become champion
- champion selection must be evidence-backed
- challenger must be compared against champion on the same evaluation surface
- stale or invalid challengers must not remain indistinguishable from active challengers

## 6. Score history

Registry must store score history, not only the latest score.

At minimum:
- governing objective
- return score
- risk score
- turnover score
- stability score
- applicability score
- evaluation date
- evaluation surface id

Selection logic may use more than the last score, but the history must exist regardless.

## 7. Search-budget linkage

Search-budget summary is mandatory and must include:
- tried models
- tried seeds
- tried architectures
- tried reward variants
- tried hyperparameter variants
- total candidate count

Promotion without this is invalid.

## 8. Coverage / experience metadata

Registry should track:
- train sample count
- eval sample count
- realized trade count
- covered symbols
- covered venues
- covered streams
- active date range

These fields must be clearly documented as either:
- policy-evidence-derived
or
- dataset-surface-derived

They must not pretend stronger evidence than they actually represent.

## 9. Lineage

Lineage must support:
- parent artifact or policy id
- ancestor tracking
- promotion history
- paper/sim linkage
- deployment linkage

Lineage must not stop at only one direct parent if deeper ancestry is available.

## 10. Deployment linkage

A deployable record must link to:
- deployment artifact
- evaluation report
- paper/sim report
- runtime compatibility metadata

## 11. Prohibited registry failures

Forbidden:
- untracked champion
- unscored champion
- artifact-less champion
- missing search-budget record
- missing evaluation-surface linkage
- registry that only stores winners and forgets search history
