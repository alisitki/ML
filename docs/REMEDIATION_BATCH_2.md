# Remediation Batch 2

## 1. Verification summary

- Runtime verification before the patch confirmed that `PolicyRuntimeBridge` only enforced adapter consistency, target-asset ownership, required streams, required field families, scale-label presence, and allowed venues.
- Runtime did not verify observation schema version, full scale-spec equality, raw tensor shapes, derived-surface contract/version, derived channel identity/order, feature dimension agreement, or deprecated runtime-adapter quarantine.
- The active training/export path produced only `linear-policy-v1` artifacts. `momentum-baseline-v1` still existed in code as a deprecated runtime branch, but not as the active training family.
- No serialized legacy artifact fixtures existed in-repo, so legacy behavior was verified by constructing current artifacts and removing the new strict metadata.

## 2. Confirmed current gaps

- New artifacts had no strict runtime contract that bound training-time feature layout to runtime observation layout.
- Runtime could still encounter "wrong artifact + wrong observation" combinations that looked runnable until deep inside feature normalization or model scoring.
- Derived-surface inclusion from Batch 1 increased the cost of silent mismatch because channel identity/order and feature dimension now materially affect inference.
- Legacy support was implicit rather than explicit; deprecated momentum runtime support still sat on the normal dispatch path.

## 3. Applied patches

- Added a structured strict runtime contract under `runtime_metadata` for new artifacts.
- Added a shared runtime-contract helper so trainer and runtime derive the same scale-shape, derived-template, signature, and feature-dimension expectations.
- Updated the active trainer to emit strict runtime metadata and mirror tags for new artifacts.
- Tightened `PolicyRuntimeBridge` to validate strict contracts early and explicitly.
- Added a narrow temporary legacy compat window for deterministic legacy `linear-policy-v1` artifacts only.
- Removed `momentum-baseline-v1` from the normal runtime acceptance path; it is now explicitly rejected.
- Sorted derived channels at builder output so identity/order is deterministic and rejectable.

## 4. Rejection rules added

- Reject if the artifact schema version is unsupported.
- Reject new artifacts that omit `runtime_metadata.strict_runtime_contract`.
- Reject on observation schema version mismatch.
- Reject on scale-spec mismatch.
- Reject on raw tensor shape mismatch.
- Reject on derived-surface contract/version mismatch.
- Reject on derived channel identity/order mismatch.
- Reject on feature-dimension mismatch between artifact payload, runtime contract, and live observation.
- Reject deprecated `momentum-baseline-v1` artifacts from the normal runtime path.
- Reject legacy artifacts unless the missing strict contract can be reconstructed deterministically and unambiguously.

## 5. Tests added/updated

- Added strict runtime-contract tests for:
  - new artifact accept path
  - missing strict metadata rejection
  - scale-spec rejection
  - raw-shape rejection
  - derived contract/version rejection
  - derived channel reorder/missing/extra rejection
  - feature-dimension rejection
  - legacy compat acceptance with warning
  - ambiguous legacy rejection
  - deprecated momentum rejection
- Updated training and serialization tests so new artifacts must round-trip with strict runtime metadata and strict compatibility tags.

## 6. Docs updated

- `docs/PROJECT_STATE.md`
- `docs/BACKLOG.md`
- `docs/DECISIONS.md`
- `docs/POLICY_ARTIFACT_SCHEMA.md`
- `docs/OBSERVATION_SCHEMA.md`

## 7. Remaining risks

- The temporary legacy compat window is still open; that is safer than silent fallback, but it is not the final steady state.
- The canonical production observation preset/config (`1m×8`, `5m×8`, `15m×8`, `60m×12`) is still not wired as an explicit training config.
- Walk-forward folds are still metadata-only in the trainer path, so backtest-overfitting risk remains above the constitutional ideal until fold consumption is resolved.
- At the end of Batch 2, the active trainer remained NumPy-based; D-011 still applied.

## 8. Recommended next small batch

1. Decide whether walk-forward fold consumption should become a dedicated training backlog item before QL-100.
2. Add the canonical production observation preset/config so QL-004 can move closer to done.
3. Plan the retirement of the temporary legacy compat window after legacy artifact stock is refreshed.

## Verification gates

- `.venv/bin/pytest -q tests/test_policy_runtime_contracts.py tests/test_training_loop.py tests/test_v2_serialization.py` → `22 passed`
- `.venv/bin/pytest` → `96 passed`
- `.venv/bin/ruff check .` → passed
- `.venv/bin/mypy src` → passed
- `git diff --check` → clean
