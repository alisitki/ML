# QL-033 R5 Remote Rerun - 2026-04-24

## Verdict

`QL-033 R5` did not pass proof-slice acceptance. The latest retained attempt is
classified as `RUN_CONTROL_FAILURE` for acceptance purposes, not as a successful
performance result.

`Phase C1` remains closed. This evidence does not claim offline closure PASS, model
adaptation readiness, runtime readiness, or live readiness.

Classification:

```text
task_phase=current-phase hardening
layer=offline_training
business_effect=research_throughput
execution_mode=continuity_baseline
risk_focus=[replay_mismatch, leakage, venue_semantic_drift]
```

Semantic contract, selector hyperparameters, quotas, tier meanings, BBO precedence,
replay ordering, row alignment, artifact schema, cadence, proof slice, and
venue/symbol/stream identity were not intentionally changed.

## Code Scope

R5 added equivalence-first tests for the event-token window-base/BBO path:

- ordered raw and deduped candidate identity
- ordered informative-unit identity
- source bucket, salience, emission tags, matched reasons, canonical BBO reason
- latest-reference identity and duplicate/candidate counters
- same-timestamp and source-order inversion behavior
- no-leakage behavior for future events and burst metadata
- BBO burst clipping and reason precedence cases
- venue/symbol/source identity preservation in BBO-like payloads

The R5 implementation attempts were intentionally narrow:

- no T4 nearest-anchor redesign
- no selector hyperparameter, quota, tier, or BBO precedence changes
- no artifact schema or manifest/shard schema changes
- no runtime/live-path changes
- no venue/symbol/stream identity reduction

After the remote failures, the unaccepted eager precompute, k-way merge, and global
BBO cache hot-path changes were not left enabled as production behavior. The retained
code keeps the slow reference and ordered-equivalence test matrix so the next pass can
prove any replacement before enabling it.

## Local Verification

Local checks passed before remote reruns:

```text
pytest tests/test_event_token_cache.py tests/test_ql033_r4_validator.py
15 passed

pytest tests/test_streaming_trajectory.py::TestBuildToDirectory::test_build_writes_event_token_cache_manifest_and_aligned_shards tests/test_archive_run_bundle.py tests/test_prune_local_outputs.py tests/test_ql033_r4_validator.py
12 passed

ruff check src/quantlab_ml/trajectories/event_token_cache.py tests/test_event_token_cache.py scripts/validate_ql033_r4.py tests/test_ql033_r4_validator.py
pass

git diff --check
pass
```

## Proof Slice

Same proof slice as R4:

- time: `2026-01-25T16:00:00Z` to `2026-01-25T23:59:00Z`
- symbols: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`
- venues: `binance`, `bybit`, `okx`
- streams: `trade+bbo`
- cadence: `60s`
- pinned baseline import proof: `bdb1bb4fadd6685e6eb5fb5f65beebc73034da21`

Remote instance reused:

```text
provider=vast.ai
instance_id=35509817
gpu=RTX 5090
cpu_cores_effective=48.0
disk_gb=500
ssh_host=81.183.231.113
ssh_port=48851
```

## Remote Attempts

### Attempt 0

Root:

```text
/workspace/runs/ql033-r5-windowbase-20260424
```

Archive:

```text
s3://quantlab-archive/quantlab/remote-runs/ql033-r5-windowbase-20260424/
```

Thin mirror:

```text
outputs/ql033-r5-windowbase-20260424-thin
```

Outcome:

- baseline real: `966.73s`
- gate: `2176s`
- run_a real: `1290.78s`
- run_a exit: `137`
- run_b: not started
- full `event_token_cache_v1` manifest/shards: missing
- partial selector profile: missing
- archive verification: pass, verified at `2026-04-24T08:19:01Z`
- remote prune: pass, `767.5M` removed

Interpretation:

The eager full-split lane/dedupe/BBO precompute path was rejected. It consumed memory
before event-cache partial profiling could be written and exited before producing
semantic payload evidence.

### Rerun 1

Root:

```text
/workspace/runs/ql033-r5-windowbase-20260424-rerun1
```

Archive:

```text
s3://quantlab-archive/quantlab/remote-runs/ql033-r5-windowbase-20260424-rerun1/
```

Thin mirror:

```text
outputs/ql033-r5-windowbase-20260424-rerun1-thin
```

Outcome:

- baseline real: `991.45s`
- gate: `2231s`
- run_a real: `1071.86s`
- run_a exit: `143`
- run_b: not started
- full `event_token_cache_v1` manifest/shards: missing
- archive verification: pass, verified at `2026-04-24T08:57:33Z`
- remote prune: pass

Partial development profile at stop:

```text
rows_processed=15
last_decision_time=2026-01-25T16:14:00+00:00
window_base_precompute_wall_sec=58.31416194385383
bbo_significance_wall_sec=25.498060200712644
t4_resolution_wall_sec=0.6676986159291118
per_split_total_selector_wall_sec=68.18449795909692
window_base_cache_miss_count=15
window_base_cache_hit_count=0
```

Tracked classification:

```text
window_base_miss_path
```

Interpretation:

The deterministic k-way merge path remained non-accepted. Its 15-row partial profile
was useful only as hotspot localization evidence, not as proof-slice acceptance.

### Rerun 2

Root:

```text
/workspace/runs/ql033-r5-windowbase-20260424-rerun2
```

Archive:

```text
s3://quantlab-archive/quantlab/remote-runs/ql033-r5-windowbase-20260424-rerun2/
```

Thin mirror:

```text
outputs/ql033-r5-windowbase-20260424-rerun2-thin
```

Outcome:

- baseline real: `980.67s`
- gate: `2207s`
- run_a real: `1091.23s`
- run_a exit: `143`
- run_b: not started
- full `event_token_cache_v1` manifest/shards: missing
- archive verification: pass, verified at `2026-04-24T09:35:27Z`
- remote prune: pass

Partial development profile at stop:

```text
rows_processed=19
last_decision_time=2026-01-25T16:18:00+00:00
window_base_precompute_wall_sec=66.36563651205506
bbo_significance_wall_sec=27.749766912544146
t4_resolution_wall_sec=1.172531544812955
per_split_total_selector_wall_sec=78.72318105911836
window_base_cache_miss_count=19
window_base_cache_hit_count=0
```

Tracked classification:

```text
window_base_miss_path
```

Interpretation:

`run_a` exited with `143` (`SIGTERM`) after `1091.23s`, below the `2207s` build gate.
The retained evidence does not prove a normal build-gate timeout, a manual
interruption, or an external resource issue. Because only 19 rows were processed,
`run_a` did not produce full manifest/shards, and `run_b` did not start, this attempt
is `RUN_CONTROL_FAILURE` for acceptance. The partial selector profile may only
localize the next R6 target to the first-pass `window_base`/BBO miss-path.

## Acceptance Status

R5 failed required acceptance:

- `run_a` full `event_token_cache_v1` manifest/shards: fail
- `run_b` full `event_token_cache_v1` manifest/shards: fail, not started
- row/shard alignment: fail, not available
- semantic payload tree hash match: fail, not available
- retention/adjacency/starvation thresholds: fail, not available
- latest `run_a` exit reason: `143` / `SIGTERM`, below the build gate, so acceptance
  classification is `RUN_CONTROL_FAILURE`
- incomplete build-time and artifact-size multipliers are not pass evidence
- archive upload and receipt verification: pass for all three attempts
- thin mirror and remote prune: pass for all three attempts

## Next Blocker

The next R6 target is localized to `window_base_miss_path` / BBO-local cost by the
partial profiles. That localization is not acceptance evidence and must not be read
as R5 performance PASS.

The next narrow pass should not broaden into Phase C, model work, runtime work, or a
general selector refactor. It should start from the archived R5 evidence and use a
lower-risk implementation shape:

- keep the proven global-sort order unless an alternative is both ordered-equivalent
  and faster on a representative local/remote micro-slice
- remove global per-event BBO assessment caches from the hot path
- measure and target the dominant `window_base_miss_path` sub-costs; BBO-local
  optimization is acceptable only if it projects enough improvement under the `2.25x`
  gate
- if optimizing BBO, compute burst-start BBO tuple once per window-local burst and
  current BBO tuple once per candidate without retaining unbounded split-level state
- keep the slow reference path and ordered equivalence tests
- run a representative micro-profile over the first-pass miss-only region, roughly the
  first 359 miss rows plus the first cache-reuse transition when feasible, before any
  full paid proof rerun

## R6 Candidate Micro-Profile Status

Current `HEAD` now includes a narrow R6 candidate implementation for
`window_base_miss_path` / BBO-local work:

- the slow reference path remains available
- deterministic global-sort ordering and dedupe semantics are unchanged
- R4 T4 nearest-anchor behavior is unchanged
- BBO burst-start tuple extraction is reused only within the current window-local
  burst
- no global per-event BBO cache was added

Local continuity smoke profile:

```text
outputs/analysis/ql033-r6-windowbase-micro-profile/ql033_r6_window_base_micro_profile.synthetic.json
```

Validation report:

```text
outputs/analysis/ql033-r6-windowbase-micro-profile/ql033_r6_window_base_micro_profile.validation.json
```

The smoke profile covered `360` synthetic first-pass miss rows plus `360` cache-reuse
rows over the proof-slice-shaped market scope and produced ordered reference/candidate
equivalence. It is not representative proof-slice market-data evidence because the
local QL-033 mirrors retain manifests and diagnostics only; event shard payloads are
absent. The micro-profile validator therefore classifies it as
`insufficient_evidence`, blocked by missing first-pass miss-region evidence and absent
projected build-gate multiplier. Full remote R6 proof remains blocked pending a
representative micro-profile on the real problematic development split region.
