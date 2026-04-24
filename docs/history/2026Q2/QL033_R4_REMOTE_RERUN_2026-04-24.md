# QL-033 R4 Remote Rerun - 2026-04-24

## Scope

`QL-033 R4` was a current-phase offline hardening run for `event_token_cache_v1`.
It did not change the semantic contract, selector hyperparameters, quotas, tier
meanings, BBO precedence, replay ordering, row alignment, artifact schema, or
venue/symbol identity. `Phase C1` remained closed.

Proof slice:

- time range: `2026-01-25T16:00:00Z` to `2026-01-25T23:59:00Z`
- symbols: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`
- venues: `binance`, `bybit`, `okx`
- streams: `trade`, `bbo`
- cadence: `60s`
- pinned baseline commit: `bdb1bb4fadd6685e6eb5fb5f65beebc73034da21`

## Main R4 Attempt

Remote run root:

```text
/workspace/runs/ql033-r4-hotpath-20260424-rerun2
```

Thin local mirror:

```text
outputs/ql033-r4-hotpath-20260424-rerun2-thin
```

Archive prefix:

```text
s3://quantlab-archive/quantlab/remote-runs/ql033-r4-hotpath-20260424-rerun2/
```

Outcome: `FAIL`.

Reason:

```text
run_a_failed_or_exceeded_build_time_gate
```

Timing:

| Stage | real_s | user_s | sys_s | Status |
| --- | ---: | ---: | ---: | --- |
| baseline | 971.28 | 867.11 | 133.03 | pass |
| gate | 2186 | n/a | n/a | ceil(971.28 * 2.25) |
| run_a | 2190.58 | 1996.86 | 224.85 | fail, timeout 124 |
| run_b | n/a | n/a | n/a | not started |

Build-time multiplier:

```text
2190.58 / 971.28 = 2.2553537599868214
```

## Selector Evidence At Failure

Partial selector profile:

```text
run_a/trajectories/event_token_cache_v1/development_partial_selector_profile.json
```

Captured values:

- `partial_split_completion_status`: `in_progress`
- `rows_processed`: `940`
- `raw_candidate_count`: `103639570`
- `post_compression_informative_unit_count`: `4569538`
- `window_base_cache_miss_count`: `359`
- `window_base_cache_hit_count`: `581`
- `window_base_precompute_wall_sec`: `598.7133183603873`
- `t4_resolution_wall_sec`: `20.58997399022337`
- `bbo_significance_wall_sec`: `136.32036141422577`
- `quota_fill_wall_sec`: `11.345391703187488`
- `last_decision_time`: `2026-01-25T19:41:00+00:00`

Tier counts:

- `T0`: `1191127`
- `T1`: `18101`
- `T2`: `234278`
- `T3`: `81840`
- `T4`: `2559582`
- `T5`: `462391`
- `T6`: `486`
- `T7`: `21733`

Interpretation:

- R4 fixed the R3 dominant T4 anchor scan cost. R3 reported `t4_resolution_wall_sec=414.015s` at `362` rows; R4 reached `940` rows with `t4_resolution_wall_sec=20.590s`.
- Cache reuse engaged after the first `359` miss-path rows. R3 reached only `3` cache hits before timeout; R4 reached `581` cache hits.
- The remaining blocker is first-pass target-independent window-base/BBO miss-path cost, not T4 resolution.

## Validation And Retention

Tracked validator output:

```text
validation_report.json
validation_report.md
validation_report_post_archive.json
validation_report_post_archive.md
```

Validator blocked by:

- `run_a_complete`
- `run_b_complete`
- `row_alignment`
- `determinism`
- `build_time_multiplier`
- `truncation_rate`
- `weighted_target_symbol_retained_rate`
- `weighted_burst_retention_rate`
- `cross_venue_ordered_adjacency_rate`
- `trade_to_bbo_ordered_adjacency_rate`
- `symbol_with_zero_retained_tokens_count_p95`

These downstream checks failed because full `event_token_cache_v1` manifest/shards
were not produced before the gate.

Archive and prune:

- archive receipt verification: `verified`
- verified at: `2026-04-24T06:41:58Z`
- uploaded files: `63`
- retained class: `partial`
- remote prune: complete
- pruned files: `34`
- pruned bytes: `915715717`
- remote thin root size after prune: `268K`
- local thin mirror size after post-archive validation report: `288K`

Two earlier short setup failures also followed archive-first handling:

- `ql033-r4-hotpath-20260424`: missing `/usr/bin/time` in the base image
- `ql033-r4-hotpath-20260424-rerun1`: baseline worktree already existed as a Git worktree file

Both were archived under their respective remote-run prefixes before prune. They
were operator-runner failures, not selector evidence. Thin local mirrors are:

- `outputs/ql033-r4-hotpath-20260424-setup-fail-thin`
- `outputs/ql033-r4-hotpath-20260424-rerun1-setup-fail-thin`

## Next Blocker

Do not proceed to `Phase C1`.

Next work should be an `R5` current-phase hardening pass that preserves all R4
semantic constraints and attacks the first-pass miss path:

- sliding/incremental target-independent window-base construction across adjacent decision timestamps
- BBO burst-unit reuse across adjacent windows where canonical burst semantics permit it
- equivalence tests before implementation
- same proof-slice rerun after local validator/tests pass
