---
name: quantlab-governance
description: Use this skill whenever a task can affect market-scope truth, exchange-aware canonicalization, online feature state, offline/online parity, runtime inference safety, evaluation integrity, or live execution behavior in the QuantLab multi-exchange futures ML system.
---

# QuantLab Governance Skill

Use this skill before proposing changes in any governance-sensitive area.

## Activate this skill when

Activate when the task touches:

- exchanges, symbols, or stream-family scope,
- canonical event parsing or normalization,
- unsupported/missing/stale semantics,
- online feature/state construction,
- replay or state rebuild logic,
- reward or evaluation logic,
- policy artifacts or registry lineage,
- runtime inference behavior,
- live execution handoff, order safety, or kill-switch behavior,
- recovery, reconnect, or observability on the live path.

If the task can change what the model sees, what the model means, or what live trading does, activate this skill.

---

## Required checks

### 1. Market-scope integrity

Check whether the task preserves the declared market scope:

- 3 venues
- 10 symbols
- 5 canonical stream families
- sparse venue availability

Block or flag any task that silently widens or distorts the market scope.

### 2. Canonical surface integrity

Check whether the task preserves:

- exchange identity
- symbol identity
- stream-family identity
- explicit unsupported vs missing vs stale semantics
- event meaning across offline and online paths

Block the task if unsupported inputs can be confused with zeros, nulls, or stale values.

### 3. Offline/online parity

Assume parity risk until disproven.

Check:

- identical feature semantics between replay and runtime,
- same treatment of missing/stale/unsupported inputs,
- same aggregation or state update rules,
- same normalization assumptions when relevant,
- replay equivalence or state rebuild evidence.

Any live-path change without parity evidence is a blocker.

### 4. Time-ordering and recovery discipline

Check:

- event-time ordering policy,
- out-of-order handling,
- deduplication,
- idempotency,
- reconnect behavior,
- recovery or warm-start behavior,
- stale-state policy.

If the task changes runtime state without explicit recovery semantics, flag it.

### 5. Evaluation integrity

Check:

- leakage discipline,
- walk-forward integrity,
- purge/embargo correctness,
- untouched test protection,
- search-budget transparency,
- reward parity between training and evaluation.

Any weakening is a blocker unless the task exists to repair it.

### 6. Runtime and execution safety

Check:

- runtime consumes declared inputs only,
- executor remains thin,
- no hidden strategy logic is added downstream,
- live safety actions are explicit,
- venue-specific costs and feasibility remain explicit where needed.

If stale or partial inputs can silently produce live actions, treat as blocker or high-risk finding.

### 7. Commercial relevance

State the primary business effect:

- `expected_edge`
- `parity_integrity`
- `capital_protection`
- `latency_freshness_safety`
- `research_throughput`
- `continuity_debt_retirement`
- `docs_hygiene_only`

If a task expands complexity but does not plausibly improve edge, parity, safety, or throughput, downgrade priority.

---

## Blocker conditions

Classify as blocker if the task would:

- weaken parity between offline and runtime,
- allow unsupported streams to masquerade as real values,
- allow stale state to drive live action without explicit policy,
- blur the inference/executor boundary,
- change evaluation truth without versioned explanation,
- remove venue identity where economics still depend on venue,
- create live behavior that cannot be reconstructed from artifacts and logs.

---

## Required answer format

When active, answer in this structure:

1. `task_classification`
   - layer
   - business effect
   - execution mode
   - risk focus

2. `rules_touched`
   - relevant documents and contracts

3. `findings`
   - blocker vs non-blocker
   - exact failure mode
   - parity / safety / economic consequence

4. `required_evidence`
   - tests
   - replay checks
   - runtime checks
   - what is still unproven

5. `recommendation`
   - proceed / proceed with guardrails / do not proceed yet
   - smallest safe next step

Be concrete.  
Do not give generic approval language.
