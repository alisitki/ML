# Task Template

## 1. Task classification

### Layer
Choose one primary layer:
- `data_plane`
- `canonicalization`
- `observation_surface`
- `offline_training`
- `evaluation`
- `registry_artifacts`
- `runtime_inference`
- `executor_risk`
- `docs_only`

### Business effect
Choose one:
- `expected_edge`
- `parity_integrity`
- `capital_protection`
- `latency_freshness_safety`
- `research_throughput`
- `docs_hygiene_only`

### Execution mode
Choose one:
- `smoke_debug`
- `continuity_baseline`
- `real_training`
- `runtime_live_path`

## 2. Task restatement

Restate the task in one paragraph.

## 3. Why now

State:
- what failure mode it reduces,
- what business value it improves,
- why it should happen now.

## 4. Governing docs

List the relevant:
- strategy docs
- constitution / runtime boundary
- contracts
- runbooks
- state / backlog / decisions

## 5. Main risks

Call out explicitly:
- leakage risk
- overfitting risk
- offline/online parity risk
- venue semantic drift
- unsupported-vs-missing misuse
- runtime compatibility drift
- selector/executor boundary drift
- capital risk / live path safety risk

## 6. Smallest valid plan

Give the smallest plan that solves the task without widening scope.

## 7. Files likely touched

List the likely files.

## 8. Definition of done

State:
- tests required
- docs required
- state updates required
- evidence scope
- remaining unverified assumptions

## 9. Explicit non-goals

State what this task must not turn into.
