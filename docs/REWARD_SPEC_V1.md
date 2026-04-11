# Reward Specification V1

## Purpose

This document freezes the official v1 reward formula for QuantLab.

This is the canonical implementation target for v1.
If code disagrees with this document, code is wrong.

## 1. Governing formula

For a single evaluated action at decision time `t` and selected venue `v`:

`reward_v1 = gross_return - fee_cost - slippage_cost - risk_penalty - turnover_penalty + funding_component - infeasible_penalty_if_applicable`

If action is `abstain`, then:
- `gross_return = 0`
- `fee_cost = 0`
- `slippage_cost = 0`
- `turnover_penalty = 0`
- `funding_component = 0`
- `risk_penalty = 0`
unless a non-standard policy-state rule explicitly says otherwise.

## 2. Gross return

For directional actions:

`gross_return = direction_sign * ((p_t+h - p_t) / p_t)`

Where:
- `direction_sign = +1` for long
- `direction_sign = -1` for short
- `p_t` is the selected venue reference price at decision time
- `p_t+h` is the selected venue replay reference price at horizon end

V1 uses horizon-end reference price only.
Intermediate path is retained in timeline but not directly integrated into gross return in v1.

## 3. Venue selection

Reward is venue-aware.

The selected venue must be explicit in reward context.
No exchange averaging is allowed in v1 reward.

If no venue is selected, reward evaluation is invalid.
There is no canonical `venue=None` fallback in v1.

## 4. Fee cost

V1 fee cost is applied as:

`fee_cost = fee_bps / 10000`

Fee regime is venue-specific.

If the evaluation surface does not distinguish maker/taker, use the configured venue fee proxy directly.

## 5. Slippage cost

V1 slippage cost is applied as:

`slippage_cost = slippage_proxy_bps / 10000`

Slippage proxy is venue-specific.
If no richer model is available, use configured venue-level proxy.

## 6. Funding component

Funding applies only if relevant to the selected venue and instrument.

V1 funding term:

`funding_component = - funding_rate_effective`

Where `funding_rate_effective` must be:
- taken from reward context,
- freshness-aware,
- zeroed or marked invalid according to implementation policy if stale beyond configured freshness threshold.

V1 default:
- if funding freshness exceeds threshold, funding contribution is set to `0`
- stale funding must still be represented in diagnostics

## 7. Risk penalty

V1 risk penalty is:

`risk_penalty = risk_aversion_lambda * abs(gross_return)`

This is a simple symmetric penalty.
It is acknowledged as a coarse v1 proxy, not the final economic truth.

## 8. Turnover penalty

V1 turnover penalty is:

`turnover_penalty = turnover_lambda * turnover_event`

Where:
- `turnover_event = 1` if the evaluated decision changes exposure state
- `turnover_event = 0` otherwise

Exposure change includes:
- flat -> long
- flat -> short
- long -> flat
- short -> flat
- long -> short
- short -> long

Hold without state change has zero turnover penalty in v1.

## 9. Infeasible action penalty

V1 infeasible-action semantic is:

- force abstain
- apply infeasible penalty

V1 formula:

`infeasible_penalty_if_applicable = infeasible_penalty_lambda if action infeasible else 0`

This is mandatory for v1.

## 10. No-trade semantics

`abstain` is a valid first-class action.

V1 reward for abstain is exactly zero unless infeasible-action semantics or explicit non-standard position carry rule says otherwise.

## 11. Official v1 parameter names

The canonical v1 parameters are:

- `fee_bps`
- `slippage_proxy_bps`
- `risk_aversion_lambda`
- `turnover_lambda`
- `infeasible_penalty_lambda`
- `funding_freshness_threshold_seconds`
- `reward_horizon_steps`

## 12. Versioning

The official reward version string for this document is:

`reward_v1`

Any code path using different math must use a different version id.
