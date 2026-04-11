# Reward Specification

## Purpose

This document defines the canonical economic objective and reward semantics for QuantLab.

This is the mathematical and operational companion to the constitutional statement:

net profit
- risk penalty
- unnecessary trading penalty

## 1. Governing objective

For every candidate policy, the governing objective is:

`objective = net_profit - risk_penalty - unnecessary_trading_penalty`

All terms must be computed under realistic post-cost assumptions.

## 2. Mandatory economic components

Every reward or evaluation path must account for:

- execution-side price reference
- fee cost
- slippage cost
- funding impact where relevant
- risk penalty
- unnecessary trading penalty

Ignoring any of the above is forbidden for official evaluation or promotion.

## 3. Reward levels

QuantLab distinguishes three levels:

### A. Decision-time context
What is known at decision time:
- venue-specific reference
- fee regime
- slippage proxy input
- funding freshness
- previous position state
- turnover state
- hold horizon configuration

### B. Replay timeline
What becomes known over the reward horizon:
- venue-specific reference series
- realized horizon path
- carry/funding path where applicable

### C. Evaluated reward
What is computed from the policy decision against the replay timeline.

## 4. Venue-aware rule

Reward must be venue-aware.

Forbidden:
- exchange-average reward reference in core reward logic
- reward evaluation that hides venue selection

If a venue matters for execution, it must matter for reward.

## 5. Position awareness

The reward contract is inventory-aware.

Reward may depend on:
- previous position side
- hold age
- turnover accumulator
- venue continuity or change

Stateless reward logic is insufficient for official policy evaluation.

## 6. Risk penalty

Risk penalty must exist.

At minimum, the implementation must make explicit:
- the mathematical form
- whether it is symmetric or asymmetric
- whether it is path-aware or snapshot-only
- whether it depends on leverage, volatility, drawdown, or adverse move

Any implementation-specific formula must be versioned.

## 7. Unnecessary trading penalty

Unnecessary trading penalty must exist.

At minimum, the implementation must make explicit:
- whether it is applied per action, per turnover, or per position change
- whether it scales with notional, leverage, or frequency
- whether no-trade is always admissible
- whether churn is penalized independently from fees

Any implementation-specific formula must be versioned.

## 8. Funding treatment

Funding must be freshness-aware.

Rules:
- stale funding must not silently behave as current funding
- funding freshness must be represented in reward context
- funding logic must be documented and versioned

## 9. Slippage treatment

Slippage may begin as a proxy model, but it must be explicit.

The implementation must state:
- how slippage is estimated
- whether it is venue-specific
- whether it depends on liquidity, spread, or size
- whether it is constant or adaptive

## 10. Fee treatment

Fees must be explicit and venue-aware.

The implementation must state:
- fee regime
- whether maker/taker distinction exists
- how fee version is tied to the evaluation surface

## 11. Infeasible action semantics

Infeasible actions are not free.

Default semantic:
- force abstain + penalty

This means:
- the trade is not executed,
- but the policy still receives the configured infeasible-action penalty.

Any deviation from this default must be explicit and versioned.

## 12. Parity rule

Training reward path and evaluation reward path must use the same core reward semantics.

Differences between training and evaluation must be explicit and justified.

Silent reward drift between training and evaluation is forbidden.

## 13. Official reward versions

Every official reward implementation must have a version id.

The version record must include:
- fee model version
- slippage model version
- funding treatment version
- risk penalty version
- unnecessary trading penalty version
- infeasible action treatment version

## 14. Prohibited reward shortcuts

Forbidden for official policy ranking:
- pre-cost reward only
- direction-accuracy-only reward
- scalar reward that hides venue choice
- ignoring inventory state where policy state matters
- stale funding treated as fresh without marking
