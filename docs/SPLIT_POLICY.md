# Split Policy

## Purpose

This document defines the canonical split discipline for QuantLab.

Its goal is to prevent:
- leakage
- backtest overfitting
- hidden train/test contamination
- false confidence from time-inappropriate validation

## 1. Core principle

QuantLab uses time-ordered evaluation.

Random split is forbidden.

## 2. Default policy

Default split policy:
- custom walk-forward

The exact fold-generation algorithm must be deterministic and versioned.

## 3. Why custom walk-forward

QuantLab data may be:
- event-time based
- irregularly spaced
- cross-symbol and cross-exchange
- multi-horizon
- overlap-sensitive

Therefore, generic CV helpers are not the constitutional default.

## 4. `TimeSeriesSplit` usage rule

`TimeSeriesSplit` may be used only:
- on suitable equally spaced surfaces,
- as a helper,
- when it does not violate overlap or causality constraints.

It is not the canonical default.

## 5. Purge + embargo

If label windows, holding horizons, or information sets overlap:
- purge is mandatory
- embargo is mandatory

These widths must be derived from the information horizon and documented.

## 6. Fold requirements

Every fold definition must explicitly specify:
- train start/end
- validation start/end if present
- test start/end
- purge width
- embargo width
- horizon assumptions
- overlap assumptions

## 7. Final untouched test

A final untouched test set must exist.

It may not be reused for:
- tuning
- architecture search
- policy selection
- champion decisions
- repeated model comparison

## 8. Multi-horizon rule

If multiple horizons are used, split generation must respect the maximum information reach.

A shorter horizon does not override longer overlap risk.

## 9. Cross-symbol / cross-exchange rule

Shared information across symbols or exchanges must be treated as part of the same time-dependent information set.

A split is invalid if it pretends those information sets are independent when they are not.

## 10. Split artifacts

Every official training/evaluation run must record:
- split policy version
- fold generation config
- fold boundaries
- purge / embargo widths
- final untouched test identifier
