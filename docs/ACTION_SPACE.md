# Action Space

## Purpose

This document defines the canonical action-space semantics for QuantLab.

It covers:
- action ownership
- inventory awareness
- feasibility structure
- venue/size/leverage dimensions
- no-trade semantics

## 1. Core principle

QuantLab policies are inventory-aware and single-asset in action ownership.

A policy acts on one target asset at a time, while using possibly broader context.

## 2. Canonical action identity

At minimum, action families must support:
- `abstain`
- `enter_long`
- `enter_short`
- `hold`
- `exit`

Implementations may refine these further, but these semantics must remain clear.

## 3. Inventory awareness

The action space is inventory-aware.

Action semantics may depend on:
- previous position side
- previous venue
- hold age
- turnover accumulator

Stateless-action design is not sufficient as the canonical policy surface.

## 4. Feasibility structure

Action feasibility must be represented separately from observation.

Canonical feasibility dimensions are:

`action × venue × size_band × leverage_band`

Each coordinate must represent:
- `feasible: bool`
- `reason: str` or structured reason code

## 5. No-trade rule

`abstain` is always allowed unless the system explicitly enters a non-standard operational mode.

No-trade must remain a first-class action, not an implicit absence of action.

## 6. Decision-time rule

Feasibility must be constructed using decision-time-known information only.

Forbidden:
- future-aware feasibility
- next-price-dependent feasibility
- next-timestamp-dependent feasibility

## 7. Venue semantics

Venue is part of the action space, not hidden in reward-only logic.

If venue matters for execution, it must be explicitly represented in action or feasibility semantics.

## 8. Size and leverage semantics

Size and leverage may be banded in v1.

At minimum, size/leverage logic must remain:
- explicit
- versioned
- bounded
- compatible with executor and capital-allocation rules

## 9. Compatibility rule

Any compatibility adapter that collapses the action space must be clearly marked as non-canonical.

The canonical action space must remain:
- inventory-aware
- venue-aware
- feasibility-aware
