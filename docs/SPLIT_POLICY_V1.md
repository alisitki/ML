# Split Policy V1

## Purpose

This document freezes the official v1 split-generation policy for QuantLab.

If code disagrees with this document, code is wrong.

## 1. Official v1 split version

The official split version string is:

`split_v1_walkforward`

## 2. Canonical structure

V1 uses:

- one train segment
- one validation segment
- one final untouched test segment

Within development iterations, walk-forward folds are generated inside the train+validation region.
The final untouched test segment remains untouched until final reporting.

## 3. Canonical temporal order

`train -> validation -> final_untouched_test`

No fold may violate this order.

## 4. Fold generation rule

Within the development region:

- each fold train window must end strictly before its validation window begins
- each fold validation window must end strictly before the next fold’s future region
- fold generation must be deterministic

## 5. Purge width rule

V1 purge width is:

`purge_width = max_label_horizon`

Where `max_label_horizon` is the maximum information reach, in time units or steps, used by:
- reward horizon
- holding horizon
- any target construction
- any future-dependent evaluation target

## 6. Embargo width rule

V1 embargo width is:

`embargo_width = max_label_horizon`

This may be increased in future versions, but not reduced below the maximum information reach in v1.

## 7. Final untouched test rule

The final untouched test segment:
- may be run only after development selection is complete
- may not be used for tuning
- may not be used for model or reward selection
- may not be used for champion decision exploration loops

It is for final evidence only.

## 8. Cross-symbol / cross-exchange treatment

All symbols and exchanges participating in the same timeline are treated as sharing one time-dependent information surface.

Split generation may not pretend they are independent independent samples.

## 9. V1 operational expectation

Code implementing v1 split policy must produce and store:
- fold boundaries
- purge width
- embargo width
- final untouched test boundary
- split version id
