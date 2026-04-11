# Promotion Gate

A candidate policy or model may be promoted only if every mandatory item below is satisfied.

## A. Split discipline

- [ ] evaluation is time-ordered
- [ ] random split was not used
- [ ] walk-forward protocol is documented
- [ ] purge was applied where overlap exists
- [ ] embargo was applied where overlap exists

## B. Leakage discipline

- [ ] all learned preprocessing fit on train only
- [ ] no future information appears in features
- [ ] no future information appears in masks
- [ ] no future information appears in reward construction
- [ ] no cross-split contamination exists
- [ ] final untouched test was not used for tuning or selection

## C. Search-budget transparency

- [ ] total tried models recorded
- [ ] total tried seeds recorded
- [ ] total tried architectures recorded
- [ ] total tried reward variants recorded
- [ ] total tried hyperparameter variants recorded
- [ ] total candidate policy count recorded
- [ ] promotion report includes search-budget summary

## D. Economic validity

- [ ] objective includes fees
- [ ] objective includes funding
- [ ] objective includes slippage
- [ ] objective includes risk penalty
- [ ] objective includes unnecessary trading penalty
- [ ] post-cost objective is positive
- [ ] result is not driven by unrealistic execution assumptions

## E. Champion comparison

- [ ] challenger was evaluated on the same surface as champion
- [ ] challenger beats champion on governing objective
- [ ] comparison report is attached
- [ ] superiority is not based on one lucky slice only

## F. Reproducibility

- [ ] data snapshot id recorded
- [ ] code commit/hash recorded
- [ ] config recorded
- [ ] seed recorded
- [ ] runtime stack recorded
- [ ] rerun reproduces within acceptable tolerance

## G. Artifact completeness

- [ ] training artifact exists
- [ ] deployment / inference artifact exists
- [ ] evaluation report exists
- [ ] paper/sim report exists
- [ ] registry entry is complete

## H. Runtime discipline

- [ ] runtime path uses inference artifact only
- [ ] no live learning is involved
- [ ] executor remains thin
- [ ] selector / runtime boundary is respected

## Promotion decision

If any mandatory check fails:
- do not promote
- keep the candidate as candidate or challenger
- record the failure reason explicitly
