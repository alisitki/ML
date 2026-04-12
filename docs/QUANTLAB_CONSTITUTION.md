# QuantLab Constitution

## 1. System identity

QuantLab is an ML-first policy discovery engine.

Its fixed architecture is:
1. offline training
2. runtime selector / inference
3. thin executor

The executor only does:
- feasibility checks
- capital allocation
- order execution

The executor must never become the hidden strategy brain.

## 2. Policy identity

QuantLab may produce many policies.

Each policy is single-asset in action ownership:
- one policy owns decisions for one target asset at a time

Cross-symbol and cross-exchange context may be used in learning and inference,
but action ownership remains single-asset.

## 3. Split discipline

Random split is forbidden.

Default split policy:
- custom walk-forward

Additional rule:
- if label windows, holding horizons, or information sets overlap,
  purge + embargo is mandatory.

`TimeSeriesSplit` is not the constitutional default.
It may be used only as a helper on suitable equally spaced surfaces.

## 4. Leakage policy

Leakage tolerance is zero.

All learned transforms must fit on train only:
- normalization
- scaling
- PCA
- learned feature transforms
- tokenization
- embeddings
- any learned preprocessing

No future statistics may influence train-time decisions.

## 5. Final untouched test rule

A final untouched test set must exist.

It may not be reused for:
- hyperparameter tuning
- architecture selection
- policy selection
- champion decisions
- repeated exploratory comparison

## 6. Search-budget transparency

Search budget is mandatory and promotion-critical.

The following must be recorded:
- total tried models
- total tried seeds
- total tried architectures
- total tried reward variants
- total tried hyperparameter variants
- total candidate policy count

Only reporting winners is forbidden.

Promotion decisions must explicitly include search-budget summary.

## 7. Governing objective

The governing objective is:

net profit
- risk penalty
- unnecessary trading penalty

Fees, funding, slippage, and realistic execution costs are mandatory.

A policy that looks good before costs is not good.

## 8. Paper / sim before live

Promotion path is fixed:

offline training
-> time-ordered out-of-sample evaluation
-> paper/sim
-> only then live candidate

Direct live promotion is forbidden.

## 9. Champion / challenger regime

Champion / challenger is mandatory.

New policies do not automatically replace the current champion.
Every challenger must be compared against the current champion on the same evaluation surface.

Registry is mandatory.
Old good policy knowledge is not discarded.

## 10. Reproducibility

Bit-for-bit reproducibility across every environment is not required.

Required target:
acceptable-tolerance reproducibility under:
- the same data snapshot
- the same code/config
- the same software/hardware stack

Every official run must record:
- seed
- framework version
- device
- CUDA / cuDNN details if relevant
- data snapshot id

## 11. Artifact discipline

A champion does not exist without artifacts.

Required artifacts:
- data snapshot id
- code commit/hash
- training config
- training artifact
- deployment / inference artifact
- evaluation report
- paper/sim report

## 12. Deployment discipline

Training artifact and deployment artifact are separate.

Runtime uses inference artifacts only.
Live learning is forbidden in the initial operating model.

PyTorch is the default training stack.
Meaningful training defaults to GPU execution when available.
Remote rented GPU compute is the preferred execution target for real training when available.
Local CPU / laptop runs are continuity-only:
- smoke
- debugging
- tiny baseline continuity
- short validation
ONNX and TensorRT may be used for inference acceleration only.
Runtime inference acceleration choices do not define the training execution target.

## 13. Learning-surface discipline

The learning surface must preserve market structure.

Required preservation:
- time
- symbol
- exchange
- stream
- field-level raw information

Do not destroy the edge by:
- exchange averaging
- symbol averaging
- stream scalar collapse
- hidden reduction inside core contracts

Compatibility reductions may exist only in explicit compatibility layers.

## 14. Reward discipline

Reward must remain economic.

Reward may not drift into:
- prediction-only scoring
- pre-cost ranking
- reward definitions that ignore realistic execution assumptions

## 15. Promotion gate

A policy may be promoted only if all are true:
- passes time-ordered OOS evaluation
- passes leakage checks
- has a complete search-budget record
- has positive post-cost governing objective
- beats the current champion on the same surface
- is reproducible within acceptable tolerance

## 16. Explicit prohibitions

The following are forbidden:
- random split
- global normalization fit on full dataset
- test-data-driven model selection
- reporting only the best result
- promoting from one attractive backtest
- ignoring fees / funding / slippage
- artifact-less champion declaration
- silent live learning
- hidden strategy logic inside executor
