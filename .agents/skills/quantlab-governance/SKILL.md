---
name: quantlab-governance
description: Use this skill whenever the task touches learning surface design, reward logic, evaluation, split policy, leakage risk, registry, champion/challenger, policy artifacts, runtime inference architecture, selector boundary, or deployment discipline in the QuantLab ML repo.
---

# QuantLab Governance Skill

Apply these rules before proposing code or architecture changes.

## 1. Protect system identity

QuantLab is an ML-first policy discovery engine.

Fixed architecture:
- offline training
- runtime selector / inference
- thin executor

Do not collapse these layers.
Do not move policy intelligence into the executor.

## 2. Preserve learning-surface truth

Do not allow core contracts to silently reintroduce:
- exchange averaging
- symbol averaging
- stream scalar collapse
- hidden reductions that destroy lead-lag or cross-exchange structure

Compatibility reductions may exist only in explicit compatibility modules.

## 3. Treat evaluation as a safety system

Always check:
- split discipline
- leakage risk
- overlap risk
- final untouched test discipline
- search-budget transparency
- champion / challenger compatibility
- artifact completeness

If a change weakens any of these, call it out explicitly.

## 4. Reward discipline

The governing objective is always:

net profit
- risk penalty
- unnecessary trading penalty

Fees, funding, slippage, and realistic execution assumptions are mandatory.

If reward drifts toward prediction-only scoring, call that out as a design failure.

## 5. Split discipline

Default split:
- custom walk-forward

If horizons or information sets overlap:
- purge + embargo required

Random split is forbidden.

## 6. Leakage discipline

Assume leakage risk until disproven.

Check:
- train-only fit for all learned transforms
- no future-aware masks
- no target leakage in rewards
- no future-aware feasibility logic
- no final test reuse for tuning or champion choice

## 7. Registry / selection discipline

Registry is mandatory.
Champion / challenger is mandatory.

Reject designs that:
- allow unscored champions
- ignore search budget
- rely only on one pretty backtest
- hide the number of tried variants

## 8. Runtime discipline

Runtime uses inference artifacts only.
No silent live learning.

PyTorch is the training stack.
ONNX and TensorRT may be used only for runtime inference acceleration.

## 9. What to produce in answers

When working on governance-sensitive changes, report:
- constitutional or contract rule affected
- exact risk
- code or config location
- blocker vs non-blocker
- what test should prove the fix

Prefer short, concrete findings over vague approval.
